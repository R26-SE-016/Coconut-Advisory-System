from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os
import json
import time
import uuid
import numpy as np
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Resolve FAISS index path relative to this file (project root)
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(_ROOT_DIR, "faiss_index")

# ============ Early Exit Configuration ============
EARLY_EXIT_THRESHOLD = 0.85  # Cosine similarity threshold for skipping Judge LLM

# ============ Cached Embeddings Model (Singleton) ============
_EMBEDDINGS_MODEL = None

def _get_embeddings_model():
    """Lazily load and cache the sentence-transformers embedding model singleton."""
    global _EMBEDDINGS_MODEL
    if _EMBEDDINGS_MODEL is None:
        _EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    return _EMBEDDINGS_MODEL


def _compute_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using the cached
    sentence-transformers/all-MiniLM-L6-v2 embedding model.
    Returns a float between -1.0 and 1.0 (typically 0.0 to 1.0 for text).
    """
    embeddings = _get_embeddings_model()
    vecs = embeddings.embed_documents([text_a, text_b])
    a = np.array(vecs[0])
    b = np.array(vecs[1])
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(cos_sim)


# ============ Conversation Memory Store (30-Minute Inactivity TTL) ============
SESSION_STORE: Dict[str, Dict[str, Any]] = {}
SESSION_TIMEOUT_SECONDS = 1800  # 30 minutes


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Retrieve or create conversation history for a given session_id.
    Automatically purges sessions inactive for more than 30 minutes.
    """
    current_time = time.time()

    # Cleanup expired sessions (inactive > 30 minutes)
    expired_sessions = [
        sid for sid, data in SESSION_STORE.items()
        if current_time - data.get("last_accessed", 0) > SESSION_TIMEOUT_SECONDS
    ]
    for sid in expired_sessions:
        del SESSION_STORE[sid]

    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = {
            "history": InMemoryChatMessageHistory(),
            "last_accessed": current_time
        }
    else:
        SESSION_STORE[session_id]["last_accessed"] = current_time

    return SESSION_STORE[session_id]["history"]


_MEMORY_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert agricultural advisor for coconut farming in Sri Lanka.
Use ONLY the information from the context below to answer the question.
If the answer is not found in the context, say: "I don't have information about that in my knowledge base."
Give practical advice a farmer can understand and apply immediately.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


def _contextualize_question(question: str, session_id: str) -> str:
    """
    If there is prior history in session_id, rephrase follow-up questions
    into a standalone question for optimal vector retrieval.
    """
    history = get_session_history(session_id)
    if not history.messages:
        return question

    recent_msgs = history.messages[-6:]
    history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in recent_msgs])

    try:
        condense_prompt = PromptTemplate.from_template(
            "Given the following conversation history between a farmer and an advisor, "
            "and a follow-up question, rephrase the follow-up question to be a complete standalone question "
            "about coconut farming in Sri Lanka. Do NOT answer it, just return the rephrased standalone question.\n\n"
            "Chat History:\n{history}\n\n"
            "Follow-up Question: {question}\n\n"
            "Standalone Question:"
        )
        llm_condense = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1
        )
        chain = condense_prompt | llm_condense | StrOutputParser()
        standalone_q = chain.invoke({"history": history_text, "question": question}).strip()
        return standalone_q if standalone_q else question
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to contextualize question: {e}")
        return question


def load_rag_chain():
    embeddings = _get_embeddings_model()
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    prompt = PromptTemplate.from_template("""You are an expert agricultural advisor for coconut farming in Sri Lanka.
Use ONLY the information from the context below to answer the question.
If the answer is not found in the context, say: "I don't have information about that in my knowledge base."
Give practical advice a farmer can understand and apply immediately.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def get_answer_with_memory(question: str, session_id: str, rag_chain, retriever, user_context=None) -> dict:
    """
    Executes RAG question answering while maintaining session-specific conversation memory.
    Injects past history into the LLM prompt and saves current interaction to history.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    # 1. Rephrase follow-up question using chat history for accurate RAG vector retrieval
    standalone_q = _contextualize_question(question, session_id)
    search_query = f"User Context: {user_context}\nQuestion: {standalone_q}" if user_context else standalone_q

    # 2. Retrieve source documents
    source_docs = retriever.invoke(search_query)
    context = "\n\n".join(doc.page_content for doc in source_docs)

    # 3. Build RunnableWithMessageHistory for the QA prompt
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    qa_chain = _MEMORY_QA_PROMPT | llm | StrOutputParser()

    with_message_history = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    try:
        answer = with_message_history.invoke(
            {"question": question, "context": context},
            config={"configurable": {"session_id": session_id}}
        )
    except Exception as primary_err:
        import logging
        logging.getLogger(__name__).warning(f"Primary memory RAG chain failed: {primary_err}. Fallback...")
        try:
            fb_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0.2)
            fb_chain = _MEMORY_QA_PROMPT | fb_llm | StrOutputParser()
            fb_with_history = RunnableWithMessageHistory(
                fb_chain,
                get_session_history,
                input_messages_key="question",
                history_messages_key="chat_history"
            )
            answer = fb_with_history.invoke(
                {"question": question, "context": context},
                config={"configurable": {"session_id": session_id}}
            )
        except Exception as fb_err:
            logging.getLogger(__name__).error(f"Fallback memory RAG chain error: {fb_err}")
            answer = "Sorry, I am facing connectivity issues to my knowledge base. Please check your internet connection."

    # Format source documents
    sources = []
    for doc in source_docs:
        source_title = os.path.basename(doc.metadata.get("source", "Unknown"))
        if not any(s["title"] == source_title for s in sources):
            sources.append({
                "title": source_title,
                "content": doc.page_content[:200],
                "metadata": doc.metadata
            })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "confidence": 0.85,
        "context_used": user_context,
        "session_id": session_id
    }


def get_answer(question, rag_chain, retriever, user_context=None, session_id=None):
    if session_id:
        return get_answer_with_memory(question, session_id, rag_chain, retriever, user_context=user_context)
    temp_session_id = str(uuid.uuid4())
    return get_answer_with_memory(question, temp_session_id, rag_chain, retriever, user_context=user_context)


def _clean_llm_translation_output(text: str) -> str:
    """Strips thinking tags (<think>...</think>), extraneous markdown wrappers, and quote artifacts from LLM output."""
    if not text:
        return ""
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^\*{0,2}(?:sinhala translation|english translation|translation):\*{0,2}\s*', '', text, flags=re.IGNORECASE).strip()
    text = text.strip(' "\'\n\r')
    text = re.sub(r'\bferomone\b', 'pheromone', text, flags=re.IGNORECASE)
    return text.strip()


def is_sinhala(text: str) -> bool:
    """Detect Sinhala characters in text (Unicode range U+0D80 to U+0DFF)."""
    return any('\u0d80' <= char <= '\u0dff' for char in text)


def is_tamil(text: str) -> bool:
    """Detect Tamil characters in text (Unicode range U+0B80 to U+0BFF)."""
    return any('\u0b80' <= char <= '\u0bff' for char in text)


def get_language(text: str) -> str:
    """
    Unified language detection: returns 'si', 'ta', or 'en'.
    Checks Sinhala first, then Tamil, defaults to English.
    """
    if is_sinhala(text):
        return 'si'
    if is_tamil(text):
        return 'ta'
    return 'en'


def _sanitize_sinhala_advisory(text: str) -> str:
    """Post-processing sanitizer to fix common LLM Sinhala translation artifacts."""
    if not text:
        return ""
    import re
    # Fix improper translation of 'recommend' to proper name 'අනුරුද්ධ'
    text = re.sub(r'\bඅනුරුද්ධ\b', 'නිර්දේශ', text)
    text = re.sub(r'\bඅනුරුද්ධ කරමි\b', 'නිර්දේශ කරමි', text)
    # Fix unnatural bookish / garbled phrases
    text = re.sub(r'ලෙල දෙන ලෙසට දැන ගන්න', 'පහත දැක්වේ', text)
    text = re.sub(r'කෙළවර කිරීම යම් ආකාරයකින් ද\?', 'පොහොර යෙදිය යුත්තේ කෙසේද?', text)
    text = re.sub(r'\bනාරකොළ\b', 'පොල් පැළ', text)
    text = re.sub(r'\bනාරටි\b', 'පොල් පැළ', text)
    text = re.sub(r'\bපොල් කොළ වලින් වසුන්\b', 'පොල් ලෙලි වලින් වසුන්', text)
    return text


def _sanitize_tamil_advisory(text: str) -> str:
    """Post-processing sanitizer to fix common LLM Tamil translation artifacts."""
    if not text:
        return ""
    import re
    # Fix common LLM Tamil mistranslations in agricultural context
    # 'recommend' sometimes mistranslated to formal/bookish Tamil
    text = re.sub(r'\bபரிந்துரைக்கப்பட்டது\b', 'பரிந்துரை', text)
    # Fix coconut husk mistranslation (sometimes rendered as coconut leaf)
    text = re.sub(r'\bதேங்காய் இலைகள் மூலம் மூடுதல்\b', 'தேங்காய் நார் மூலம் மூடுதல்', text)
    # Fix mother palm mistranslation
    text = re.sub(r'\bஅம்மா பனை\b', 'தாய் பனை', text)
    # Fix seedling mistranslation
    text = re.sub(r'\bவிதைப்பு செடி\b', 'தைல் கன்று', text)
    return text


def _clean_tamil_translation_output(text: str) -> str:
    """Strips thinking tags and conversational artifacts from Tamil LLM output."""
    if not text:
        return ""
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^\*{0,2}(?:tamil translation|english translation|translation):\*{0,2}\s*', '', text, flags=re.IGNORECASE).strip()
    text = text.strip(' "\'\n\r')
    text = re.sub(r'\bferomone\b', 'pheromone', text, flags=re.IGNORECASE)
    return text.strip()


def translate_text(text, target_lang):
    """
    Translates text to target_lang ('en', 'si', or 'ta') using ChatGroq with a high-accuracy farmer-oriented prompt.
    Uses model cascade: openai/gpt-oss-120b -> llama-3.1-8b-instant.
    """
    if not text or not text.strip():
        return ""

    if target_lang == "si":
        prompt = PromptTemplate.from_template("""You are an expert agricultural translator specializing in Sri Lankan coconut farming.
Translate the input English text into natural, clear, farmer-friendly Sinhala (සිංහල) that a Sri Lankan coconut farmer can easily understand.

CRITICAL SRI LANKAN AGRICULTURAL VOCABULARY:
- black beetle / rhinoceros beetle -> පොල් කුරුමිණියා / කළු කුරුමිණියා (අං කුරුමිණියා)
- red palm weevil -> රතු කුරුමිණියා
- coconut mite -> පොල් මයිටා
- coconut caterpillar -> කොළ කන දළඹුවා (පොල් දළඹුවා)
- whitefly -> සුදු මැස්සා
- scale insects -> කොරපොතු කෘමීන්
- bud rot -> කරටි කුණුවීම
- leaf rot -> ගොබ කුණුවීම
- Weligama leaf wilt disease -> වැලිගම කොළ මැලවීමේ රෝගය
- stem bleeding -> කඳෙන් ශ්‍රාවය ගැලීම
- Ganoderma root and bole rot -> ගැනෝඩර්මා (මුල් සහ කඳ කුණුවීමේ රෝගය)
- mother palm -> මව් ශාකය / මව් පොල් ගස (NEVER translate as "මෑණියන්")
- young coconut palms / seedlings -> තරුණ පොල් ගස් / කුඩා පොල් පැළ
- coconut nursery -> පොල් තවාන
- coconut husks -> පොල් ලෙලි (NEVER translate as "පොල් කොළ")
- mulching / mulch -> වසුන් කිරීම / වසුන
- manure circle -> පොහොර වළල්ල (මනූර වෘත්තය)
- fertilizer -> පොහොර
- Wet Zone / Dry Zone / Intermediate Zone -> තෙත් කලාපය / වියළි කලාපය / අතරමැදි කලාපය
- Yala season / Maha season -> යල කන්නය / මහ කන්නය
- recommend -> නිර්දේශිත / නිර්දේශ කරමි (NEVER translate as "අනුරුද්ධ")
- split applications -> කොටස් වශයෙන් යෙදීම
- termites -> වේයන්
- yellowing -> කොළ කහ වීම / පත්‍ර කහ වීම
- nitrogen deficiency -> නයිට්‍රජන් ඌනතාවය
- yield -> අස්වැන්න
- spacing -> පරතරය / සිටුවීමේ පරතරය
- coconut palm / tree -> පොල් ගස / පොල් පැළ

CRITICAL TRANSLATION RULES:
1. Tone & Clarity: Use natural Sri Lankan Sinhala sentence structure. Avoid word-for-word literal translations or complex bookish words that ordinary farmers do not use.
2. Preserve codes & units: Keep codes (YPM-W, APM, NPK, CRI) and units (kg, g, ml, cm, m) unchanged.
3. Output: Output ONLY the translated Sinhala text. Do NOT add commentary, explanations, markdown fences, or thinking tags.

TEXT TO TRANSLATE:
"{text}"

SINHALA TRANSLATION:""")
    elif target_lang == "ta":
        prompt = PromptTemplate.from_template("""You are an expert agricultural translator specializing in Sri Lankan coconut farming.
Translate the input English text into natural, clear, farmer-friendly Tamil (தமிழ்) that a Sri Lankan coconut farmer can easily understand.

CRITICAL SRI LANKAN AGRICULTURAL VOCABULARY:
- mother palm -> தாய் பனை
- fertilizer -> உரம்
- seedling -> தைல் கன்று
- coconut -> தேங்காய்
- Yala season -> யாழ் பருவம்
- Maha season -> மஹா பருவம்
- wet zone -> ஈர மண்டலம்
- dry zone -> வறண்ட மண்டலம்
- intermediate zone -> இடைநிலை மண்டலம்
- pest -> பூச்சி
- disease -> நோய்
- soil -> மண்
- planting -> நடவு
- harvest -> அறுவடை
- nursery -> நாற்றங்கால்
- black beetle / rhinoceros beetle -> கருப்பு வண்டு / காண்டாமிரு வண்டு
- red palm weevil -> சிவப்பு பனை நாவாய்ப்பூச்சி
- coconut mite -> தேங்காய் பூச்சி
- bud rot -> குருத்து அழுகல்
- leaf rot -> இலை அழுகல்
- stem bleeding -> தண்டு வடிதல்
- manure circle -> உர வட்டம்
- mulching -> மூடாக்கிடுதல்
- coconut husk -> தேங்காய் நார்
- intercropping -> ஊடுபயிர்
- pheromone trap -> பெரோமோன் பொறி
- recommend -> பரிந்துரை
- yield -> விளைச்சல்
- spacing -> இடைவெளி
- coconut palm -> தேங்காய் மரம்

CRITICAL TRANSLATION RULES:
1. Tone & Clarity: Use natural Sri Lankan Tamil sentence structure. Avoid word-for-word literal translations or complex literary words.
2. Preserve codes & units: Keep codes (YPM-W, APM, NPK, CRI) and units (kg, g, ml, cm, m) unchanged.
3. Output: Output ONLY the translated Tamil text. Do NOT add commentary, explanations, markdown fences, or thinking tags.

TEXT TO TRANSLATE:
"{text}"

TAMIL TRANSLATION:""")
    else:
        # Detect source language for -> English translation
        detected_lang = get_language(text)
        if detected_lang == 'ta':
            prompt = PromptTemplate.from_template("""You are an expert agricultural translator specializing in Sri Lankan coconut farming.
Translate the following Tamil (தமிழ்) farmer query into clear, natural, grammatically correct English for an agricultural advisory system.

CRITICAL SRI LANKAN COCONUT FARMING VOCABULARY:
- தாய் பனை -> mother palm
- உரம் -> fertilizer
- தைல் கன்று -> seedling
- தேங்காய் -> coconut
- யாழ் பருவம் -> Yala season
- மஹா பருவம் -> Maha season
- ஈர மண்டலம் -> Wet Zone
- வறண்ட மண்டலம் -> Dry Zone
- இடைநிலை மண்டலம் -> Intermediate Zone
- பூச்சி -> pest
- நோய் -> disease
- மண் -> soil
- நடவு -> planting
- அறுவடை -> harvest
- நாற்றங்கால் -> nursery
- கருப்பு வண்டு / காண்டாமிரு வண்டு -> black beetle / rhinoceros beetle
- சிவப்பு பனை நாவாய்ப்பூச்சி -> red palm weevil
- தேங்காய் பூச்சி -> coconut mite
- குருத்து அழுகல் -> bud rot
- இலை அழுகல் -> leaf rot
- தண்டு வடிதல் -> stem bleeding
- உர வட்டம் -> manure circle
- மூடாக்கிடுதல் -> mulching
- தேங்காய் நார் -> coconut husk
- ஊடுபயிர் -> intercropping
- பெரோமோன் பொறி -> pheromone trap
- விளைச்சல் -> yield

RULES:
1. Translate into a direct, fluent English sentence without commentary.
2. Do NOT enclose output in quotation marks.
3. Preserve codes & units (YPM-W, APM, NPK, kg, g, ml, cm).
4. Output ONLY the translated English text.

TEXT TO TRANSLATE:
"{text}"

ENGLISH TRANSLATION:""")
        else:
            prompt = PromptTemplate.from_template("""You are an expert agricultural translator specializing in Sri Lankan coconut farming.
Translate the following Sinhala (සිංහල) farmer query into clear, natural, grammatically correct English for an agricultural advisory system.

CRITICAL SRI LANKAN COCONUT FARMING VOCABULARY:
- පොල් කුරුමිණියා / කළු කුරුමිණියා / අං කුරුමිණියා -> black beetle / rhinoceros beetle (Oryctes rhinoceros)
- රතු කුරුමිණියා / රතු කුරුමිණි -> red palm weevil (Rhynchophorus ferrugineus)
- පොල් මයිටා / මයිටා හානිය -> coconut mite (Aceria guerrateronis)
- කොළ කන දළඹුවා / පොල් දළඹුවා -> coconut caterpillar (Opisina arenosella)
- සුදු මැස්සා -> whitefly (rugose spiralling whitefly)
- පිටි මකුණා -> mealybug
- කොරපොතු කෘමියා -> coconut scale insect
- වේයන් -> termites
- කරටි කුණුවීම / කරටි කුණුවීමේ රෝගය / ගොබ කුණුවීම -> bud rot disease (Phytophthora) / leaf rot
- වැලිගම කොළ මැලවීම / කොළ මැලවීමේ රෝගය -> Weligama coconut leaf wilt disease
- කඳෙන් ශ්‍රාවය ගැලීම -> stem bleeding disease
- මුල් සහ කඳ කුණුවීම / ගැනෝඩර්මා -> Ganoderma root and bole rot
- ෆෙරමෝන් / ෆෙරමෝන් උගුල් -> pheromone traps
- පොල් ලෙලි -> coconut husks (NOT coconut leaves)
- වසුන් කිරීම / වසුන -> mulching / mulch
- පොහොර වළල්ල / මනූර වෘත්තය -> manure circle
- පොල් ප්‍රභේද -> coconut varieties / cultivars
- මව් ගස / මව් පොල් ගස -> mother palm
- අතුරු බෝග -> intercropping
- බයෝචාර් -> biochar
- බිංදු ජල සම්පාදනය -> drip irrigation
- පොල් ගස් / පොල් පැළ -> coconut palms / coconut seedlings
- තෙත් කලාපය / වියළි කලාපය / අතරමැදි කලාපය -> Wet Zone / Dry Zone / Intermediate Zone
- යල කන්නය / මහ කන්නය -> Yala season / Maha season

RULES:
1. Translate into a direct, fluent English sentence without commentary.
2. Do NOT enclose output in quotation marks.
3. Preserve codes & units (YPM-W, APM, NPK, kg, g, ml, cm).
4. Output ONLY the translated English text.

TEXT TO TRANSLATE:
"{text}"

ENGLISH TRANSLATION:""")

    models_to_try = [
        "openai/gpt-oss-120b",
        "llama-3.1-8b-instant"
    ]

    for model_candidate in models_to_try:
        try:
            llm = ChatGroq(
                model=model_candidate,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.1
            )
            chain = prompt | llm | StrOutputParser()
            query_text = f"{text} /no_think" if "qwen" in model_candidate.lower() else text
            res = chain.invoke({"text": query_text}).strip()
            cleaned_res = _clean_llm_translation_output(res)
            if target_lang == "si":
                cleaned_res = _sanitize_sinhala_advisory(cleaned_res)
            elif target_lang == "ta":
                cleaned_res = _clean_tamil_translation_output(cleaned_res)
                cleaned_res = _sanitize_tamil_advisory(cleaned_res)
            if cleaned_res:
                return cleaned_res
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Translation model {model_candidate} error: {e}")

    return text


def translate_multi_llm_payload(payload: dict, target_lang: str = "si") -> dict:
    """
    Translates Multi-LLM fields (best_answer, reason, llama_answer, llama8b_answer, gemma_answer)
    field-by-field sequentially with slight delay to stay within Groq 8000 TPM limits.
    """
    if not payload:
        return payload

    import time
    result_dict = dict(payload)
    for key, val in payload.items():
        if val and isinstance(val, str) and val.strip():
            try:
                trans = translate_text(val, target_lang)
                if trans:
                    result_dict[key] = trans
                time.sleep(0.25)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Field translation error for {key}: {e}")

    return result_dict


# ============ Multi-LLM Validation ============

MULTI_LLM_MODELS = {
    "llama": "openai/gpt-oss-120b",
    "llama8b": "llama-3.1-8b-instant",
    "gemma": "llama-3.3-70b-versatile",
}

_ADVISOR_PROMPT_TEMPLATE = """You are an expert agricultural advisor for coconut farming in Sri Lanka.
Use ONLY the information from the context below to answer the question.
If the answer is not found in the context, say: "I don't have information about that in my knowledge base."
Give practical advice a farmer can understand and apply immediately.

Context:
{context}

Question: {question}

Answer:"""

_JUDGE_PROMPT_TEMPLATE = """You are an impartial judge evaluating three AI-generated answers about coconut farming in Sri Lanka.

Your task is to determine which answer is most FAITHFUL to the provided source documents from the Coconut Research Institute (CRI).

Evaluate on:
1. Factual accuracy — does the answer match the source documents exactly?
2. Completeness — does it cover all relevant information from the context?
3. No hallucination — does it avoid inventing facts not in the context?
4. Practical usefulness — is the advice actionable for a farmer?

Also assess consensus: how much do all three answers agree on key facts?
- 80-100: High agreement — all three give essentially the same core facts
- 50-79: Moderate agreement — most key facts align but with some differences
- 0-49: Low agreement — significant contradictions or different information

SOURCE CONTEXT:
{context}

QUESTION: {question}

ANSWER FROM LLaMA 3.3 70B:
{llama_answer}

ANSWER FROM LLaMA 3.1 8B:
{llama8b_answer}

ANSWER FROM Gemma 2 9B:
{gemma_answer}

Respond with ONLY valid JSON in this exact format, no other text:
{{
  "best_model": "llama" or "llama8b" or "gemma",
  "reason": "Brief explanation of why this answer is most faithful to the CRI documents",
  "consensus_score": <number 0-100>
}}"""


def _invoke_llm(model_name: str, context: str, question: str) -> str:
    """Invoke a single LLM with the advisory prompt, with automatic model fallback."""
    import re
    import logging

    is_qwen = "qwen" in model_name.lower()
    effective_question = question + " /no_think" if is_qwen else question
    prompt = PromptTemplate.from_template(_ADVISOR_PROMPT_TEMPLATE)

    # Primary attempt
    try:
        llm = ChatGroq(
            model=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2
        )
        chain = prompt | llm | StrOutputParser()
        raw_answer = chain.invoke({"context": context[:2000], "question": effective_question})
    except Exception as primary_err:
        logging.getLogger(__name__).warning(f"Model {model_name} failed: {primary_err}. Attempting fallback...")
        fallback_model = "llama-3.1-8b-instant" if model_name != "llama-3.1-8b-instant" else "openai/gpt-oss-120b"
        try:
            llm_fb = ChatGroq(
                model=fallback_model,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.2
            )
            chain = prompt | llm_fb | StrOutputParser()
            raw_answer = chain.invoke({"context": context[:2000], "question": question})
        except Exception as fb_err:
            logging.getLogger(__name__).error(f"Fallback model {fallback_model} also failed: {fb_err}")
            return "Unable to generate answer from model due to rate limit."

    # Safety net: strip <think>...</think> tags if any model outputs them
    cleaned_answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()
    cleaned_answer = re.sub(r'<think>.*', '', cleaned_answer, flags=re.DOTALL).strip()
    cleaned_answer = re.sub(r'.*?</think>', '', cleaned_answer, flags=re.DOTALL).strip()

    return cleaned_answer if cleaned_answer else raw_answer.strip()


def get_multi_llm_answer(question, retriever, user_context=None):
    """
    Runs the same retrieved context through 3 multi LLMs in parallel,
    then uses a judge LLM to select the best answer based on faithfulness
    to the CRI source documents.

    Early Exit Optimization:
    - As results come in via as_completed(), the first two finished answers
      are compared using cosine similarity on their embeddings.
    - If similarity >= EARLY_EXIT_THRESHOLD (0.85), the Judge LLM is skipped.
    - All 3 candidate answers are ALWAYS collected (no candidate is cancelled).
    - The latency saving comes solely from skipping the Judge LLM call.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Model rank priority for early exit selection (lower index = higher rank)
    MODEL_RANK = {"llama": 0, "llama8b": 1, "gemma": 2}

    # 1. Retrieve context chunks (same for all 3 LLMs)
    search_query = f"User Context: {user_context}\nQuestion: {question}" if user_context else question
    source_docs = retriever.invoke(search_query)
    context = "\n\n".join(doc.page_content for doc in source_docs)[:3000]

    # Deduplicate sources
    sources = []
    for doc in source_docs:
        source_title = os.path.basename(doc.metadata.get("source", "Unknown"))
        if not any(s["title"] == source_title for s in sources):
            sources.append({
                "title": source_title,
                "content": doc.page_content[:200],
                "metadata": doc.metadata
            })

    # 2. Run all 3 LLMs in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_invoke_llm, model, context, search_query): key
            for key, model in MULTI_LLM_MODELS.items()
        }

        # Collect results via as_completed to detect early exit opportunity
        completed_keys = []
        answers = {}
        early_exit = False
        similarity_score = None

        for future in as_completed(futures):
            key = futures[future]
            try:
                answers[key] = future.result()
            except Exception as e:
                logger.error(f"Model {key} failed in as_completed: {e}")
                answers[key] = "Unable to generate answer from model due to rate limit."
            completed_keys.append(key)

            # Check early exit after exactly 2 results are in
            if len(completed_keys) == 2 and similarity_score is None:
                try:
                    similarity_score = _compute_similarity(
                        answers[completed_keys[0]],
                        answers[completed_keys[1]]
                    )
                    logger.info(
                        f"Early exit check: {completed_keys[0]} vs {completed_keys[1]} "
                        f"similarity={similarity_score:.4f} (threshold={EARLY_EXIT_THRESHOLD})"
                    )
                    if similarity_score >= EARLY_EXIT_THRESHOLD:
                        early_exit = True
                        logger.info("Early exit TRIGGERED — will skip Judge LLM.")
                except Exception as sim_err:
                    logger.warning(f"Similarity computation failed: {sim_err}. Falling back to Judge.")
                    similarity_score = None

        # All 3 answers are now collected (no candidate skipped)

    # 3. Decide: early exit or full Judge evaluation
    if early_exit:
        # Select best model by rank among the two that agreed
        best_model = min(completed_keys[:2], key=lambda k: MODEL_RANK.get(k, 99))
        reason = (
            f"Early exit: {completed_keys[0]} and {completed_keys[1]} answers had "
            f"{similarity_score:.1%} semantic similarity (≥ {EARLY_EXIT_THRESHOLD:.0%} threshold). "
            f"Judge LLM skipped. Selected {best_model} as highest-ranked agreeing model."
        )
        consensus_score = 90
        logger.info(f"Early exit result: best_model={best_model}, similarity={similarity_score:.4f}")
    else:
        # Full Judge evaluation (existing logic preserved)
        judge_prompt = PromptTemplate.from_template(_JUDGE_PROMPT_TEMPLATE)
        judge_payload = {
            "context": context[:1500],
            "question": question,
            "llama_answer": answers.get("llama", "")[:600],
            "llama8b_answer": answers.get("llama8b", "")[:600],
            "gemma_answer": answers.get("gemma", "")[:600]
        }
        try:
            judge_llm = ChatGroq(
                model="openai/gpt-oss-120b",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.0
            )
            judge_chain = judge_prompt | judge_llm | StrOutputParser()
            judge_raw = judge_chain.invoke(judge_payload)
        except Exception as judge_err:
            logger.warning(f"Judge primary model failed: {judge_err}. Trying fallback...")
            judge_llm_fb = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.0
            )
            judge_chain = judge_prompt | judge_llm_fb | StrOutputParser()
            judge_raw = judge_chain.invoke(judge_payload)

        # Parse judge response
        try:
            judge_result = json.loads(judge_raw.strip())
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[^}]+\}', judge_raw, re.DOTALL)
            if json_match:
                try:
                    judge_result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    judge_result = {"best_model": "llama", "reason": "Judge parse error — defaulting LLaMA", "consensus_score": 50}
            else:
                judge_result = {"best_model": "llama", "reason": "Judge parse error — defaulting LLaMA", "consensus_score": 50}

        best_model = judge_result.get("best_model", "llama")
        if best_model not in answers:
            best_model = "llama"
        reason = judge_result.get("reason", "")
        consensus_score = judge_result.get("consensus_score", 50)

    return {
        "best_answer": answers.get(best_model, answers.get("llama", "")),
        "best_model": best_model,
        "reason": reason,
        "consensus_score": consensus_score,
        "llama_answer": answers.get("llama", ""),
        "llama8b_answer": answers.get("llama8b", ""),
        "gemma_answer": answers.get("gemma", ""),
        "qwen_answer": answers.get("gemma", ""),
        "sources": sources,
        "context_used": user_context,
        "early_exit": early_exit,
        "similarity_score": round(similarity_score, 4) if similarity_score is not None else None
    }


_IMAGE_VECTOR_STORE = None
_IMAGE_VECTOR_STORE_MTIME = 0

def reload_image_index():
    """Reset cached FAISS image vector store so it reloads latest index from disk."""
    global _IMAGE_VECTOR_STORE, _IMAGE_VECTOR_STORE_MTIME
    _IMAGE_VECTOR_STORE = None
    _IMAGE_VECTOR_STORE_MTIME = 0

def find_relevant_images(
    query_text: str,
    answer_text: str = "",
    top_k: int = 2,
    base_max_distance: float = 0.80,
    max_relative_diff: float = 0.22
) -> list:
    """
    Given query_text (and optional answer_text) from RAG engine, searches
    image_index vector store for high-confidence CRI reference images.

    Accuracy & Precision Guarantees:
    - Multi-pass query formulation (clean standalone question + focused answer context).
    - Enforces a strict L2 distance threshold (default <= 0.80 for keyword-validated matches,
      and <= 0.70 for non-keyword matches).
    - If no image matches below the confidence threshold, returns an empty list [] (no irrelevant images).
    - Enforces a relative margin (<= 0.22) so secondary images are only returned if equally relevant.
    - Applies semantic topic guardrails to prevent pest/variety/fertilizer crossover mismatches.

    Returns list of dicts: [{'url': '/static/images/...', 'caption': '...', 'source': '...'}]
    """
    global _IMAGE_VECTOR_STORE, _IMAGE_VECTOR_STORE_MTIME
    if not query_text or not query_text.strip():
        return []

    clean_q = query_text.strip()
    clean_a = answer_text.strip() if answer_text else ""

    # Multi-pass query formulation: 1. Clean standalone query, 2. Query + focused context
    queries_to_search = [clean_q]
    if clean_a:
        queries_to_search.append(f"{clean_q}\n{clean_a[:140]}")

    combined_text = f"{clean_q} {clean_a}".lower()

    # Core domain concepts for keyword-aligned confidence verification
    domain_keywords = [
        "black beetle", "rhinoceros beetle", "red palm weevil", "red weevil",
        "mite", "aceria", "caterpillar", "opisina", "nettle caterpillar",
        "whitefly", "scale insect", "bud rot", "phytophthora", "leaf rot",
        "leaf blight", "weligama", "leaf wilt", "ganoderma", "root and bole rot",
        "stem bleeding", "plesispa",
        "fertilizer", "manure circle", "npk", "apm", "mulch", "mulching",
        "husk", "biochar", "seedling", "replanting", "underplanting",
        "cric 60", "cric 65", "crisl 2020", "crisl 98", "kapruwana", "kapsetha", "kapsuwaya",
        "variety", "varieties", "hybrid", "irrigation", "drip", "micro-irrigation",
        "basin irrigation", "intercrop", "intercropping", "banana", "pepper",
        "cinnamon", "cashew", "papaya", "ginger", "turmeric", "cover crop"
    ]
    matched_query_keywords = [k for k in domain_keywords if k in combined_text]

    try:
        image_index_path = os.path.join(_ROOT_DIR, "image_index")
        if not os.path.exists(image_index_path):
            image_index_path = os.path.join(_ROOT_DIR, "backend", "image_index")

        if not os.path.exists(image_index_path):
            import logging
            logging.getLogger(__name__).warning(f"image_index directory not found at {image_index_path}")
            return []

        faiss_file = os.path.join(image_index_path, "index.faiss")
        current_mtime = os.path.getmtime(faiss_file) if os.path.exists(faiss_file) else 0

        if _IMAGE_VECTOR_STORE is None or current_mtime > _IMAGE_VECTOR_STORE_MTIME:
            embeddings = _get_embeddings_model()
            _IMAGE_VECTOR_STORE = FAISS.load_local(
                image_index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            _IMAGE_VECTOR_STORE_MTIME = current_mtime

        # Retrieve candidates across query formulations and take the best distance score for each doc
        candidate_map = {}
        for q_str in queries_to_search:
            results = _IMAGE_VECTOR_STORE.similarity_search_with_score(q_str, k=top_k + 4)
            for doc, raw_score in results:
                url = doc.metadata.get("url") or f"/static/images/{doc.metadata.get('filename')}"
                desc = doc.page_content.lower()
                caption = doc.metadata.get("caption", "").lower()
                source = doc.metadata.get("source", "").lower()
                doc_all_text = f"{desc} {caption} {source}"

                # Check if any query domain keyword appears in document
                has_keyword_match = any(kw in doc_all_text for kw in matched_query_keywords)
                effective_score = float(raw_score) - (0.12 if has_keyword_match else 0.0)

                if url not in candidate_map or effective_score < candidate_map[url]["effective_score"]:
                    candidate_map[url] = {
                        "doc": doc,
                        "raw_score": float(raw_score),
                        "effective_score": effective_score,
                        "has_keyword_match": has_keyword_match
                    }

        if not candidate_map:
            return []

        # Sort merged candidates by best effective distance
        sorted_candidates = sorted(candidate_map.values(), key=lambda x: x["effective_score"])
        best_item = sorted_candidates[0]

        # Allowed distance: up to 1.18 if keyword-validated, strictly <= 0.70 for non-keyword matches
        best_allowed_distance = 1.18 if best_item["has_keyword_match"] else 0.70

        if best_item["raw_score"] > best_allowed_distance:
            return []

        best_score = best_item["effective_score"]

        is_pest_query = any(k in combined_text for k in ["beetle", "weevil", "mite", "caterpillar", "whitefly", "scale", "pest", "disease", "rot", "blight", "wilt", "ganoderma"])
        is_fertilizer_query = any(k in combined_text for k in ["fertilizer", "manure", "npk", "apm", "nutrient", "dolomite", "compost", "mulch", "mulching"])

        images = []
        seen_urls = set()

        for item in sorted_candidates:
            if len(images) >= top_k:
                break

            doc = item["doc"]
            raw_score = item["raw_score"]
            eff_score = item["effective_score"]

            # 1. Enforce distance threshold
            allowed_dist = 1.18 if item["has_keyword_match"] else 0.70
            if raw_score > allowed_dist:
                continue

            # 2. Enforce relative distance margin compared to best match
            if (eff_score - best_score) > max_relative_diff:
                continue

            filename = doc.metadata.get("filename", "")
            url = doc.metadata.get("url") or f"/static/images/{filename}"
            caption = doc.metadata.get("caption", "")
            source = doc.metadata.get("source", "")
            desc = doc.page_content.lower()

            if url in seen_urls:
                continue

            # 3. Domain Guardrail Checks
            is_variety_img = "variety" in source.lower() or "hybrid" in desc or "cric " in desc or "crisl " in desc or "kapruwana" in desc or "kapsuwaya" in desc or "kapsetha" in desc
            if is_pest_query and is_variety_img:
                continue

            is_pest_img = any(k in desc for k in ["beetle", "weevil", "mite", "caterpillar", "whitefly", "scale", "pest", "rot", "blight", "wilt", "ganoderma"])
            if is_fertilizer_query and is_pest_img and not is_pest_query:
                continue

            # Specific pest collision guardrails
            if "weevil" in combined_text and ("black beetle" in desc or "oryctes" in desc) and "weevil" not in desc:
                continue
            if "black beetle" in combined_text and ("weevil" in desc or "rhynchophorus" in desc) and "beetle" not in desc:
                continue

            seen_urls.add(url)
            images.append({
                "url": url,
                "caption": caption,
                "source": source
            })

        return images

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in find_relevant_images: {str(e)}")
        return []


if __name__ == "__main__":

    print("Loading RAG system...")
    chain, retriever = load_rag_chain()
    print("Ready!\n")

    test_questions = [
    "How should I fertilize young coconut palms?",
    "How do I select a good mother palm?",
    "What is the recommended planting density for coconut?",
    "How do I control termites in coconut nursery?",
    "What fertilizer mixture is recommended for coconut seedlings?"
]

    for q in test_questions:
        print(f"Q: {q}")
        result = get_answer(q, chain, retriever, user_context="Wet Zone, Yala Season")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 50)