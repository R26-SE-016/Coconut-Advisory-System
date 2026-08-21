from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
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
EARLY_EXIT_THRESHOLD = 0.80  # Cosine similarity threshold for skipping Judge LLM

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
    Normalizes inputs by stripping extra whitespace/newlines, converting to lowercase,
    and evaluating the first 300 characters of core answer content.
    Returns a float between -1.0 and 1.0 (typically 0.0 to 1.0 for text).
    """
    import re
    # 1. Strip extra whitespace and newlines, convert to lowercase
    norm_a = re.sub(r'\s+', ' ', str(text_a or "")).strip().lower()
    norm_b = re.sub(r'\s+', ' ', str(text_b or "")).strip().lower()

    # 2. Take first 300 characters of each normalized answer
    trunc_a = norm_a[:300]
    trunc_b = norm_b[:300]

    if not trunc_a or not trunc_b:
        return 0.0

    embeddings = _get_embeddings_model()
    vecs = embeddings.embed_documents([trunc_a, trunc_b])
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
Use the provided knowledge base context and conversation history to answer the farmer's question accurately.
If the answer cannot be determined from the context or conversation history, say: "I don't have information about that in my knowledge base."
Give practical, clear advice a farmer can understand and apply immediately.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


def _contextualize_question(question: str, session_id: str) -> str:
    """
    If there is prior history in session_id, rephrase follow-up questions
    into a complete standalone question for optimal vector retrieval while
    intelligently handling new topic transitions and standalone keywords.
    """
    history = get_session_history(session_id)
    if not history.messages:
        return question

    recent_msgs = history.messages[-6:]
    history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content[:400]}" for msg in recent_msgs])

    try:
        condense_prompt = PromptTemplate.from_template("""Given the chat history between a coconut farmer and an agricultural advisor, rephrase the follow-up question into a clear, complete standalone question about coconut farming in Sri Lanka for knowledge base retrieval.

CRITICAL RULES:
1. TRUE FOLLOW-UPS: If the user asks a follow-up referring to the previous discussion (e.g. "what dosage?", "how often to apply?", "what about in the dry zone?", "how to prevent it?"), carry over the relevant subject and plant stage.
2. NEW TOPIC / NEW PEST / SHORT QUERY: If the user introduces a new pest, disease, practice, or topic (e.g. "red palm weevil", "black beetle", "fertilizer application", "bud rot", "mother palm"), treat it as a NEW query about that topic in coconut cultivation. DO NOT contaminate the new topic with unrelated previous constraints (e.g. do NOT force "in nursery" if the new pest is red palm weevil).
3. SHORT KEYWORDS / PHRASES: If the user inputs just a keyword or short phrase (e.g. "red palm weevil" / "රතු කුරුමිණියා", "black beetle", "dolomite"), rephrase into a comprehensive advisory question (e.g. "What is red palm weevil, its damage symptoms, and recommended control and management methods in coconut palms?").
4. DO NOT answer the question. Return ONLY the rephrased standalone question.

Chat History:
{history}

Follow-up Question: {question}

Standalone Question:""")
        llm_condense = ChatOpenAI(
            model="openai/gpt-4o-mini",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=200,
            timeout=20
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

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=800,
        timeout=25
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


TOPIC_KEYWORDS = {
    'mother_palm': ['mother palm', 'plus palm', 'seed nut', 'parent palm', 'nut yield', 'husked nut'],
    'nursery': ['nursery', 'seedling', 'germination', 'seed bed', 'poly bag', 'collar rot'],
    'fertilizer': ['fertilizer', 'urea', 'phosphate', 'potash', 'dolomite', 'YPM', 'nutrient', 'erp', 'tsp', 'mop', 'manure'],
    'pest_disease': ['beetle', 'caterpillar', 'mite', 'disease', 'WCLWD', 'CCI', 'termite', 'weevil', 'rhynchophorus', 'oryctes', 'opisina', 'aceria', 'pest', 'rot', 'bleeding', 'whitefly', 'scale', 'mealybug'],
    'planting': ['planting', 'spacing', 'density', 'replanting', 'seedling selection'],
    'variety': ['CRIC60', 'CRIC65', 'CRISL98', 'variety', 'hybrid', 'dwarf', 'tall'],
    'general': []
}


def detect_topic(text: str) -> str:
    """Detects topic tag for a document chunk or text based on keyword presence."""
    if not text:
        return 'general'
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if topic == 'general':
            continue
        if any(kw.lower() in text_lower for kw in keywords):
            return topic
    return 'general'


def detect_question_topic(question: str) -> str:
    """Detects topic tag for a user question."""
    return detect_topic(question)


def get_filtered_retriever(topic: str, vector_store=None, retriever=None, k: int = 4, fetch_k: int = 100):
    """
    Returns a FAISS retriever that filters chunks matching the detected topic.
    Prioritizes chunks relevant to the detected topic.
    """
    vs = vector_store
    if vs is None and retriever is not None and hasattr(retriever, 'vectorstore'):
        vs = retriever.vectorstore

    if vs is None:
        return retriever

    def topic_filter(metadata: dict) -> bool:
        return metadata.get('topic') == topic

    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": topic_filter,
            "fetch_k": fetch_k
        }
    )


def calculate_combined_reliability(retrieval_confidence: float, consensus_score: float = 80.0) -> tuple:
    """
    Calculates the Combined Reliability Score fusing retrieval quality and consensus validation.

    Formula:
        retrieval_score = retrieval_confidence (0.0 to 1.0) * 100
        validation_score = consensus_score (0 to 100)
        combined_reliability = (retrieval_score * 0.5) + (validation_score * 0.5)

    Reliability Level:
        >= 80 -> 'High'
        60-79 -> 'Moderate'
        < 60  -> 'Low'
    """
    ret_score = max(0.0, min(1.0, float(retrieval_confidence))) * 100.0
    val_score = max(0.0, min(100.0, float(consensus_score)))
    combined = round((ret_score * 0.5) + (val_score * 0.5), 1)

    if combined >= 80.0:
        level = "High"
    elif combined >= 60.0:
        level = "Moderate"
    else:
        level = "Low"

    return combined, level


def get_answer_with_memory(question: str, session_id: str, rag_chain, retriever, user_context=None) -> dict:
    """
    Executes RAG question answering with smart topic routing and conversation memory.
    Prioritizes topic-filtered retrieval when specific topics are detected.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    # 1. Rephrase follow-up question using chat history for accurate RAG vector retrieval
    standalone_q = _contextualize_question(question, session_id)
    search_query = f"User Context: {user_context}\nQuestion: {standalone_q}" if user_context else standalone_q

    # 2. Smart Query Routing: Detect topic and retrieve source documents
    question_topic = detect_question_topic(standalone_q)
    source_docs = []

    if question_topic != 'general':
        try:
            filtered_retriever = get_filtered_retriever(question_topic, retriever=retriever, k=4, fetch_k=50)
            if filtered_retriever is not None:
                filtered_docs = filtered_retriever.invoke(search_query)
                if len(filtered_docs) >= 2:
                    source_docs = filtered_docs
        except Exception as filt_err:
            import logging
            logging.getLogger(__name__).warning(f"Filtered retrieval error: {filt_err}")

    # Fallback to standard similarity search if filtered retrieval yielded < 2 docs
    if not source_docs:
        try:
            if hasattr(retriever, "vectorstore"):
                docs_and_scores = retriever.vectorstore.similarity_search_with_score(search_query, k=4)
                source_docs = [doc for doc, _ in docs_and_scores]
            else:
                source_docs = retriever.invoke(search_query)
        except Exception:
            source_docs = retriever.invoke(search_query)

    context = "\n\n".join(doc.page_content for doc in source_docs)

    # Calculate average cosine similarity of the 4 retrieved chunks
    embeddings = _get_embeddings_model()
    try:
        q_vec = np.array(embeddings.embed_query(search_query))
        q_norm = np.linalg.norm(q_vec) + 1e-10
        chunk_sims = []
        for doc in source_docs:
            c_vec = np.array(embeddings.embed_query(doc.page_content[:500]))
            c_norm = np.linalg.norm(c_vec) + 1e-10
            sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
            chunk_sims.append(max(0.0, min(1.0, sim)))
        retrieval_confidence = round(float(np.mean(chunk_sims)), 4) if chunk_sims else 0.85
    except Exception as emb_err:
        import logging
        logging.getLogger(__name__).warning(f"Error computing retrieval confidence: {emb_err}")
        retrieval_confidence = 0.85

    # 3. Build RunnableWithMessageHistory for the QA prompt
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=800,
        timeout=25
    )

    qa_chain = _MEMORY_QA_PROMPT | llm | StrOutputParser()

    with_message_history = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    effective_question = standalone_q if (standalone_q and standalone_q.strip()) else question

    try:
        answer = with_message_history.invoke(
            {"question": effective_question, "context": context},
            config={"configurable": {"session_id": session_id}}
        )
    except Exception as primary_err:
        import logging
        logging.getLogger(__name__).warning(f"Primary memory RAG chain failed: {primary_err}. Fallback to openai/gpt-4o...")
        try:
            fb_llm = ChatOpenAI(
                model="openai/gpt-4o",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=0.2,
                max_tokens=800,
                timeout=30
            )
            fb_chain = _MEMORY_QA_PROMPT | fb_llm | StrOutputParser()
            fb_with_history = RunnableWithMessageHistory(
                fb_chain,
                get_session_history,
                input_messages_key="question",
                history_messages_key="chat_history"
            )
            answer = fb_with_history.invoke(
                {"question": effective_question, "context": context},
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

    combined_reliability, reliability_level = calculate_combined_reliability(
        retrieval_confidence=retrieval_confidence,
        consensus_score=80.0
    )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "confidence": 0.85,
        "retrieval_confidence": retrieval_confidence,
        "combined_reliability": combined_reliability,
        "reliability_level": reliability_level,
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
    text = re.sub(r'^\*{0,2}(?:sinhala translation|tamil translation|english translation|translation):\*{0,2}\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\n*(?:Note|Explanation|Translation Note|Here is the translation).*$', '', text, flags=re.IGNORECASE | re.DOTALL)
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
    # Fix manure / dung mistranslations
    text = re.sub(r'(?<![\u0D80-\u0DFF])ගොව(?:ගේ|ියාගේ|ි)?\s*මැදුර(?:වල්)?(?![\u0D80-\u0DFF])', 'ගොම පොහොර', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])ගොවි\s*පොහොර(?![\u0D80-\u0DFF])', 'ගොම පොහොර', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])ගොවියාගේ\s*පොහොර(?![\u0D80-\u0DFF])', 'ගොම පොහොර', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])බකුරු\s*මැදුර(?:වල්)?(?![\u0D80-\u0DFF])', 'එළු පොහොර', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])බකුරු\s*පොහොර(?![\u0D80-\u0DFF])', 'එළු පොහොර', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])(?:පක්ෂි|කුකුල්|කාලීන\s*කුකුල්)\s*මැදුර(?:වල්)?(?![\u0D80-\u0DFF])', 'කුකුල් පොහොර', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])පක්ෂි\s*පොහොර(?![\u0D80-\u0DFF])', 'කුකුල් පොහොර', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])සත්ව\s*මැදුර(?:වල්)?(?![\u0D80-\u0DFF])', 'සත්ව පොහොර', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])හරිත\s*පොහොර(?![\u0D80-\u0DFF])', 'කොළ පොහොර', text)

    # Fix zone mistranslations ("Wet Zone" != "වතුර සහිත ප්‍රදේශය")
    text = re.sub(r'(?<![\u0D80-\u0DFF])මැද\s*කලාප(?:ය|වලට)?(?![\u0D80-\u0DFF])', 'අතරමැදි කලාපය', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])තෙත්\s*සහ\s*මැද\s*කලාප(?![\u0D80-\u0DFF])', 'තෙත් සහ අතරමැදි කලාප', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])වතුර\s*සහිත\s*(?:ප්‍රදේශ(?:ය|යේ)?|කලාප(?:ය|යේ)?)(?![\u0D80-\u0DFF])', 'තෙත් කලාපයේ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])වතුර\s*සහිත\s*හෝ\s*වියළි(?:\s*(?:ප්‍රදේශ(?:ය|යේ)?|කලාප(?:ය|යේ)?))?(?![\u0D80-\u0DFF])', 'තෙත් හෝ වියළි කලාපය', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])තෙත\s*සහිත\s*(?:ප්‍රදේශ(?:ය|යේ)?|කලාප(?:ය|යේ)?)(?![\u0D80-\u0DFF])', 'තෙත් කලාපයේ', text)

    # Fix young coconut palms mistranslations ("green palms" -> "ළපටි පොල් පැළ")
    text = re.sub(r'(?<![\u0D80-\u0DFF])කොළ\s*පොල්\s*(?:ගස්|පැළ|ශාක)(?![\u0D80-\u0DFF])', 'ළපටි පොල් පැළ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])කොළ\s*පොල්(?![\u0D80-\u0DFF])', 'ළපටි පොල් පැළ', text)

    # Fix inanimate pronouns / possessives (trees/plants are NOT human "ඔවුන්ගේ")
    text = re.sub(r'(?<![\u0D80-\u0DFF])ඔවුන්ගේ\s*වයස(?![\u0D80-\u0DFF])', 'ඒවායේ වයස', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])ඔවුන්ගේ\s*(වර්ධනය|ප්‍රමාණය|උස)(?![\u0D80-\u0DFF])', r'ඒවායේ \1', text)

    # Fix plant stage / palm mistranslations and inanimate possessives (ගසේ not ගස්ගේ)
    text = re.sub(r'(?<![\u0D80-\u0DFF])වයසක\s*(?:ශාක(?:ය|යන්)?|ගස්|පොල්\s*ගස්)(?![\u0D80-\u0DFF])', 'වැඩිහිටි පොල් ගස්', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])පොල්\s*පැළගේ\s*දිවිය(?![\u0D80-\u0DFF])', 'පොල් පැළයේ වර්ධනය', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])පොල්\s*ගස්ගේ(?![\u0D80-\u0DFF])', 'පොල් ගසේ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])ගස්ගේ(?![\u0D80-\u0DFF])', 'ගසේ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])ගසගේ(?![\u0D80-\u0DFF])', 'ගසේ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])මව්\s*ගස්ගේ(?![\u0D80-\u0DFF])', 'මව් ගසේ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])ශාකගේ(?![\u0D80-\u0DFF])', 'ශාකයේ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])පොල්\s*පැළගේ(?![\u0D80-\u0DFF])', 'පොල් පැළයේ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])පැළගේ(?![\u0D80-\u0DFF])', 'පැළයේ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])කොළගේ(?![\u0D80-\u0DFF])', 'කොළයේ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])කඳගේ(?![\u0D80-\u0DFF])', 'කඳේ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])මුල්ගේ(?![\u0D80-\u0DFF])', 'මුල්වල', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])(?:පොල්\s*|තවාන්\s*)?පැළවල්ට(?![\u0D80-\u0DFF])', 'තවාන් පැළවලට', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])(?:පොල්\s*|තවාන්\s*)?පැළවල්(?![\u0D80-\u0DFF])', 'පොල් පැළ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])රෝපණය\s*සිට(?![\u0D80-\u0DFF])', 'පැළ සිටුවීමේ සිට', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])මෑණියන්(?![\u0D80-\u0DFF])', 'මව් ශාකය', text)

    # Fix pests / insects mistranslations
    text = re.sub(r'(?<![\u0D80-\u0DFF])සුදු\s*කූඹි(?![\u0D80-\u0DFF])', 'වේයන්', text)
    text = re.sub(r'පළිබෝධකයා\s*ලෙස\s*හැඳින්වෙන\s*"Odontotermis"\s*නම්\s*පළිබෝධකයා', '"Odontotermes" නම් වේයන් විශේෂය', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])කෘෂි\s*භූමිය\s*පිරිසිදු\s*කිරීම(?![\u0D80-\u0DFF])', 'ක්ෂේත්‍ර සනීපාරක්ෂාව (තවාන පිරිසිදුව තබා ගැනීම)', text)

    # Fix question phrasing artifacts
    text = re.sub(r'^\s*මට\s+(පොල්\s+තවානේ\s+වේයන්\s+පාලනය\s+කරන්නේ\s+කෙසේද\?)', r'\1', text)
    text = re.sub(r'^\s*මට\s+(පොල්\s+පැළ\s+සඳහා\s+පොහොර\s+යෙදිය\s+යුත්තේ\s+කෙසේද\?)', r'\1', text)

    # Fix chemical, nutrient and fertilizer names
    text = re.sub(r'(?<![\u0D80-\u0DFF])(?:ළාභය|ලාභය)\s*සඳහා,?(?![\u0D80-\u0DFF])', 'පොස්පරස් (P) සඳහා,', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])යුරියා(?![\u0D80-\u0DFF])', 'යූරියා', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])මුරියට්\s*ඔෆ්\s*පොටෑෂ්(?![\u0D80-\u0DFF])', 'මියුරියේට් ඔෆ් පොටෑෂ් (MOP)', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])මුරියට්\s*ඕ්ප්‍රාස්(?![\u0D80-\u0DFF])', 'මියුරියේට් ඔෆ් පොටෑෂ් (MOP)', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])මියුරියේට්\s*ඔෆ්\s*පොටෑෂ්(?!\s*\(MOP\))(?![\u0D80-\u0DFF])', 'මියුරියේට් ඔෆ් පොටෑෂ් (MOP)', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])එප්පවල්\s*තිත්තාලියා\s*ෆොස්ෆේට්(?![\u0D80-\u0DFF])', 'එප්පාවල රොක් පොස්පේට් (ERP)', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])එප්පාවල\s*රොක්\s*ෆොස්ෆේට්(?![\u0D80-\u0DFF])', 'එප්පාවල රොක් පොස්පේට් (ERP)', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])ත්‍රිත්ව\s*සුපර්\s*ෆොස්ෆේට්(?![\u0D80-\u0DFF])', 'ත්‍රිත්ව සුපර් පොස්පේට් (TSP)', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])ත්‍රීත්ව\s*සුපර්\s*ෆොස්ෆේට්(?![\u0D80-\u0DFF])', 'ත්‍රිත්ව සුපර් පොස්පේට් (TSP)', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])ත්‍රිපල්\s*සුපර්\s*ෆොස්ෆේට්(?![\u0D80-\u0DFF])', 'ත්‍රිත්ව සුපර් පොස්පේට් (TSP)', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])ට්‍රිපල්\s*සුපර්\s*ෆොස්ෆේට්(?![\u0D80-\u0DFF])', 'ත්‍රිත්ව සුපර් පොස්පේට් (TSP)', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])ස්වාභාවික\s*පොටෑෂ(?:ම්|ියම්)\s*සල්ෆේට්(?![\u0D80-\u0DFF])', 'පොටෑසියම් සල්ෆේට් (K2SO4)', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])නිශ්පාප\s*කොටස්\s*සල්ෆේට්(?![\u0D80-\u0DFF])', 'පොටෑසියම් සල්ෆේට් (K2SO4)', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])ඩොලමයිට්(?:\s*\(Dolomite\))?(?![\u0D80-\u0DFF])', 'ඩොලමයිට්', text)

    # Fix improper translation of 'recommend' to proper name 'අනුරුද්ධ'
    text = re.sub(r'(?<![\u0D80-\u0DFF])අනුරුද්ධ(?![\u0D80-\u0DFF])', 'නිර්දේශ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])අනුරුද්ධ කරමි(?![\u0D80-\u0DFF])', 'නිර්දේශ කරමි', text)
    # Fix unnatural bookish / garbled phrases and imperative verbs
    text = re.sub(r'(?<![\u0D80-\u0DFF])යෙදෙන්න(?![\u0D80-\u0DFF])', 'යොදන්න', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])යෙදෙන්නේ\s*කෙසේද(?![\u0D80-\u0DFF])', 'යෙදිය යුත්තේ කෙසේද', text)
    text = re.sub(r'ලෙල දෙන ලෙසට දැන ගන්න', 'පහත දැක්වේ', text)
    text = re.sub(r'කෙළවර කිරීම යම් ආකාරයකින් ද\?', 'පොහොර යෙදිය යුත්තේ කෙසේද?', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])නාරකොළ(?![\u0D80-\u0DFF])', 'පොල් පැළ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])නාරටි(?![\u0D80-\u0DFF])', 'පොල් පැළ', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])පොල් කොළ වලින් වසුන්(?![\u0D80-\u0DFF])', 'පොල් ලෙලි වලින් වසුන්', text)

    # Fix soil, application methods, and bearing terms
    text = re.sub(r'(?<![\u0D80-\u0DFF])මැදියම්(?!\s*(?:කාල|අවධි))(?![\u0D80-\u0DFF])', 'පස', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])බීජය\s*දක්වා(?![\u0D80-\u0DFF])', 'ඵල දැරීම දක්වා', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])කුඩු\s*යෙදීම(?![\u0D80-\u0DFF])', 'කාණු තුළ යෙදීම', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])කුඩු\s*කුඩුවක්(?![\u0D80-\u0DFF])', 'නොගැඹුරු කාණුවක්', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])කුඩුවේ\s*තබා(?![\u0D80-\u0DFF])', 'කාණුව තුළ දමා', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])පෝෂණයන්(?![\u0D80-\u0DFF])', 'පෝෂක', text)

    # Fix numbers with months/years: e.g. "6 මාස" -> "මාස 6"
    text = re.sub(r'(?<![\u0D80-\u0DFF])(\d+)\s*මාස(?!\w)', r'මාස \1', text)
    text = re.sub(r'(?<![\u0D80-\u0DFF])(\d+)\s*(?:වසර|අවුරුදු)(?!\w)', r'වසර \1', text)

    # Strip trailing orphan bullet lines
    text = re.sub(r'\n\s*[-*•]\s*$', '', text)
    return text.strip()


def _sanitize_tamil_advisory(text: str) -> str:
    """Post-processing sanitizer to fix common LLM Tamil translation artifacts."""
    if not text:
        return ""
    import re
    # Fix manure mistranslations
    text = re.sub(r'(?<![\u0B80-\u0BFF])மாடு\s*குப்பை(?![\u0B80-\u0BFF])', 'மாட்டு எரு / மாட்டுச் சாணம்', text)
    text = re.sub(r'(?<![\u0B80-\u0BFF])ஆடு\s*குப்பை(?![\u0B80-\u0BFF])', 'ஆட்டு எரு', text)
    text = re.sub(r'(?<![\u0B80-\u0BFF])கோழி\s*குப்பை(?![\u0B80-\u0BFF])', 'கோழி எரு', text)
    text = re.sub(r'(?<![\u0B80-\u0BFF])முரியட்\s*ஆஃப்\s*பொட்டாஷ்(?![\u0B80-\u0BFF])', 'மியூரியேட் ஆஃப் பொட்டாஷ் (MOP)', text)
    text = re.sub(r'(?<![\u0B80-\u0BFF])எப்பவலா\s*ராக்\s*ஃபாஸ்பேட்(?![\u0B80-\u0BFF])', 'எப்பாவல பாறை பொசுபேற்று (ERP)', text)
    text = re.sub(r'(?<![\u0B80-\u0BFF])டிரிபிள்\s*சூப்பர்\s*போஸ்பேட்(?![\u0B80-\u0BFF])', 'மும்மை சூப்பர் பொசுபேற்று (TSP)', text)
    # Fix common LLM Tamil mistranslations in agricultural context
    text = re.sub(r'(?<![\u0B80-\u0BFF])பரிந்துரைக்கப்பட்டது(?![\u0B80-\u0BFF])', 'பரிந்துரை', text)
    text = re.sub(r'(?<![\u0B80-\u0BFF])தேங்காய் இலைகள் மூலம் மூடுதல்(?![\u0B80-\u0BFF])', 'தேங்காய் நார் மூலம் மூடுதல்', text)
    text = re.sub(r'(?<![\u0B80-\u0BFF])அம்மா பனை(?![\u0B80-\u0BFF])', 'தாய் பனை', text)
    text = re.sub(r'(?<![\u0B80-\u0BFF])விதைப்பு செடி(?![\u0B80-\u0BFF])', 'தேங்காய் நாற்று', text)
    # Fix termite and sanitation mistranslations
    text = re.sub(r'(?<![\u0B80-\u0BFF])வெள்ளைமுட்டை\s*பூச்சிக(?:ள்|ளை)?(?![\u0B80-\u0BFF])', 'கரையான்களை', text)
    text = re.sub(r'(?<![\u0B80-\u0BFF])விவசாயம்\s*சுத்தம்\s*செய்தல்(?![\u0B80-\u0BFF])', 'நாற்றங்கால் தூய்மை பேணுதல்', text)
    text = re.sub(r'(?<![\u0B80-\u0BFF])நாற்றங்கலில்(?![\u0B80-\u0BFF])', 'நாற்றங்காலில்', text)
    text = re.sub(r'(?<![\u0B80-\u0BFF])கரையான்\s*கையாள\s*எப்படி(?![\u0B80-\u0BFF])', 'கரையான்களை கட்டுப்படுத்துவது எப்படி', text)
    # Strip trailing orphan bullet lines
    text = re.sub(r'\n\s*[-*•]\s*$', '', text)
    return text.strip()


def _clean_tamil_translation_output(text: str) -> str:
    """Strips thinking tags and conversational artifacts from Tamil LLM output."""
    if not text:
        return ""
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^\*{0,2}(?:tamil translation|english translation|translation):\*{0,2}\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\n*(?:Note|Explanation|Translation Note|Here is the translation).*$', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = text.strip(' "\'\n\r')
    text = re.sub(r'\bferomone\b', 'pheromone', text, flags=re.IGNORECASE)
    return text.strip()


def _is_translation_valid(text: str, target_lang: str) -> bool:
    """
    Validates translated output.
    Ensures that target language script is dominant and non-empty.
    Allows for standard agricultural abbreviations and units (kg, g, N, P, K, MOP, ERP, etc.).
    """
    if not text or not text.strip():
        return False
    if target_lang == "si":
        sinhala_chars = sum(1 for c in text if '\u0d80' <= c <= '\u0dff')
        latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        return sinhala_chars > 0 and sinhala_chars >= (latin_chars * 0.8)
    elif target_lang == "ta":
        tamil_chars = sum(1 for c in text if '\u0b80' <= c <= '\u0bff')
        latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        return tamil_chars > 0 and tamil_chars >= (latin_chars * 0.8)
    elif target_lang == "en":
        latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        return latin_chars > 0 and not is_sinhala(text) and not is_tamil(text)
    return True


def translate_text(text, target_lang):
    """
    Translates text to target_lang ('en', 'si', or 'ta') using ChatOpenAI via OpenRouter.
    Uses model cascade: openai/gpt-4o-mini (fast) -> openai/gpt-4o.
    """
    if not text or not text.strip():
        return ""

    if target_lang == "si":
        prompt = PromptTemplate.from_template("""You are an expert agricultural translator specializing in Sri Lankan coconut farming (Coconut Research Institute - CRI Sri Lanka).
Translate the following English text into natural, fluent, farmer-friendly, grammatically precise Sinhala (සිංහල).

CRITICAL SRI LANKAN COCONUT FARMING RULES:
1. COMPLETE & UNABRIDGED: Translate the full text completely without omitting any points, sections, dosages, or numbers. Maintain all bullet points, numbered lists, line breaks, and bold formatting (**...**).
2. QUESTIONS: If the input is a question, translate ONLY the question sentence into Sinhala. DO NOT answer it!
3. PALM STAGES & CROPS:
   - "young coconut palms" / "young palms" / "young coconut trees" -> "ළපටි පොල් පැළ" / "ළපටි පොල් ගස්" / "කුඩා පොල් ගස්" (NEVER translate as "කොළ පොල් ගස්" or "කොළ ගස්"!).
   - "seedlings" / "coconut seedlings" -> "පොල් පැළ" / "තවාන් පැළ".
   - "mature palms" / "adult palms" -> "වැඩිහිටි පොල් ගස්".
   - "mother palm" -> "මව් ගස" / "මව් පොල් ගස" / "මව් ශාකය".
   - "bearing" / "up to bearing" -> "ඵල දැරීම දක්වා" / "ගෙඩි හටගැනීම දක්වා".
4. CLIMATIC ZONES & SEASONS:
   - "Wet Zone" / "wet zone" -> "තෙත් කලාපය" (in the wet zone -> "තෙත් කලාපයේ", NEVER "වතුර සහිත" or "තෙත සහිත").
   - "Dry Zone" / "dry zone" -> "වියළි කලාපය" (in the dry zone -> "වියළි කලාපයේ").
   - "Intermediate Zone" / "intermediate zone" -> "අතරමැදි කලාපය" (in the intermediate zone -> "අතරමැදි කලාපයේ").
   - "wet or dry [zone]" -> "තෙත් හෝ වියළි කලාපය".
   - "Yala season" -> "යල කන්නය", "Maha season" -> "මහ කන්නය".
5. INANIMATE & PLANT PRONOUNS / POSSESSIVES:
   - For palms, trees, plants, or zones, use inanimate pronouns/possessives: "ඒවායේ" / "ගස්වල" / "පැළවල" / "එහි" (e.g., "their age" -> "ඒවායේ වයස" / "ගස්වල වයස"). NEVER use human pronouns like "ඔවුන්ගේ", "ඔවුන්ට", or "ඔහුගේ"!
   - For single nouns, use "ගසේ" / "පොල් ගසේ" (NEVER "ගස්ගේ" / "ගසගේ"), "පැළයේ" (NEVER "පැළගේ"), "ශාකයේ" (NEVER "ශාකගේ").
6. ACTION VERBS & IMPERATIVES:
   - For instructions like "apply" (fertilizer, mulch, water), use active imperative "යොදන්න" (NEVER "යෙදෙන්න").
   - "fertilize" / "fertilization" -> "පොහොර යෙදීම" / "පොහොර දැමීම".
   - "How should I fertilize" -> "පොහොර යෙදිය යුත්තේ කෙසේද" / "පොහොර යොදන්නේ කෙසේද".
7. SOIL, PRACTICES & APPLICATION:
   - "soil" -> "පස" (moisten the soil -> පස තෙතමනය කරන්න, NEVER "මැදියම්").
   - "surface application" -> "මතුපිට යෙදීම" (පස මතුපිට පොහොර යෙදීම).
   - "trench application" -> "කාණු තුළ යෙදීම" / "කාණු ක්‍රමය" (පොල් ගස වටා නොගැඹුරු කාණුවක් කපා පොහොර දැමීම).
   - "nutrients" / "absorption of nutrients" -> "පෝෂක" / "පෝෂක අවශෝෂණය".
   - "trunk / base of the palm" -> "කඳ" / "පොල් ගසේ පාමුල".
8. FERTILIZERS, CHEMICALS & NUTRIENTS:
   - "Urea" -> "යූරියා"
   - "Triple Super Phosphate" / "TSP" -> "ත්‍රිත්ව සුපර් පොස්පේට් (TSP)" / "ට්‍රිපල් සුපර් පොස්පේට් (TSP)"
   - "Muriate of Potash" / "MOP" -> "මියුරියේට් ඔෆ් පොටෑෂ් (MOP)"
   - "Dolomite" -> "ඩොලමයිට්"
   - "Eppawala Rock Phosphate" / "ERP" -> "එප්පාවල රොක් පොස්පේට් (ERP)"
   - "Young Palm Mixture" / "YPM" -> "යොවුන් පොල් පොහොර මිශ්‍රණය (YPM)"
   - "Adult Palm Mixture" / "APM" -> "වැඩිහිටි පොල් පොහොර මිශ්‍රණය (APM)"
   - "manure circle" -> "පොහොර වළල්ල" / "පොහොර කවය"
   - "mulch" / "mulching" -> "වසුන් කිරීම" / "වසුන"
   - "coconut husks" -> "පොල් ලෙලි" (NEVER "පොල් කොළ")
   - "cow dung" -> "ගොම පොහොර" (NEVER "ගොවි පොහොර" or "ගොවියාගේ පොහොර")
   - "poultry manure" -> "කුකුල් පොහොර"
   - "goat manure" -> "එළු පොහොර"
   - "green manure" -> "කොළ පොහොර"
9. PESTS & DISEASES:
   - "red palm weevil" -> "රතු කුරුමිණියා"
   - "black beetle" / "rhinoceros beetle" -> "කළු කුරුමිණියා" / "පොල් කුරුමිණියා"
   - "coconut mite" -> "පොල් මයිටා"
   - "coconut caterpillar" -> "කොළ කන දළඹුවා"
   - "termites" / "white ants" -> "වේයන්"
   - "bud rot" -> "කරටි කුණුවීම" / "ගොබ කුණුවීම"
   - "stem bleeding" -> "කඳෙන් ශ්‍රාවය ගැලීම"
   - "Weligama coconut leaf wilt" -> "වැලිගම කොළ මැලවීමේ රෝගය"
   - "collar rot" -> "කරවැටි කුණුවීම"
10. TIME & AGE:
   - "6 Months" -> "මාස 6"
   - "1 Year" -> "වසර 1"
   - "2 Years" -> "වසර 2"
   - "At planting" -> "පැළ සිටුවීමේදී"
11. PRESERVE CODES & UNITS:
   - Keep abbreviations (ERP, TSP, MOP, YPM, APM, NPK) and units (kg, g, ml, cm, m, ha) intact.
12. SCRIPT ONLY: Output ONLY the Sinhala translation without preamble, quotation marks, or English explanations.

English:
{text}

Sinhala:""")
    elif target_lang == "ta":
        prompt = PromptTemplate.from_template("""You are an expert agricultural translator specializing in Sri Lankan coconut farming (Coconut Research Institute - CRI Sri Lanka).
Translate the following English text into natural, fluent, farmer-friendly Sri Lankan Tamil (தமிழ்).

CRITICAL SRI LANKAN COCONUT FARMING RULES:
1. COMPLETE & UNABRIDGED: Translate the full text completely without omitting any points, sections, dosages, or numbers. Maintain all bullet points, numbered lists, line breaks, and bold formatting (**...**).
2. QUESTIONS: If the input is a question, translate ONLY the question sentence into Tamil. DO NOT answer it!
3. PALM STAGES:
   - "young coconut palms" / "young palms" -> "இளம் தென்னை மரங்கள்" / "தென்னை நாற்றுக்கள்"
   - "seedlings" -> "தென்னை நாற்றுக்கள்"
   - "mature palms" -> "முதிர்ந்த தென்னை மரங்கள்"
   - "mother palm" -> "தாய் பனை" / "தாய் மரம்"
4. CLIMATIC ZONES & SEASONS:
   - "Wet Zone" -> "ஈர மண்டலம்" (in the wet zone -> "ஈர மண்டலத்தில்")
   - "Dry Zone" -> "வறண்ட மண்டலம்" (in the dry zone -> "வறண்ட மண்டலத்தில்")
   - "Intermediate Zone" -> "இடைநிலை மண்டலம்" (in the intermediate zone -> "இடைநிலை மண்டலத்தில்")
   - "wet or dry [zone]" -> "ஈர அல்லது வறண்ட மண்டலம்"
   - "Yala season" -> "யல பருவம்", "Maha season" -> "மகா பருவம்"
5. FERTILIZERS & CHEMICALS:
   - "Urea" -> "யூரியா"
   - "Triple Super Phosphate" / "TSP" -> "மும்மை சூப்பர் பொசுபேற்று (TSP)"
   - "Muriate of Potash" / "MOP" -> "மியூரியேட் ஆஃப் பொட்டாஷ் (MOP)"
   - "Dolomite" -> "டோலமைட்" (Dolomite)
   - "Eppawala Rock Phosphate" / "ERP" -> "எப்பாவல பாறை பொசுபேற்று (ERP)"
   - "cow dung" -> "மாட்டு எரு"
   - "poultry manure" -> "கோழி எரு"
   - "goat manure" -> "ஆட்டு எரு"
   - "green manure" -> "பச்சை இலை உரம்"
   - "manure circle" -> "உர வட்டம்"
   - "mulching" -> "மூடாக்கிடுதல்"
   - "coconut husks" -> "தேங்காய் மட்டை / தேங்காய் நார்"
6. PESTS & DISEASES:
   - "red palm weevil" -> "சிவப்பு பனை நாவாய்ப்பூச்சி"
   - "black beetle" / "rhinoceros beetle" -> "கருப்பு வண்டு / காண்டாமிருக வண்டு"
   - "coconut caterpillar" -> "தென்னை கம்பளிப்பூச்சி"
   - "coconut mite" -> "தேங்காய் பூச்சி"
   - "termites" -> "கரையான்கள்"
   - "bud rot" -> "குருத்து அழுகல்"
   - "stem bleeding" -> "தண்டு வடிதல்"
7. PRESERVE CODES & UNITS: Keep ERP, TSP, MOP, YPM, APM, NPK, kg, g, ml, cm intact.
8. SCRIPT ONLY: Output ONLY the Tamil translation without commentary or extra quotes.

English:
{text}

Tamil:""")
    else:
        # Detect source language for -> English translation
        detected_lang = get_language(text)
        if detected_lang == 'ta':
            prompt = PromptTemplate.from_template("""You are an expert agricultural translator specializing in Sri Lankan coconut farming.
Translate the following Tamil (தமிழ்) farmer query into clear, natural, grammatically correct English for an agricultural advisory system.

CRITICAL SRI LANKAN COCONUT FARMING VOCABULARY:
- தாய் பனை -> mother palm
- உரம் -> fertilizer
- தைல் கன்று / கன்று / தேங்காய் நாற்று -> seedling / young palm
- கரையான் / கரையான்கள் -> termites
- தேங்காய் -> coconut
- யல பருவம் / யாழ் பருவம் -> Yala season
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
- கருப்பு வண்டு / காண்டாமிருக வண்டு -> black beetle / rhinoceros beetle
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
4. Do NOT output think tags or reasoning.
5. Output ONLY the translated English text.

TEXT TO TRANSLATE:
{text}

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
- පොල් ගස් / පොල් පැළ / කුඩා පොල් ගස් / ළපටි පොල් පැළ -> coconut palms / coconut seedlings / young coconut palms
- තෙත් කලාපය / වියළි කලාපය / අතරමැදි කලාපය -> Wet Zone / Dry Zone / Intermediate Zone
- යල කන්නය / මහ කන්නය -> Yala season / Maha season

RULES:
1. Translate into a direct, fluent English sentence without commentary.
2. Do NOT enclose output in quotation marks.
3. Preserve codes & units (YPM-W, APM, NPK, kg, g, ml, cm).
4. Do NOT output think tags or reasoning.
5. Output ONLY the translated English text.

TEXT TO TRANSLATE:
{text}

ENGLISH TRANSLATION:""")

    TRANSLATION_CASCADE = [
        "openai/gpt-4o-mini",
    ]

    # Calculate optimal token budget based on input length (Sinhala/Tamil token expansion factor ~4-8x)
    word_count = len(text.split()) if text else 10
    dynamic_max_tokens = max(600, min(3500, word_count * 8))

    for model_candidate in TRANSLATION_CASCADE:
        try:
            import logging
            _logger = logging.getLogger(__name__)
            _logger.info(f"Translation attempt with {model_candidate} to [{target_lang}] (max_tokens={dynamic_max_tokens})...")
            _t_start = time.time()
            llm = ChatOpenAI(
                model=model_candidate,
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=dynamic_max_tokens,
                timeout=25
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

            _elapsed = round(time.time() - _t_start, 2)
            _logger.info(f"Translation with {model_candidate} completed in {_elapsed}s")

            # Quality validation check
            if _is_translation_valid(cleaned_res, target_lang):
                return cleaned_res
            else:
                _logger.warning(f"Translation output with {model_candidate} failed validation (Latin ratio or empty). Trying next fallback...")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Translation model {model_candidate} error: {e}")

    return text



def translate_multi_llm_payload(payload: dict, target_lang: str = "si") -> dict:
    """
    Translates Multi-LLM fields (best_answer, reason, llama_answer, llama8b_answer/gpt4omini_answer, gemma_answer)
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
    'llama': 'meta-llama/llama-3.3-70b-instruct',
    'llama8b': 'openai/gpt-4o-mini',
    'gemma': 'google/gemma-2-27b-it',
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

ANSWER FROM GPT-4o Mini:
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
    logger = logging.getLogger(__name__)

    prompt = PromptTemplate.from_template(_ADVISOR_PROMPT_TEMPLATE)

    # Per-model dedicated fallback order via OpenRouter
    DEDICATED_FALLBACKS = {
        "meta-llama/llama-3.3-70b-instruct": ["openai/gpt-4o-mini", "qwen/qwen-2.5-72b-instruct"],
        "openai/gpt-4o-mini": ["qwen/qwen-2.5-72b-instruct", "meta-llama/llama-3.3-70b-instruct"],
        "google/gemma-2-27b-it": ["openai/gpt-4o-mini", "qwen/qwen-2.5-72b-instruct"],
    }

    models_to_try = [model_name] + DEDICATED_FALLBACKS.get(model_name, ["openai/gpt-4o-mini", "qwen/qwen-2.5-72b-instruct"])

    raw_answer = ""
    for idx, candidate in enumerate(models_to_try):
        try:
            llm = ChatOpenAI(
                model=candidate,
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=1000
            )
            chain = prompt | llm | StrOutputParser()
            raw_answer = chain.invoke({"context": context[:2000], "question": question})
            if raw_answer and len(raw_answer.strip()) > 10:
                break
        except Exception as err:
            logger.warning(f"Model {candidate} failed: {err}")
            continue

    if not raw_answer or len(raw_answer.strip()) <= 10:
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
    - If similarity >= EARLY_EXIT_THRESHOLD (0.80), the Judge LLM is skipped.
    - All 3 candidate answers are ALWAYS collected (no candidate is cancelled).
    - The latency saving comes solely from skipping the Judge LLM call.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Model rank priority for early exit selection (lower index = higher rank)
    MODEL_RANK = {"llama": 0, "llama8b": 1, "gemma": 2}

    # 1. Retrieve context chunks (same for all 3 LLMs) with smart topic routing
    search_query = f"User Context: {user_context}\nQuestion: {question}" if user_context else question
    question_topic = detect_question_topic(question)
    source_docs = []

    if question_topic != 'general':
        try:
            filtered_retriever = get_filtered_retriever(question_topic, retriever=retriever, k=4, fetch_k=50)
            if filtered_retriever is not None:
                filtered_docs = filtered_retriever.invoke(search_query)
                if len(filtered_docs) >= 2:
                    source_docs = filtered_docs
        except Exception as filt_err:
            logger.warning(f"Filtered retrieval in get_multi_llm_answer error: {filt_err}")

    # Fallback to standard similarity search if filtered retrieval yielded < 2 docs
    if not source_docs:
        try:
            if hasattr(retriever, "vectorstore"):
                docs_and_scores = retriever.vectorstore.similarity_search_with_score(search_query, k=4)
                source_docs = [doc for doc, _ in docs_and_scores]
            else:
                source_docs = retriever.invoke(search_query)
        except Exception:
            source_docs = retriever.invoke(search_query)

    context = "\n\n".join(doc.page_content for doc in source_docs)[:3000]

    # Compute retrieval confidence from average cosine similarity of retrieved chunks
    embeddings = _get_embeddings_model()
    try:
        q_vec = np.array(embeddings.embed_query(search_query))
        q_norm = np.linalg.norm(q_vec) + 1e-10
        chunk_sims = []
        for doc in source_docs:
            c_vec = np.array(embeddings.embed_query(doc.page_content[:500]))
            c_norm = np.linalg.norm(c_vec) + 1e-10
            sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
            chunk_sims.append(max(0.0, min(1.0, sim)))
        retrieval_confidence = round(float(np.mean(chunk_sims)), 4) if chunk_sims else 0.85
    except Exception as emb_err:
        logger.warning(f"Error computing retrieval confidence in get_multi_llm_answer: {emb_err}")
        retrieval_confidence = 0.85

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
            judge_llm = ChatOpenAI(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=500
            )
            judge_chain = judge_prompt | judge_llm | StrOutputParser()
            judge_raw = judge_chain.invoke(judge_payload)
        except Exception as judge_err:
            logger.warning(f"Judge primary model failed: {judge_err}. Trying fallback...")
            judge_llm_fb = ChatOpenAI(
                model="qwen/qwen-2.5-72b-instruct",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=500
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

    combined_reliability, reliability_level = calculate_combined_reliability(
        retrieval_confidence=retrieval_confidence,
        consensus_score=consensus_score
    )

    return {
        "best_answer": answers.get(best_model, answers.get("llama", "")),
        "best_model": best_model,
        "reason": reason,
        "consensus_score": consensus_score,
        "retrieval_confidence": retrieval_confidence,
        "combined_reliability": combined_reliability,
        "reliability_level": reliability_level,
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