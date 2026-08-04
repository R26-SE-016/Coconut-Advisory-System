# step2_rag_engine.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import json
from concurrent.futures import ThreadPoolExecutor

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Resolve FAISS index path relative to this file (project root)
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(_ROOT_DIR, "faiss_index")

def load_rag_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
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


def get_answer(question, rag_chain, retriever, user_context=None):
    search_query = f"User Context: {user_context}\nQuestion: {question}" if user_context else question
    try:
        answer = rag_chain.invoke(search_query)
    except Exception as primary_err:
        import logging
        logging.getLogger(__name__).warning(f"Primary RAG chain failed: {primary_err}. Attempting fallback model...")
        try:
            fb_llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"), temperature=0.2)
            source_docs_fb = retriever.invoke(search_query)
            context_fb = "\n\n".join(doc.page_content for doc in source_docs_fb)
            prompt_fb = PromptTemplate.from_template("""You are an expert agricultural advisor for coconut farming in Sri Lanka.
Use ONLY the information from the context below to answer the question.
If the answer is not found in the context, say: "I don't have information about that in my knowledge base."
Give practical advice a farmer can understand and apply immediately.

Context:
{context}

Question: {question}

Answer:""")
            chain_fb = prompt_fb | fb_llm | StrOutputParser()
            answer = chain_fb.invoke({"context": context_fb, "question": question})
        except Exception as fb_err:
            logging.getLogger(__name__).error(f"Fallback RAG chain error: {fb_err}")
            answer = "Sorry, I am facing connectivity issues to my knowledge base. Please check your internet connection."

    source_docs = retriever.invoke(search_query)
    sources = []
    for doc in source_docs:
        source_title = os.path.basename(doc.metadata.get("source", "Unknown"))
        # Avoid duplicate sources
        if not any(s["title"] == source_title for s in sources):
            sources.append({
                "title": source_title,
                "content": doc.page_content[:200],  # First 200 chars as preview
                "metadata": doc.metadata
            })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "confidence": 0.85,  # Placeholder confidence score
        "context_used": user_context
    }

def get_plain_answer(question, user_context=None):
    """
    Queries the LLM directly without any RAG context.
    Used for comparison to show the value of the RAG system.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7
    )
    
    prompt = PromptTemplate.from_template("""You are an AI assistant. Answer the following question to the best of your general knowledge.
    
User Context: {user_context}
Question: {question}
    
Answer:""")
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "user_context": user_context or "None"})
    
    return {
        "question": question,
        "answer": answer
    }
def _clean_llm_translation_output(text: str) -> str:
    """Strips thinking tags (<think>...</think>) and extraneous markdown wrappers from LLM output."""
    if not text:
        return ""
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

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
    return text

def translate_text(text, target_lang):
    """
    Translates text to target_lang ('en' or 'si') using ChatGroq with a high-accuracy farmer-oriented prompt.
    Uses model cascade: qwen/qwen3.6-27b -> openai/gpt-oss-120b -> llama-3.3-70b-versatile -> llama-3.1-8b-instant.
    """
    if not text or not text.strip():
        return ""

    if target_lang == "si":
        prompt = PromptTemplate.from_template("""You are an expert agricultural translator specializing in Sri Lankan coconut farming.
Translate the input English text into natural, clear, farmer-friendly Sinhala (සිංහල) that a Sri Lankan coconut farmer can easily understand.

CRITICAL TRANSLATION RULES:
1. Tone & Clarity: Use natural Sri Lankan Sinhala sentence structure. Avoid word-for-word literal translations or complex bookish words that ordinary farmers do not use.
2. Question Translation:
   - "How should I fertilize young coconut palms?" -> "තරුණ පොල් ගස් වලට පොහොර යෙදිය යුත්තේ කෙසේද?"
   - "How do I select a good mother palm?" -> "හොඳ මව් පොල් ගසක් තෝරා ගන්නේ කෙසේද?"
   - "How do I control termites in coconut nursery?" -> "පොල් තවානේ වේයන් පාලනය කරන්නේ කෙසේද?"
   - "What fertilizer mixture is recommended for coconut seedlings?" -> "කුඩා පොල් පැළ සඳහා නිර්දේශිත පොහොර මිශ්‍රණය කුමක්ද?"
   - "How to manage yellowing of coconut leaves in wet zone?" -> "තෙත් කලාපයේ පොල් පත්‍ර කහ වීම පාලනය කරන්නේ කෙසේද?"
   - "What is the recommended spacing for planting coconut palms?" -> "පොල් ගස් සිටුවීමට නිර්දේශිත පරතරය කුමක්ද?"
3. Agricultural Terminology:
   - mother palm -> මව් ශාකය / මව් පොල් ගස (NEVER translate as "මෑණියන්")
   - young coconut palms / seedlings -> තරුණ පොල් ගස් / කුඩා පොල් පැළ
   - coconut nursery -> පොල් තවාන
   - fertilizer -> පොහොර
   - Wet Zone / Dry Zone / Intermediate Zone -> තෙත් කලාපය / වියළි කලාපය / අතරමැදි කලාපය
   - Yala season / Maha season -> යල කන්නය / මහ කන්නය
   - recommend -> නිර්දේශිත / නිර්දේශ කරමි (NEVER translate as "අනුරුද්ධ")
   - split applications -> කොටස් වශයෙන් යෙදීම
   - termites -> වේයන්
   - rhinoceros beetle -> රයිනෝසරස් කුරුමිණියා (අං කුරුමිණියා)
   - yellowing -> කොළ කහ වීම / පත්‍ර කහ වීම
   - nitrogen deficiency -> නයිට්‍රජන් ඌනතාවය
   - soil moisture stress -> පසේ තෙතමනය හිඟකම
   - yield -> අස්වැන්න
   - spacing -> පරතරය / සිටුවීමේ පරතරය
   - coconut palm / tree -> පොල් ගස / පොල් පැළ
4. Preserve codes & units: Keep codes (YPM-W, APM, NPK, CRI) and units (kg, g, ml, cm, m) unchanged.
5. Output: Output ONLY the translated Sinhala text. Do NOT add commentary, explanations, markdown fences, or thinking tags.

TEXT TO TRANSLATE:
"{text}"

SINHALA TRANSLATION:""")
    else:
        prompt = PromptTemplate.from_template("""You are an expert agricultural translator specializing in Sri Lankan coconut farming.
Translate the following Sinhala (සිංහල) text into clear, natural English for an agricultural advisory search query.

TRANSLATION RULES:
1. Output ONLY the clear English translation.
2. Use accurate Sri Lankan agricultural terminology:
   - පොල් ගස් / පොල් පැළ -> coconut palms / coconut seedlings
   - පොහොර -> fertilizer
   - පොල් තවාන -> coconut nursery
   - තෙත් කලාපය -> Wet Zone
   - යල කන්නය -> Yala season
   - වේයන් -> termites
   - රයිනෝසරස් කුරුමිණියා / අං කුරුමිණියා -> rhinoceros beetle
3. Preserve codes & units (YPM-W, APM, kg, g, NPK).
4. Output ONLY the translated text, nothing else.

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
            if cleaned_res:
                return cleaned_res
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Translation model {model_candidate} error: {e}")

    return text


def translate_multi_llm_payload(payload: dict, target_lang: str = "si") -> dict:
    """
    Translates Multi-LLM fields (best_answer, reason, llama_answer, llama8b_answer, qwen_answer)
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
    "qwen": "gemma2-9b-it",
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
{qwen_answer}

Respond with ONLY valid JSON in this exact format, no other text:
{{
  "best_model": "llama" or "llama8b" or "qwen",
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
    Runs the same retrieved context through 3 different LLMs in parallel,
    then uses a judge LLM to select the best answer based on faithfulness
    to the CRI source documents.
    """
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
            key: executor.submit(_invoke_llm, model, context, search_query)
            for key, model in MULTI_LLM_MODELS.items()
        }
        answers = {key: future.result() for key, future in futures.items()}

    # 3. Judge evaluation (Use llama-3.1-8b-instant with 500k TPD limit to avoid 70B rate limits)
    judge_prompt = PromptTemplate.from_template(_JUDGE_PROMPT_TEMPLATE)
    try:
        judge_llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0
        )
        judge_chain = judge_prompt | judge_llm | StrOutputParser()
        judge_payload = {
            "context": context[:1500],
            "question": question,
            "llama_answer": answers["llama"][:600],
            "llama8b_answer": answers["llama8b"][:600],
            "qwen_answer": answers["qwen"][:600]
        }
        judge_raw = judge_chain.invoke(judge_payload)
    except Exception as judge_err:
        import logging
        logging.getLogger(__name__).warning(f"Judge primary model failed: {judge_err}. Trying fallback...")
        judge_llm_fb = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0
        )
        judge_chain = judge_prompt | judge_llm_fb | StrOutputParser()
        judge_raw = judge_chain.invoke(judge_payload)

    # 4. Parse judge response
    try:
        judge_result = json.loads(judge_raw.strip())
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from the response
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

    return {
        "best_answer": answers[best_model],
        "best_model": best_model,
        "reason": judge_result.get("reason", ""),
        "consensus_score": judge_result.get("consensus_score", 50),
        "llama_answer": answers["llama"],
        "llama8b_answer": answers["llama8b"],
        "qwen_answer": answers["qwen"],
        "sources": sources,
        "context_used": user_context
    }


_IMAGE_VECTOR_STORE = None
_IMAGE_VECTOR_STORE_MTIME = 0

def reload_image_index():
    """Reset cached FAISS image vector store so it reloads latest index from disk."""
    global _IMAGE_VECTOR_STORE, _IMAGE_VECTOR_STORE_MTIME
    _IMAGE_VECTOR_STORE = None
    _IMAGE_VECTOR_STORE_MTIME = 0

def find_relevant_images(answer_text: str, top_k: int = 2) -> list:
    """
    Given answer_text generated by RAG engine, searches image_index vector store
    for top_k semantically relevant CRI reference images.
    Returns list of dicts: [{'url': '/static/images/...', 'caption': '...', 'source': '...'}]
    """
    global _IMAGE_VECTOR_STORE, _IMAGE_VECTOR_STORE_MTIME
    if not answer_text or not answer_text.strip():
        return []

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
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
            _IMAGE_VECTOR_STORE = FAISS.load_local(
                image_index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            _IMAGE_VECTOR_STORE_MTIME = current_mtime

        docs = _IMAGE_VECTOR_STORE.similarity_search(answer_text, k=top_k)
        images = []
        for doc in docs:
            filename = doc.metadata.get("filename", "")
            url = doc.metadata.get("url") or f"/static/images/{filename}"
            caption = doc.metadata.get("caption", "")
            source = doc.metadata.get("source", "")

            # Ensure image is not duplicate
            if not any(img["url"] == url for img in images):
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