# step2_rag_engine.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

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
        model="llama-3.3-70b-versatile",
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
    answer = rag_chain.invoke(search_query)

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
def translate_text(text, target_lang):
    """
    Translates text to target_lang ('en' or 'si') using the LLM.
    """
    if not text.strip():
        return ""
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1
    )
    
    lang_name = "Sinhala" if target_lang == "si" else "English"
    prompt = PromptTemplate.from_template("""You are a professional translator fluent in English and Sinhala, specializing in agricultural terminology (specifically coconut farming in Sri Lanka).
Translate the following text into {lang_name}. 
Ensure that specific agricultural terms are translated accurately to their standard Sri Lankan agricultural equivalents. For example:
- "Bud Rot" or "Bad Rot" should be translated to "කරටි කුණුවීම" in Sinhala.
- "coconut cultivation" to "පොල් වගාව".
- "fertilizer" to "පොහොර".
- "diseases" to "රෝග".
- "pest" to "පළිබෝධකයා".
- "leaf" or "leaves" to "කොළ" or "පත්".

Keep the exact structure, spacing, and formatting (like bullet points, lines, stars) intact.
Only return the translated text. Do not include any explanations, introductory notes, or extra comments.

Text to translate:
{text}

Translation:""")
    
    chain = prompt | llm | StrOutputParser()
    translation = chain.invoke({"text": text, "lang_name": lang_name})
    return translation.strip()


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