import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Ensure current directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2_rag_engine import (
    FAISS_INDEX_PATH,
    detect_question_topic,
    get_filtered_retriever,
    _get_embeddings_model
)

def test_questions():
    questions = [
        "How do I select a good mother palm?",
        "What are symptoms of collar rot in coconut seedlings?",
        "What is the CRIC65 variety and its characteristics?",
        "How do I prepare nursery beds for coconut?"
    ]

    print("=" * 80)
    print("TESTING SMART TOPIC-FILTERED RETRIEVAL ON 4 TARGET QUESTIONS")
    print("=" * 80)

    embeddings = _get_embeddings_model()
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    for q_idx, q in enumerate(questions, start=1):
        detected_topic = detect_question_topic(q)
        print(f"\n[{q_idx}/4] Question: {q}")
        print(f"Detected Topic: '{detected_topic}'")
        print("-" * 80)

        # Retrieve using smart topic-filtered retriever
        docs = []
        if detected_topic != 'general':
            retriever = get_filtered_retriever(detected_topic, vector_store=vector_store, k=4, fetch_k=100)
            filtered_docs = retriever.invoke(q)
            if len(filtered_docs) >= 2:
                docs = filtered_docs

        if not docs:
            docs = vector_store.similarity_search(q, k=4)

        q_emb = model.encode(q)
        q_norm = np.linalg.norm(q_emb) + 1e-10

        for chunk_idx, doc in enumerate(docs, start=1):
            chunk_topic = doc.metadata.get('topic', 'general')
            content_snippet = doc.page_content.replace('\n', ' ').strip()[:100]
            
            c_emb = model.encode(doc.page_content)
            c_norm = np.linalg.norm(c_emb) + 1e-10
            sim = float(np.dot(q_emb, c_emb) / (q_norm * c_norm))

            print(f"  Chunk {chunk_idx}: [Topic: {chunk_topic:<12s}] | Cosine Sim: {sim:.4f}")
            print(f"    Snippet: \"{content_snippet}...\"\n")

if __name__ == "__main__":
    test_questions()
