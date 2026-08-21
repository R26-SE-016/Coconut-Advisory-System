import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2_rag_engine import (
    FAISS_INDEX_PATH,
    _get_embeddings_model,
    detect_question_topic,
    get_filtered_retriever,
    TOPIC_KEYWORDS
)

def run_diagnostics():
    print("=" * 80)
    print("FAISS INDEX & METADATA FILTERING DIAGNOSTICS")
    print("=" * 80)

    embeddings = _get_embeddings_model()
    vs = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 1. Inspect docstore for first 10 chunks
    print("\n--- 1. FIRST 10 CHUNKS METADATA IN FAISS INDEX ---")
    docstore = vs.docstore._dict
    total_docs = len(docstore)
    print(f"Total chunks in docstore: {total_docs}\n")

    first_10_keys = list(docstore.keys())[:10]
    for idx, key in enumerate(first_10_keys, start=1):
        doc = docstore[key]
        print(f"Chunk {idx:2d} (ID: {key[:8]}...):")
        print(f"  Metadata: {doc.metadata}")
        print(f"  Snippet : \"{doc.page_content.replace(chr(10), ' ')[:90]}...\"\n")

    # Count topic tags across all docs in docstore
    topic_distribution = {}
    for doc in docstore.values():
        t = doc.metadata.get('topic', 'MISSING')
        topic_distribution[t] = topic_distribution.get(t, 0) + 1
    print("Total Topic Tag Distribution in Vectorstore:")
    for t, count in sorted(topic_distribution.items(), key=lambda x: -x[1]):
        print(f"  • {t:15s}: {count} chunks")

    # 2. Test specific question: 'How do I select a good mother palm?'
    question = "How do I select a good mother palm?"
    print("\n" + "=" * 80)
    print(f"--- 2. QUESTION ANALYSIS: \"{question}\" ---")
    print("=" * 80)

    detected_topic = detect_question_topic(question)
    print(f"Question: \"{question}\"")
    print(f"Detected Topic: '{detected_topic}'")
    print(f"Matched Keywords in Topic '{detected_topic}': {[kw for kw in TOPIC_KEYWORDS.get(detected_topic, []) if kw.lower() in question.lower()]}")
    print(f"Filter Applied: lambda metadata: metadata.get('topic') == '{detected_topic}'")

    q_emb = model.encode(question)
    q_norm = np.linalg.norm(q_emb) + 1e-10

    # 3. Retrieve WITHOUT metadata filter (Standard Baseline)
    print("\n" + "-" * 80)
    print("--- 3. RETRIEVAL WITHOUT METADATA FILTER (STANDARD FAISS BASELINE) ---")
    print("-" * 80)
    docs_unfiltered = vs.similarity_search(question, k=4)
    for i, doc in enumerate(docs_unfiltered, start=1):
        c_emb = model.encode(doc.page_content)
        c_norm = np.linalg.norm(c_emb) + 1e-10
        sim = float(np.dot(q_emb, c_emb) / (q_norm * c_norm))
        print(f"Chunk {i}: Topic = [{doc.metadata.get('topic', 'None')}] | Cosine Similarity = {sim:.4f}")
        print(f"  Source : {doc.metadata.get('source', 'Unknown')}")
        print(f"  Snippet: \"{doc.page_content.replace(chr(10), ' ')[:120]}...\"\n")

    # 4. Retrieve WITH metadata filter (Smart Topic Filter)
    print("-" * 80)
    print(f"--- 4. RETRIEVAL WITH METADATA FILTER (Topic == '{detected_topic}') ---")
    print("-" * 80)
    retriever = get_filtered_retriever(detected_topic, vector_store=vs, k=4, fetch_k=100)
    docs_filtered = retriever.invoke(question)
    for i, doc in enumerate(docs_filtered, start=1):
        c_emb = model.encode(doc.page_content)
        c_norm = np.linalg.norm(c_emb) + 1e-10
        sim = float(np.dot(q_emb, c_emb) / (q_norm * c_norm))
        print(f"Chunk {i}: Topic = [{doc.metadata.get('topic', 'None')}] | Cosine Similarity = {sim:.4f}")
        print(f"  Source : {doc.metadata.get('source', 'Unknown')}")
        print(f"  Snippet: \"{doc.page_content.replace(chr(10), ' ')[:120]}...\"\n")

    # 5. Search for any chunk containing 'mother palm' across entire knowledge base
    print("-" * 80)
    print("--- 5. ALL CHUNKS IN KNOWLEDGE BASE CONTAINING 'MOTHER PALM' ---")
    print("-" * 80)
    matching_chunks = []
    for doc_id, doc in docstore.items():
        if "mother palm" in doc.page_content.lower() or "plus palm" in doc.page_content.lower():
            c_emb = model.encode(doc.page_content)
            c_norm = np.linalg.norm(c_emb) + 1e-10
            sim = float(np.dot(q_emb, c_emb) / (q_norm * c_norm))
            matching_chunks.append((sim, doc))

    matching_chunks.sort(key=lambda x: -x[0])
    print(f"Found {len(matching_chunks)} chunks containing 'mother palm' / 'plus palm':\n")
    for rank, (sim, doc) in enumerate(matching_chunks, start=1):
        print(f"Match {rank}: Topic = [{doc.metadata.get('topic')}] | Cosine Sim = {sim:.4f} | Source = {doc.metadata.get('source')}")
        print(f"  Text: \"{doc.page_content.replace(chr(10), ' ')[:150]}...\"\n")

if __name__ == "__main__":
    run_diagnostics()
