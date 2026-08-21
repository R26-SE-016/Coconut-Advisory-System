"""
Retrieval Evaluation Script: Precision@K and Recall@K (Semantic Relevance Scoring)
===================================================================================
Evaluates the RAG retrieval performance of the SaruPol AI Coconut Advisory System.
Measures Precision@4 and Recall@4 across 20 domain-specific benchmark questions
using Semantic Relevance Judgement (Sentence-Transformers cosine similarity scoring)
against Coconut Research Institute (CRI) knowledge base documents.

Usage:
    python -m evaluation.precision_recall_eval
    or
    python evaluation/precision_recall_eval.py
"""

import sys
import os
import json
import re
import numpy as np
from datetime import datetime, timezone
from collections import Counter
from typing import List, Dict, Any, Tuple

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Paths
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss_index")
QUESTIONS_FILE_PATH = os.path.join(PROJECT_ROOT, "evaluation", "test_questions.json")
RESULTS_FILE_PATH = os.path.join(PROJECT_ROOT, "evaluation", "precision_recall_results.json")

# Evaluation hyperparameters
TOP_K_RETRIEVAL = 4               # Production retrieval depth (k=4)
BENCHMARK_K = 20                  # Benchmark candidate pool depth for Recall@K (k=20)
SEMANTIC_SIMILARITY_THRESHOLD = 0.65  # Semantic relevance cosine similarity cutoff (0.65)

# Predefined clean short titles for 20 benchmark questions
QUESTION_SHORT_TITLES = {
    1: "Fertilizer at planting",
    2: "Mother palm selection",
    3: "Planting density",
    4: "Termite control in nursery",
    5: "Palm spacing",
    6: "Young palm fertilization",
    7: "Manage leaf yellowing",
    8: "Seedling collar rot",
    9: "Quality seedling selection",
    10: "Wet zone planting season",
    11: "Black beetle control",
    12: "Bearing palm fertilizer",
    13: "CRIC65 variety traits",
    14: "Nursery bed preparation",
    15: "Organic fertilizers",
    16: "Seedling water management",
    17: "Dry zone fertilizer",
    18: "Plesispa beetle control",
    19: "Vigorous seedling traits",
    20: "Replanting old plantation"
}

_MODEL = None


def get_sentence_transformer():
    """Lazily load SentenceTransformer model singleton."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def load_vector_store():
    """Load FAISS index identical to the production RAG pipeline."""
    print("Loading embedding model and FAISS vector index...", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_INDEX_PATH}'. Please run step1_build_index.py first."
        )

    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS vector store loaded successfully.\n", flush=True)
    return vector_store


def load_test_questions() -> List[Dict[str, Any]]:
    """Load benchmark evaluation questions from test_questions.json."""
    if not os.path.exists(QUESTIONS_FILE_PATH):
        raise FileNotFoundError(f"Test questions file not found at '{QUESTIONS_FILE_PATH}'.")

    with open(QUESTIONS_FILE_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} test questions from test_questions.json.\n", flush=True)
    return questions


def compute_semantic_relevance(
    question: str,
    chunk_text: str,
    model: SentenceTransformer,
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD
) -> Tuple[bool, float]:
    """
    Semantic relevance judgement:
    Computes cosine similarity between question embedding and chunk embedding.
    Returns (is_relevant, similarity_score).
    """
    q_emb = model.encode(question, normalize_embeddings=True)
    c_emb = model.encode(chunk_text, normalize_embeddings=True)
    sim = float(np.dot(q_emb, c_emb))
    is_rel = sim >= threshold
    return is_rel, sim


def extract_matched_keywords(chunk_text: str, keywords: List[str]) -> List[str]:
    """Helper to detect keyword occurrences in chunk text for supplementary tracking."""
    text_lower = chunk_text.lower()
    matched = []
    for kw in keywords:
        kw_clean = kw.strip().lower()
        if not kw_clean:
            continue
        if re.search(r'\b' + re.escape(kw_clean) + r'\b', text_lower) or kw_clean in text_lower:
            matched.append(kw)
    return list(dict.fromkeys(matched))


def format_source_summary(sources: List[str]) -> str:
    """Format list of source documents into count summary e.g. 'English.pdf x3, Sinhala.pdf x1'."""
    counts = Counter(sources)
    formatted = [f"{os.path.basename(src)} x{cnt}" if cnt > 1 else os.path.basename(src) for src, cnt in counts.items()]
    return ", ".join(formatted) if formatted else "None"


def generate_short_title(question_id: int, question: str) -> str:
    """Create a concise 25-30 char display title for table formatting."""
    if question_id in QUESTION_SHORT_TITLES:
        return QUESTION_SHORT_TITLES[question_id]
    clean_q = question.replace("What is the recommended ", "") \
                        .replace("How should I ", "") \
                        .replace("How do I ", "") \
                        .replace("What are symptoms of ", "") \
                        .replace("What are characteristics of a ", "") \
                        .replace("What organic fertilizers can I use for ", "Organic fertilizers for ") \
                        .replace("What fertilizer should I apply when ", "Fertilizer at ") \
                        .replace("What is the ", "") \
                        .replace("?", "").strip()
    if len(clean_q) > 26:
        return clean_q[:24] + ".."
    return clean_q


def run_retrieval_evaluation(threshold: float = SEMANTIC_SIMILARITY_THRESHOLD):
    """Main evaluation pipeline implementing Steps 1 to 7 using Semantic Relevance Scoring."""
    print("=" * 96, flush=True)
    print(f"SARUPOL AI — RETRIEVAL EVALUATION (SEMANTIC RELEVANCE JUDGEMENT, THRESHOLD = {threshold:.2f})", flush=True)
    print("=" * 96, flush=True)
    print()

    # Step 1: Load FAISS index & SentenceTransformer model
    vector_store = load_vector_store()
    model = get_sentence_transformer()
    questions = load_test_questions()

    results = []

    print(f"Evaluating {len(questions)} questions with Semantic Relevance Judgement (threshold={threshold:.2f})...", flush=True)
    print("-" * 96, flush=True)

    # Step 2: Evaluate each question
    for q_item in questions:
        q_id = q_item["id"]
        question = q_item["question"]
        keywords = q_item.get("relevant_keywords", [])

        # Retrieve top 20 candidate chunks from FAISS
        docs_top20 = vector_store.similarity_search(question, k=BENCHMARK_K)
        docs_top4 = docs_top20[:TOP_K_RETRIEVAL]

        # Batch encode question and all top 20 retrieved chunks
        q_embedding = model.encode(question, normalize_embeddings=True)
        chunk_texts_top20 = [d.page_content for d in docs_top20]
        chunk_embeddings_top20 = model.encode(chunk_texts_top20, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

        # Compute cosine similarities for all 20 chunks
        sims_top20 = np.dot(chunk_embeddings_top20, q_embedding)

        # Evaluate top 4 chunks (Production retrieval k=4)
        top4_chunks_data = []
        relevant_top4_count = 0
        top4_sources = []

        for rank in range(TOP_K_RETRIEVAL):
            doc = docs_top4[rank]
            sim = float(sims_top20[rank])
            is_rel = sim >= threshold
            if is_rel:
                relevant_top4_count += 1

            source_file = doc.metadata.get("source", "Unknown")
            top4_sources.append(source_file)
            preview = doc.page_content.strip()[:100].replace("\n", " ")
            matched_kws = extract_matched_keywords(doc.page_content, keywords)

            top4_chunks_data.append({
                "rank": rank + 1,
                "source": source_file,
                "similarity_score": round(sim, 4),
                "is_relevant": is_rel,
                "preview": preview,
                "matched_keywords": matched_kws
            })

        # Evaluate all top 20 chunks for candidate benchmark pool
        total_relevant_in_top20 = int(np.sum(sims_top20 >= threshold))

        # Step 3: Calculate Precision@4 and Recall@4
        precision_at_4 = relevant_top4_count / TOP_K_RETRIEVAL
        recall_at_4 = (
            relevant_top4_count / total_relevant_in_top20
            if total_relevant_in_top20 > 0
            else (1.0 if relevant_top4_count == 0 else 0.0)
        )

        sources_summary = format_source_summary(top4_sources)
        short_title = generate_short_title(q_id, question)

        results.append({
            "id": q_id,
            "question": question,
            "short_title": short_title,
            "precision_at_4": round(precision_at_4, 4),
            "recall_at_4": round(recall_at_4, 4),
            "relevant_retrieved": relevant_top4_count,
            "k": TOP_K_RETRIEVAL,
            "total_relevant_in_top20": total_relevant_in_top20,
            "sources_summary": sources_summary,
            "retrieved_chunks": top4_chunks_data
        })

    # Step 4: Calculate overall averages
    total_q = len(results)
    avg_precision_4 = sum(r["precision_at_4"] for r in results) / total_q if total_q > 0 else 0.0
    avg_recall_4 = sum(r["recall_at_4"] for r in results) / total_q if total_q > 0 else 0.0

    perfect_precision_count = sum(1 for r in results if r["precision_at_4"] == 1.0)
    high_precision_count = sum(1 for r in results if r["precision_at_4"] >= 0.75)
    low_precision_count = sum(1 for r in results if r["precision_at_4"] < 0.50)

    # Step 5: Print formatted results table
    print()
    print("=" * 96, flush=True)
    print("RETRIEVAL EVALUATION RESULTS TABLE", flush=True)
    print("=" * 96, flush=True)
    header = f"{'ID':<4} {'Question (short)':<28} {'P@4':<8} {'R@4':<8} {'Relevant/4':<12} {'Sources'}"
    print(header, flush=True)
    print("-" * 96, flush=True)

    for r in results:
        p_str = f"{r['precision_at_4']:.2f}"
        r_str = f"{r['recall_at_4']:.2f}"
        rel_str = f"{r['relevant_retrieved']}/{r['k']}"
        print(f"{r['id']:<4} {r['short_title']:<28} {p_str:<8} {r_str:<8} {rel_str:<12} {r['sources_summary']}", flush=True)

    print("-" * 96, flush=True)

    # Step 6: Print summary
    print()
    print("=" * 60, flush=True)
    print("RETRIEVAL EVALUATION SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Evaluation Method:                   Semantic Relevance Judgement", flush=True)
    print(f"Similarity Threshold:                {threshold:.2f}", flush=True)
    print(f"Total Questions:                     {total_q}", flush=True)
    print(f"Average Precision@4:                 {avg_precision_4:.2f}", flush=True)
    print(f"Average Recall@4:                    {avg_recall_4:.2f}", flush=True)
    print(f"Questions with P@4 = 1.0 (perfect):  {perfect_precision_count} ({perfect_precision_count/total_q*100:.1f}%)", flush=True)
    print(f"Questions with P@4 >= 0.75:          {high_precision_count} ({high_precision_count/total_q*100:.1f}%)", flush=True)
    print(f"Questions with P@4 < 0.50:           {low_precision_count} ({low_precision_count/total_q*100:.1f}%)", flush=True)
    print("=" * 60, flush=True)
    print()

    # Step 7: Save results to evaluation/precision_recall_results.json
    output_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_name": "SaruPol RAG Retrieval Evaluation (Semantic Relevance Scoring)",
        "configuration": {
            "evaluation_method": "Semantic Relevance Judgement (Cosine Similarity)",
            "semantic_similarity_threshold": threshold,
            "retrieval_k": TOP_K_RETRIEVAL,
            "benchmark_pool_k": BENCHMARK_K,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_store": "FAISS",
            "total_questions": total_q
        },
        "summary": {
            "total_questions": total_q,
            "average_precision_at_4": round(avg_precision_4, 4),
            "average_recall_at_4": round(avg_recall_4, 4),
            "perfect_precision_count": perfect_precision_count,
            "high_precision_count": high_precision_count,
            "low_precision_count": low_precision_count
        },
        "detailed_results": results
    }

    with open(RESULTS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results successfully saved to:\n  -> {RESULTS_FILE_PATH}\n", flush=True)


if __name__ == "__main__":
    run_retrieval_evaluation()
