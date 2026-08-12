"""
Early Exit Performance Evaluation Script
==========================================
Tests 10 sample coconut farming questions and measures the performance
impact of the early exit optimization in the Multi-LLM validation pipeline.

For each question, records:
- Semantic similarity score between first two completed candidates
- Whether early exit was triggered (similarity >= 0.85)
- Actual response time (with early exit if triggered)
- Estimated response time without early exit

Usage:
    python -m evaluation.early_exit_eval
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2_rag_engine import load_rag_chain, get_multi_llm_answer, EARLY_EXIT_THRESHOLD


# ============ Sample Questions for Evaluation ============

SAMPLE_QUESTIONS = [
    "How should I fertilize young coconut palms?",
    "How do I select a good mother palm?",
    "What is the recommended planting density for coconut?",
    "How do I control termites in coconut nursery?",
    "What fertilizer mixture is recommended for coconut seedlings?",
    "How to manage yellowing of coconut leaves in wet zone?",
    "What is the recommended spacing for planting coconut palms?",
    "How do I identify rhinoceros beetle damage in coconut?",
    "What are the best practices for coconut nursery management?",
    "How to improve coconut yield in dry zone areas?",
]

# Estimated average Judge LLM call time in seconds (based on typical Groq API latency)
ESTIMATED_JUDGE_LATENCY_SEC = 4.0


def measure_early_exit_performance():
    """
    Run 10 sample coconut farming questions through the Multi-LLM pipeline
    and record performance metrics for research evaluation.

    Prints a formatted table showing:
    - Question (truncated)
    - Similarity Score
    - Early Exit Triggered (Y/N)
    - Actual Response Time (ms)
    - Estimated Time Without Early Exit (ms)
    - Time Saved (ms)
    """
    print("=" * 100)
    print("EARLY EXIT PERFORMANCE EVALUATION")
    print(f"Threshold: {EARLY_EXIT_THRESHOLD}")
    print("=" * 100)
    print()

    # Load RAG chain
    print("Loading RAG system...")
    _, retriever = load_rag_chain()
    print("RAG system loaded successfully.\n")

    # Results storage
    results = []

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"[{i}/{len(SAMPLE_QUESTIONS)}] Testing: {question[:60]}...")

        start_time = time.time()
        try:
            result = get_multi_llm_answer(question, retriever, user_context="Wet Zone | Yala Season (August)")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "question": question,
                "similarity_score": None,
                "early_exit": False,
                "response_time_ms": None,
                "estimated_without_ee_ms": None,
                "time_saved_ms": None,
                "error": str(e),
            })
            continue
        elapsed_ms = int((time.time() - start_time) * 1000)

        early_exit = result.get("early_exit", False)
        similarity_score = result.get("similarity_score", None)

        # Estimate time without early exit:
        # If early exit triggered, add estimated Judge LLM latency to get "without" time
        # If early exit didn't trigger, the actual time IS the "without" time
        if early_exit:
            estimated_without_ee_ms = elapsed_ms + int(ESTIMATED_JUDGE_LATENCY_SEC * 1000)
            time_saved_ms = int(ESTIMATED_JUDGE_LATENCY_SEC * 1000)
        else:
            estimated_without_ee_ms = elapsed_ms
            time_saved_ms = 0

        results.append({
            "question": question,
            "similarity_score": similarity_score,
            "early_exit": early_exit,
            "response_time_ms": elapsed_ms,
            "estimated_without_ee_ms": estimated_without_ee_ms,
            "time_saved_ms": time_saved_ms,
            "best_model": result.get("best_model", "unknown"),
        })

        status = "⚡ EARLY EXIT" if early_exit else "🔍 FULL JUDGE"
        sim_str = f"{similarity_score:.4f}" if similarity_score is not None else "N/A"
        print(f"  {status} | Similarity: {sim_str} | Time: {elapsed_ms}ms")
        print()

    # ============ Print Results Table ============
    print()
    print("=" * 120)
    print("RESULTS TABLE")
    print("=" * 120)

    # Header
    header = f"{'#':<3} {'Question':<50} {'Sim Score':<12} {'Early Exit':<12} {'Time (ms)':<12} {'Est. w/o EE':<14} {'Saved (ms)':<12}"
    print(header)
    print("-" * 120)

    early_exit_count = 0
    total_time_saved = 0
    total_actual_time = 0
    total_estimated_time = 0

    for i, r in enumerate(results, 1):
        q_short = r["question"][:48] + ".." if len(r["question"]) > 48 else r["question"]
        sim = f"{r['similarity_score']:.4f}" if r.get("similarity_score") is not None else "N/A"
        ee = "YES" if r["early_exit"] else "NO"
        t_ms = f"{r['response_time_ms']}" if r.get("response_time_ms") is not None else "ERR"
        est = f"{r['estimated_without_ee_ms']}" if r.get("estimated_without_ee_ms") is not None else "ERR"
        saved = f"{r['time_saved_ms']}" if r.get("time_saved_ms") is not None else "ERR"

        print(f"{i:<3} {q_short:<50} {sim:<12} {ee:<12} {t_ms:<12} {est:<14} {saved:<12}")

        if r["early_exit"]:
            early_exit_count += 1
        if r.get("time_saved_ms") is not None:
            total_time_saved += r["time_saved_ms"]
        if r.get("response_time_ms") is not None:
            total_actual_time += r["response_time_ms"]
        if r.get("estimated_without_ee_ms") is not None:
            total_estimated_time += r["estimated_without_ee_ms"]

    print("-" * 120)

    # Summary
    total_q = len(results)
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Questions Tested:       {total_q}")
    print(f"Early Exits Triggered:        {early_exit_count}/{total_q} ({early_exit_count/total_q*100:.0f}%)")
    print(f"Total Actual Time:            {total_actual_time}ms")
    print(f"Total Estimated (no EE):      {total_estimated_time}ms")
    print(f"Total Time Saved:             {total_time_saved}ms")
    if total_estimated_time > 0:
        pct_saved = (total_time_saved / total_estimated_time) * 100
        print(f"Overall Latency Reduction:    {pct_saved:.1f}%")
    avg_actual = total_actual_time / total_q if total_q > 0 else 0
    avg_estimated = total_estimated_time / total_q if total_q > 0 else 0
    print(f"Avg Response Time (with EE):  {avg_actual:.0f}ms")
    print(f"Avg Response Time (w/o EE):   {avg_estimated:.0f}ms")
    print("=" * 60)


if __name__ == "__main__":
    measure_early_exit_performance()
