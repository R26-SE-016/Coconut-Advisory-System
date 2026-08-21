"""
Early Exit Performance Evaluation Script
==========================================
Tests 10 sample coconut farming questions and measures the performance
impact of the early exit optimization in the Multi-LLM validation pipeline.

Features:
- Robust rate-limit resilience with 20s inter-question delay countdown
- Automatic retry on rate limit (with 30s cooldown)
- Detailed rate limit issue tracking and reporting

Usage:
    python -m evaluation.early_exit_eval
"""

import sys
import os
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()

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

# Estimated average Judge LLM call time in seconds (based on OpenRouter API latency)
ESTIMATED_JUDGE_LATENCY_SEC = 4.0
INTER_QUESTION_DELAY_SEC = 15
RATE_LIMIT_COOLDOWN_SEC = 30


def countdown_delay(seconds: int, message: str = "Waiting before next question..."):
    """Displays a dynamic countdown timer in the console."""
    for remaining in range(seconds, 0, -1):
        print(f"\r  ⏱️  {message} ({remaining}s remaining)... ", end="", flush=True)
        time.sleep(1)
    print(f"\r  ⏱️  {message} (Done)                            ", flush=True)


def is_rate_limit_result(result: dict, exception: Exception = None) -> bool:
    """Checks if a result or exception was caused by an API rate limit."""
    if exception is not None:
        err_msg = str(exception).lower()
        if "rate limit" in err_msg or "429" in err_msg or "tpm" in err_msg or "rpm" in err_msg:
            return True

    if isinstance(result, dict):
        answers = [
            result.get("llama_answer", ""),
            result.get("llama8b_answer", ""),
            result.get("gemma_answer", ""),
            result.get("best_answer", ""),
            result.get("reason", "")
        ]
        for a in answers:
            if isinstance(a, str) and "rate limit" in a.lower():
                return True

    return False


def measure_early_exit_performance():
    """
    Run 10 sample coconut farming questions through the Multi-LLM pipeline
    and record performance metrics for research evaluation.
    """
    print("=" * 100)
    print("EARLY EXIT PERFORMANCE EVALUATION (RATE LIMIT RESILIENT)")
    print(f"Threshold: {EARLY_EXIT_THRESHOLD} | Inter-question delay: {INTER_QUESTION_DELAY_SEC}s | Rate limit cooldown: {RATE_LIMIT_COOLDOWN_SEC}s")
    print("=" * 100)
    print()

    # Load RAG chain
    print("Loading RAG system...")
    _, retriever = load_rag_chain()
    print("RAG system loaded successfully.\n")

    # Results storage
    results = []
    rate_limited_questions = []

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"[{i}/{len(SAMPLE_QUESTIONS)}] Testing: {question}")

        start_time = time.time()
        result = None
        rate_limit_hit = False
        retried = False

        # Primary attempt
        try:
            result = get_multi_llm_answer(question, retriever, user_context="Wet Zone | Yala Season (August)")
            if is_rate_limit_result(result):
                rate_limit_hit = True
        except Exception as e:
            if is_rate_limit_result(None, e):
                rate_limit_hit = True
            else:
                print(f"  ❌ Error on question {i}: {e}")

        # Retry logic if rate limit detected
        if rate_limit_hit:
            print(f"  ⚠️  Rate limit detected on primary attempt! Applying {RATE_LIMIT_COOLDOWN_SEC}s cooldown...")
            countdown_delay(RATE_LIMIT_COOLDOWN_SEC, f"Rate limit cooldown for Q{i}")
            print(f"  🔄 Retrying question {i}...")
            retried = True
            start_time = time.time()  # Reset timer for retry
            try:
                result = get_multi_llm_answer(question, retriever, user_context="Wet Zone | Yala Season (August)")
                if is_rate_limit_result(result):
                    rate_limited_questions.append((i, question, "Rate limit persisted after retry"))
                else:
                    rate_limited_questions.append((i, question, "Resolved after 1 retry"))
            except Exception as retry_err:
                print(f"  ❌ Retry failed for question {i}: {retry_err}")
                rate_limited_questions.append((i, question, f"Failed on retry: {retry_err}"))

        elapsed_ms = int((time.time() - start_time) * 1000)

        if result is None:
            results.append({
                "question": question,
                "similarity_score": None,
                "early_exit": False,
                "response_time_ms": None,
                "estimated_without_ee_ms": None,
                "time_saved_ms": None,
                "error": "Failed after retry",
                "retried": retried
            })
            continue

        early_exit = result.get("early_exit", False)
        similarity_score = result.get("similarity_score", None)

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
            "retried": retried
        })

        status = "⚡ EARLY EXIT" if early_exit else "🔍 FULL JUDGE"
        sim_str = f"{similarity_score:.4f}" if similarity_score is not None else "N/A"
        retry_tag = " (Retried)" if retried else ""
        print(f"  {status} | Similarity: {sim_str} | Time: {elapsed_ms}ms{retry_tag}")

        # Inter-question delay countdown (only between questions, not after the last one)
        if i < len(SAMPLE_QUESTIONS):
            countdown_delay(INTER_QUESTION_DELAY_SEC, "Waiting before next question")
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

    # Rate limit report
    print()
    print("=" * 60)
    print("RATE LIMIT ISSUES REPORT")
    print("=" * 60)
    if rate_limited_questions:
        print(f"Detected {len(rate_limited_questions)} rate limit events during evaluation:")
        for q_id, q_text, status in rate_limited_questions:
            print(f"  • [Q{q_id}] \"{q_text[:45]}...\" -> {status}")
    else:
        print("✅ Zero rate limit issues detected across all 10 evaluation questions!")
    print("=" * 60)


if __name__ == "__main__":
    measure_early_exit_performance()

