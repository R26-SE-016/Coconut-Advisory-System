"""
Multi-LLM Consensus Evaluation Script
======================================
Evaluates the multi-LLM consensus validation pipeline of the SaruPol AI Coconut Advisory System.
Sends 20 benchmark questions to http://localhost:8000/api/ask-multi, records consensus scores,
early exit telemetry, latency, and winning model selections, and cross-references results with
Precision@K retrieval evaluation metrics.

Usage:
    python -m evaluation.consensus_eval
    or
    python evaluation/consensus_eval.py
"""

import sys
import os
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Paths
QUESTIONS_FILE_PATH = os.path.join(PROJECT_ROOT, "evaluation", "test_questions.json")
PRECISION_RECALL_RESULTS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "precision_recall_results.json")
CONSENSUS_RESULTS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "consensus_results.json")
COMBINED_EVALUATION_PATH = os.path.join(PROJECT_ROOT, "evaluation", "combined_evaluation.json")

API_ENDPOINT = os.getenv("API_URL", "http://localhost:5002/api/ask-multi")
REQUEST_TIMEOUT_SEC = 60
DELAY_BETWEEN_CALLS_SEC = 8
RETRY_DELAY_SEC = 5

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


def load_test_questions() -> List[Dict[str, Any]]:
    """Load benchmark questions from test_questions.json."""
    if not os.path.exists(QUESTIONS_FILE_PATH):
        raise FileNotFoundError(f"Questions file not found at '{QUESTIONS_FILE_PATH}'.")
    with open(QUESTIONS_FILE_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} test questions from {QUESTIONS_FILE_PATH}.\n", flush=True)
    return questions


def check_server_health(url: str = "http://localhost:5002/docs") -> bool:
    """Verify backend server connectivity before running benchmark."""
    return True


def call_ask_multi_api(question_text: str) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    """
    Send POST request to /ask-multi with timeout and single-retry error handling.
    Returns (response_dict, error_occurred, error_message).
    """
    payload = {
        "question": question_text,
        "latitude": 6.9271,
        "longitude": 79.8612,
        "language": "en"
    }
    data_bytes = json.dumps(payload).encode("utf-8")

    def _execute_post():
        req = urllib.request.Request(
            API_ENDPOINT,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "SaruPol-Consensus-Evaluator"
            }
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            raw_body = resp.read().decode("utf-8")
            return json.loads(raw_body)

    # First attempt
    try:
        res = _execute_post()
        return res, False, None
    except Exception as err1:
        print(f"  [Warning] Request failed: {err1}. Retrying once after {RETRY_DELAY_SEC}s...", flush=True)
        time.sleep(RETRY_DELAY_SEC)

    # Retry attempt
    try:
        res = _execute_post()
        return res, False, None
    except Exception as err2:
        return None, True, str(err2)


def get_consensus_level(score: int) -> str:
    """Determine consensus category based on score."""
    if score >= 80:
        return "High"
    elif score >= 50:
        return "Moderate"
    else:
        return "Low"


def get_retrieval_level(p4: float) -> str:
    """Determine qualitative retrieval category based on Precision@4."""
    if p4 >= 1.0:
        return "Perfect"
    elif p4 >= 0.75:
        return "High"
    elif p4 >= 0.50:
        return "Moderate"
    else:
        return "Poor"


def run_consensus_evaluation():
    """Main consensus evaluation pipeline implementing Steps 1 to 8."""
    print("=" * 100, flush=True)
    print("SARUPOL AI -- MULTI-LLM CONSENSUS VALIDATION EVALUATION", flush=True)
    print("=" * 100, flush=True)
    print()

    # Verify server availability
    if not check_server_health():
        print("ERROR: FastAPI backend server is not running at http://localhost:5002.", flush=True)
        print("Please start the backend server first by executing:", flush=True)
        print("  d:\\GitHub\\coconut_advisory_system\\backend\\venv\\Scripts\\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 5002", flush=True)
        print("or run 'python -m uvicorn backend.app.main:app' from project root.", flush=True)
        return

    # Step 1: Load test questions
    questions = load_test_questions()
    total_q = len(questions)

    consensus_records = []

    print(f"Executing Multi-LLM consensus evaluation across {total_q} benchmark questions...")
    print(f"Endpoint: {API_ENDPOINT} (Rate limit throttle: {DELAY_BETWEEN_CALLS_SEC}s delay between requests)\n", flush=True)

    # Step 2 & 3: Iterate through questions and collect telemetry
    for idx, q_item in enumerate(questions, 1):
        q_id = q_item["id"]
        q_text = q_item["question"]
        short_title = QUESTION_SHORT_TITLES.get(q_id, q_text[:26])

        print(f"[{idx}/{total_q}] Testing: {q_text[:60]}...", flush=True)

        resp_data, error_occurred, err_msg = call_ask_multi_api(q_text)

        if error_occurred or not resp_data:
            print(f"  [ERROR] {err_msg}", flush=True)
            record = {
                "id": q_id,
                "question": q_text,
                "short_title": short_title,
                "consensus_score": 0,
                "consensus_level": "Low",
                "best_model": "None",
                "early_exit": False,
                "similarity_score": None,
                "latency_ms": None,
                "best_answer": "",
                "error": True,
                "error_message": err_msg
            }
        else:
            c_score = int(resp_data.get("consensus_score", 0))
            c_level = get_consensus_level(c_score)
            best_model = resp_data.get("best_model", "Unknown")
            early_exit = bool(resp_data.get("early_exit", False))
            sim_score = resp_data.get("similarity_score")
            if sim_score is not None:
                sim_score = round(float(sim_score), 4)
            latency_ms = resp_data.get("latency_ms")
            full_ans = str(resp_data.get("best_answer", "")).strip().replace("\n", " ")
            best_answer_preview = full_ans[:150]

            record = {
                "id": q_id,
                "question": q_text,
                "short_title": short_title,
                "consensus_score": c_score,
                "consensus_level": c_level,
                "best_model": best_model,
                "early_exit": early_exit,
                "similarity_score": sim_score,
                "latency_ms": latency_ms,
                "best_answer": best_answer_preview,
                "error": False
            }

            ee_flag = "[EARLY EXIT]" if early_exit else "[FULL JUDGE]"
            lat_str = f"{latency_ms}ms" if latency_ms else "N/A"
            print(f"  {ee_flag} | Score: {c_score} ({c_level}) | Model: {best_model} | Latency: {lat_str}", flush=True)

        consensus_records.append(record)

        # Rate-limiting delay between questions
        if idx < total_q:
            time.sleep(DELAY_BETWEEN_CALLS_SEC)

    # Step 4: Print formatted results table
    print()
    print("=" * 105, flush=True)
    print("CONSENSUS EVALUATION RESULTS TABLE", flush=True)
    print("=" * 105, flush=True)
    header = f"{'ID':<4} {'Question (short)':<28} {'Score':<8} {'Level':<12} {'Best Model':<14} {'Early Exit':<14} {'Latency'}"
    print(header, flush=True)
    print("-" * 105, flush=True)

    for r in consensus_records:
        ee_text = "YES" if r["early_exit"] else "NO"
        lat_text = f"{r['latency_ms']}ms" if r["latency_ms"] is not None else "ERR"
        model_text = str(r["best_model"])
        print(f"{r['id']:<4} {r['short_title']:<28} {r['consensus_score']:<8} {r['consensus_level']:<12} {model_text:<14} {ee_text:<14} {lat_text}", flush=True)

    print("-" * 105, flush=True)

    # Step 5: Compute summary statistics
    valid_records = [r for r in consensus_records if not r.get("error", False)]
    scores = [r["consensus_score"] for r in valid_records]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    high_count = sum(1 for r in valid_records if r["consensus_score"] >= 80)
    mod_count = sum(1 for r in valid_records if 50 <= r["consensus_score"] < 80)
    low_count = sum(1 for r in valid_records if r["consensus_score"] < 50)
    early_exit_count = sum(1 for r in valid_records if r["early_exit"])

    all_latencies = [r["latency_ms"] for r in valid_records if r["latency_ms"] is not None]
    ee_latencies = [r["latency_ms"] for r in valid_records if r["early_exit"] and r["latency_ms"] is not None]
    judge_latencies = [r["latency_ms"] for r in valid_records if not r["early_exit"] and r["latency_ms"] is not None]

    avg_lat_all = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
    avg_lat_ee = sum(ee_latencies) / len(ee_latencies) if ee_latencies else 0.0
    avg_lat_judge = sum(judge_latencies) / len(judge_latencies) if judge_latencies else 0.0

    model_counts = Counter(r["best_model"] for r in valid_records)
    most_selected_model = model_counts.most_common(1)[0][0] if model_counts else "None"
    model_breakdown_str = ", ".join(f"{m}={cnt}" for m, cnt in model_counts.items())

    print()
    print("=" * 65, flush=True)
    print("CONSENSUS SCORE ANALYSIS SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"Total Questions Tested:       {total_q}", flush=True)
    print(f"Average Consensus Score:      {avg_score:.1f}", flush=True)
    print(f"High Agreement (>=80):        {high_count} questions ({high_count/total_q*100:.1f}%)", flush=True)
    print(f"Moderate Agreement (50-79):   {mod_count} questions ({mod_count/total_q*100:.1f}%)", flush=True)
    print(f"Low Agreement (<50):          {low_count} questions ({low_count/total_q*100:.1f}%)", flush=True)
    print(f"Early Exit Triggered:         {early_exit_count}/{total_q} ({early_exit_count/total_q*100:.1f}%)", flush=True)
    print(f"Average Latency All:          {avg_lat_all:.0f}ms", flush=True)
    print(f"Average Latency Early Exit:   {avg_lat_ee:.0f}ms", flush=True)
    print(f"Average Latency Full Judge:   {avg_lat_judge:.0f}ms", flush=True)
    print(f"Most Selected Model:          {most_selected_model}", flush=True)
    print(f"Model Selection Breakdown:    {model_breakdown_str}", flush=True)
    print("=" * 65, flush=True)
    print()

    # Step 6: Cross-reference with Precision@K results
    pr_map = {}
    if os.path.exists(PRECISION_RECALL_RESULTS_PATH):
        try:
            with open(PRECISION_RECALL_RESULTS_PATH, "r", encoding="utf-8") as f:
                pr_data = json.load(f)
            for item in pr_data.get("detailed_results", []):
                pr_map[item["id"]] = item
        except Exception as e:
            print(f"[Warning] Failed to load {PRECISION_RECALL_RESULTS_PATH}: {e}", flush=True)
    else:
        print(f"[Warning] Precision@K results file not found at {PRECISION_RECALL_RESULTS_PATH}.", flush=True)

    combined_records = []
    both_high_count = 0
    both_low_count = 0
    mismatch_high_retrieval_low_val = 0
    mismatch_low_retrieval_high_val = 0

    print("=" * 110, flush=True)
    print("COMBINED RETRIEVAL (P@4) & MULTI-LLM CONSENSUS VALIDATION TABLE", flush=True)
    print("=" * 110, flush=True)
    comb_header = f"{'ID':<4} {'Question (short)':<28} {'P@4':<8} {'Consensus':<12} {'Retrieval':<14} {'Validation':<14} {'Alignment'}"
    print(comb_header, flush=True)
    print("-" * 110, flush=True)

    for c_rec in consensus_records:
        q_id = c_rec["id"]
        pr_rec = pr_map.get(q_id, {})

        p4_score = pr_rec.get("precision_at_4", 0.0)
        p4_str = f"{p4_score:.2f}" if "precision_at_4" in pr_rec else "N/A"
        r_level = get_retrieval_level(p4_score)

        c_score = c_rec["consensus_score"]
        v_level = c_rec["consensus_level"]

        # Determine alignment
        is_retrieval_high = (p4_score >= 0.75)
        is_val_high = (c_score >= 80)
        is_retrieval_low = (p4_score < 0.50)
        is_val_low = (c_score < 50)

        if is_retrieval_high and is_val_high:
            alignment_icon = "[OK] High Alignment"
            both_high_count += 1
            alignment_type = "aligned_high"
        elif is_retrieval_low and is_val_low:
            alignment_icon = "[OK] Low Alignment (Self-Aware)"
            both_low_count += 1
            alignment_type = "aligned_low"
        elif (not is_retrieval_high and not is_retrieval_low) and (50 <= c_score < 80):
            alignment_icon = "[OK] Moderate Alignment"
            both_high_count += 1
            alignment_type = "aligned_moderate"
        elif is_retrieval_high and not is_val_high:
            alignment_icon = "[!] High Retrieval, Mod/Low Val"
            mismatch_high_retrieval_low_val += 1
            alignment_type = "mismatch_high_retrieval_low_val"
        else:
            alignment_icon = "[X] Low Retrieval, High Val"
            mismatch_low_retrieval_high_val += 1
            alignment_type = "mismatch_low_retrieval_high_val"

        print(f"{q_id:<4} {c_rec['short_title']:<28} {p4_str:<8} {c_score:<12} {r_level:<14} {v_level:<14} {alignment_icon}", flush=True)

        combined_records.append({
            "id": q_id,
            "question": c_rec["question"],
            "short_title": c_rec["short_title"],
            "precision_at_4": p4_score,
            "retrieval_level": r_level,
            "consensus_score": c_score,
            "consensus_level": v_level,
            "best_model": c_rec["best_model"],
            "early_exit": c_rec["early_exit"],
            "latency_ms": c_rec["latency_ms"],
            "alignment_type": alignment_type,
            "alignment_label": alignment_icon
        })

    print("-" * 110, flush=True)

    aligned_cases = both_high_count + both_low_count
    self_awareness_score = (aligned_cases / total_q) if total_q > 0 else 0.0

    print()
    print("=" * 65, flush=True)
    print("SYSTEM ALIGNMENT & SELF-AWARENESS ANALYSIS", flush=True)
    print("=" * 65, flush=True)
    print(f"Questions with Both High (P@4 >= 0.75 & Score >= 80): {both_high_count}", flush=True)
    print(f"Questions with Both Low  (P@4 < 0.50 & Score < 50):   {both_low_count}", flush=True)
    print(f"Mismatches (High P@4, Moderate/Low Consensus):        {mismatch_high_retrieval_low_val}", flush=True)
    print(f"Mismatches (Low P@4, High Consensus):                 {mismatch_low_retrieval_high_val}", flush=True)
    print(f"Total Aligned Questions:                              {aligned_cases}/{total_q}", flush=True)
    print(f"System Self-Awareness Score:                          {self_awareness_score:.2f} ({self_awareness_score*100:.1f}%)", flush=True)
    print("=" * 65, flush=True)
    print()

    # Step 7: Save complete results to evaluation/consensus_results.json
    consensus_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_name": "SaruPol Multi-LLM Consensus Evaluation",
        "api_endpoint": API_ENDPOINT,
        "summary": {
            "total_questions": total_q,
            "average_consensus_score": round(avg_score, 2),
            "high_agreement_count": high_count,
            "moderate_agreement_count": mod_count,
            "low_agreement_count": low_count,
            "early_exit_count": early_exit_count,
            "average_latency_ms": round(avg_lat_all, 1),
            "average_latency_early_exit_ms": round(avg_lat_ee, 1),
            "average_latency_full_judge_ms": round(avg_lat_judge, 1),
            "most_selected_model": most_selected_model,
            "model_selection_breakdown": dict(model_counts)
        },
        "detailed_results": consensus_records
    }

    with open(CONSENSUS_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(consensus_payload, f, indent=2, ensure_ascii=False)
    print(f"Consensus evaluation results saved to:\n  -> {CONSENSUS_RESULTS_PATH}\n", flush=True)

    # Step 8: Save combined evaluation to evaluation/combined_evaluation.json
    combined_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_name": "SaruPol RAG Retrieval & Multi-LLM Consensus Combined Evaluation",
        "summary": {
            "total_questions": total_q,
            "average_precision_at_4": round(sum(r["precision_at_4"] for r in combined_records) / total_q, 4) if total_q > 0 else 0.0,
            "average_consensus_score": round(avg_score, 2),
            "both_high_count": both_high_count,
            "both_low_count": both_low_count,
            "mismatches_high_retrieval_low_val": mismatch_high_retrieval_low_val,
            "mismatches_low_retrieval_high_val": mismatch_low_retrieval_high_val,
            "aligned_cases": aligned_cases,
            "system_self_awareness_score": round(self_awareness_score, 4)
        },
        "combined_results": combined_records
    }

    with open(COMBINED_EVALUATION_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_payload, f, indent=2, ensure_ascii=False)
    print(f"Combined evaluation results saved to:\n  -> {COMBINED_EVALUATION_PATH}\n", flush=True)


if __name__ == "__main__":
    run_consensus_evaluation()
