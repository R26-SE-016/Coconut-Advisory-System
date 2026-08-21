"""
BLEU & BERTScore Evaluation for SaruPol AI Sinhala Translation Pipeline
=======================================================================
Evaluates the quality of English → Sinhala translations produced by
translate_text() against expert-curated reference Sinhala translations
using:
  1. NLTK sentence-level and corpus-level BLEU scoring (n-gram overlap)
  2. Multilingual BERTScore (Precision, Recall, F1) for semantic similarity

Usage:
    python -m evaluation.bleu_eval
"""

import sys
import os
import json
import time
from datetime import datetime, timezone
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Paths
TEST_SENTENCES_PATH = os.path.join(PROJECT_ROOT, "evaluation", "bleu_test_sentences.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "bleu_results.json")

# Translation API delay between calls (seconds)
DELAY_BETWEEN_CALLS_SEC = 2
BERTSCORE_MODEL = "bert-base-multilingual-cased"


# ============================================================
# Sinhala Tokenizer for BLEU
# ============================================================

def tokenize_sinhala(text: str) -> List[str]:
    """
    Tokenize Sinhala text for BLEU evaluation.
    Strategy: split on whitespace first, then on individual Sinhala Unicode
    characters within each token. Numbers, abbreviations, and Latin text
    are kept as single tokens.
    """
    if not text:
        return []

    import re
    text = text.strip()

    # Normalize: remove trailing punctuation marks common in Sinhala
    text = re.sub(r'[.?,!;:]+$', '', text)

    # Split on spaces
    space_tokens = text.split()
    tokens = []

    for token in space_tokens:
        # If the token is purely Latin/digits/abbreviations, keep it as-is
        if re.match(r'^[A-Za-z0-9()/%\-+.]+$', token):
            tokens.append(token.lower())
        else:
            sinhala_chars = re.findall(r'[\u0D80-\u0DFF]+|[^\u0D80-\u0DFF\s]+', token)
            for sc in sinhala_chars:
                if re.match(r'^[A-Za-z0-9()/%\-+.]+$', sc):
                    tokens.append(sc.lower())
                else:
                    tokens.append(sc)

    return tokens


# ============================================================
# BLEU Score Calculation
# ============================================================

def calculate_sentence_bleu(reference: str, hypothesis: str) -> float:
    """Calculate sentence-level BLEU score between reference and hypothesis Sinhala texts."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    ref_tokens = tokenize_sinhala(reference)
    hyp_tokens = tokenize_sinhala(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    smoothie = SmoothingFunction().method1
    try:
        score = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            smoothing_function=smoothie
        )
        return round(score, 4)
    except Exception:
        return 0.0


def calculate_corpus_bleu(references: List[str], hypotheses: List[str]) -> float:
    """Calculate corpus-level BLEU score across all sentence pairs."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    ref_token_lists = []
    hyp_token_lists = []

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = tokenize_sinhala(ref)
        hyp_tokens = tokenize_sinhala(hyp)
        if ref_tokens and hyp_tokens:
            ref_token_lists.append([ref_tokens])
            hyp_token_lists.append(hyp_tokens)

    if not ref_token_lists:
        return 0.0

    smoothie = SmoothingFunction().method1
    try:
        score = corpus_bleu(
            ref_token_lists,
            hyp_token_lists,
            smoothing_function=smoothie
        )
        return round(score, 4)
    except Exception:
        return 0.0


def get_bleu_level(score: float) -> str:
    """Classify BLEU score into quality level."""
    if score >= 0.7:
        return "Excellent"
    elif score >= 0.5:
        return "Good"
    elif score >= 0.3:
        return "Acceptable"
    else:
        return "Poor"


# ============================================================
# BERTScore Calculation
# ============================================================

def calculate_bert_scores(references: List[str], hypotheses: List[str]) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate BERTScore (Precision, Recall, F1) using multilingual BERT.
    Measures semantic meaning similarity rather than exact surface token overlap.
    """
    from bert_score import score as bertscore_fn

    if not references or not hypotheses:
        return [], [], []

    try:
        P, R, F1 = bertscore_fn(
            cands=hypotheses,
            refs=references,
            model_type=BERTSCORE_MODEL,
            verbose=False
        )
        p_list = [round(float(p), 4) for p in P]
        r_list = [round(float(r), 4) for r in R]
        f1_list = [round(float(f), 4) for f in F1]
        return p_list, r_list, f1_list
    except Exception as e:
        print(f"  Warning: BERTScore computation encountered an error: {e}")
        zeroes = [0.0] * len(references)
        return zeroes, zeroes, zeroes


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def run_bleu_evaluation():
    """Run the complete BLEU & BERTScore evaluation pipeline."""
    from step2_rag_engine import translate_text

    print("=" * 90)
    print("  SaruPol AI — BLEU & BERTScore Evaluation for Sinhala Translation Pipeline")
    print("=" * 90)
    print()

    # Step 1: Load test sentences
    print("[Step 1] Loading test sentences from bleu_test_sentences.json...")
    with open(TEST_SENTENCES_PATH, "r", encoding="utf-8") as f:
        test_sentences = json.load(f)

    total = len(test_sentences)
    print(f"  Loaded {total} test sentences.\n")

    # Step 2: Translate each sentence and calculate BLEU scores
    print("[Step 2] Translating and computing sentence-level BLEU scores...\n")

    results = []
    valid_references = []
    valid_hypotheses = []
    valid_indices = []

    for i, item in enumerate(test_sentences):
        sid = item["id"]
        english = item["english"]
        reference = item.get("reference_sinhala", "")
        topic = item.get("topic", "general")

        # Skip if no reference
        if not reference or not reference.strip():
            print(f"  [{i+1}/{total}] ID {sid}: SKIPPED (no reference)")
            results.append({
                "id": sid,
                "english": english,
                "reference_sinhala": reference,
                "hypothesis_sinhala": "",
                "bleu_score": 0.0,
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
                "level": "Skipped",
                "topic": topic,
                "error": True
            })
            continue

        # Translate
        short_en = english[:55] + ("..." if len(english) > 55 else "")
        print(f"  [{i+1}/{total}] Translating: {short_en}")

        try:
            hypothesis = translate_text(english, "si")
            error = False
        except Exception as e:
            print(f"           ERROR: {e}")
            hypothesis = ""
            error = True

        if not hypothesis or not hypothesis.strip():
            print(f"           FAILED: Empty translation output.")
            results.append({
                "id": sid,
                "english": english,
                "reference_sinhala": reference,
                "hypothesis_sinhala": "",
                "bleu_score": 0.0,
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
                "level": "Failed",
                "topic": topic,
                "error": True
            })
            continue

        # Calculate BLEU
        bleu = calculate_sentence_bleu(reference, hypothesis)
        level = get_bleu_level(bleu)

        print(f"           BLEU: {bleu:.4f} ({level})")

        rec_idx = len(results)
        results.append({
            "id": sid,
            "english": english,
            "reference_sinhala": reference,
            "hypothesis_sinhala": hypothesis,
            "bleu_score": bleu,
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "level": level,
            "topic": topic,
            "error": error
        })

        valid_references.append(reference)
        valid_hypotheses.append(hypothesis)
        valid_indices.append(rec_idx)

        # Delay between translation calls
        if i < total - 1:
            time.sleep(DELAY_BETWEEN_CALLS_SEC)

    # Step 3: Compute BERTScore on all valid pairs
    print(f"\n[Step 3] Computing semantic BERTScore with {BERTSCORE_MODEL}...")
    p_scores, r_scores, f1_scores = calculate_bert_scores(valid_references, valid_hypotheses)

    for idx, p, r, f1 in zip(valid_indices, p_scores, r_scores, f1_scores):
        results[idx]["bertscore_precision"] = p
        results[idx]["bertscore_recall"] = r
        results[idx]["bertscore_f1"] = f1

    # Step 4: Corpus BLEU
    print(f"[Step 4] Calculating corpus-level BLEU score...")
    corpus_bleu_score = calculate_corpus_bleu(valid_references, valid_hypotheses)

    # Step 5: Topic-level Aggregations
    print(f"[Step 5] Calculating topic-level BLEU and BERTScore aggregations...\n")
    topic_bleu = defaultdict(list)
    topic_bert_f1 = defaultdict(list)

    for r in results:
        if not r.get("error", False):
            t = r["topic"]
            topic_bleu[t].append(r["bleu_score"])
            topic_bert_f1[t].append(r["bertscore_f1"])

    topic_averages = {}
    for topic in sorted(topic_bleu.keys()):
        b_scores = topic_bleu[topic]
        f_scores = topic_bert_f1[topic]
        avg_b = round(sum(b_scores) / len(b_scores), 4) if b_scores else 0.0
        avg_f = round(sum(f_scores) / len(f_scores), 4) if f_scores else 0.0
        topic_averages[topic] = {
            "average_bleu": avg_b,
            "average_bertscore_f1": avg_f,
            "count": len(b_scores),
            "level": get_bleu_level(avg_b)
        }

    # Step 6: Print formatted results table
    print("\n" + "=" * 105)
    print(f"{'ID':>4} | {'English (short)':36s} | {'BLEU':>6} | {'BERTScore F1':>12} | {'Level':>10} | {'Topic'}")
    print("-" * 105)

    for r in results:
        short = r["english"][:34] + ".." if len(r["english"]) > 34 else r["english"]
        if r.get("error"):
            print(f"{r['id']:4d} | {short:36s} | {'--':>6} | {'--':>12} | {'ERROR':>10} | {r['topic']}")
        else:
            print(f"{r['id']:4d} | {short:36s} | {r['bleu_score']:6.4f} | {r['bertscore_f1']:12.4f} | {r['level']:>10} | {r['topic']}")
    print("=" * 105)

    # Step 7: Print summary
    valid_results = [r for r in results if not r.get("error", False)]
    valid_bleu = [r["bleu_score"] for r in valid_results]
    valid_bert_p = [r["bertscore_precision"] for r in valid_results]
    valid_bert_r = [r["bertscore_recall"] for r in valid_results]
    valid_bert_f1 = [r["bertscore_f1"] for r in valid_results]

    avg_bleu = round(sum(valid_bleu) / len(valid_bleu), 4) if valid_bleu else 0.0
    avg_bert_p = round(sum(valid_bert_p) / len(valid_bert_p), 4) if valid_bert_p else 0.0
    avg_bert_r = round(sum(valid_bert_r) / len(valid_bert_r), 4) if valid_bert_r else 0.0
    avg_bert_f1 = round(sum(valid_bert_f1) / len(valid_bert_f1), 4) if valid_bert_f1 else 0.0

    best_topic = max(topic_averages.items(), key=lambda x: x[1]["average_bertscore_f1"]) if topic_averages else ("N/A", {"average_bertscore_f1": 0.0})
    worst_topic = min(topic_averages.items(), key=lambda x: x[1]["average_bertscore_f1"]) if topic_averages else ("N/A", {"average_bertscore_f1": 0.0})

    excellent = sum(1 for s in valid_bleu if s >= 0.7)
    good = sum(1 for s in valid_bleu if 0.5 <= s < 0.7)
    acceptable = sum(1 for s in valid_bleu if 0.3 <= s < 0.5)
    poor = sum(1 for s in valid_bleu if s < 0.3)

    print(f"\n{'='*70}")
    print(f"  BLEU & BERTSCORE EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total sentences:              {total}")
    print(f"  Successfully evaluated:       {len(valid_results)}")
    print(f"  Skipped / Failed:             {total - len(valid_results)}")
    print(f"")
    print(f"  Average BLEU Score:           {avg_bleu:.4f}")
    print(f"  Corpus BLEU Score:            {corpus_bleu_score:.4f}")
    print(f"  Average BERTScore Precision:  {avg_bert_p:.4f}")
    print(f"  Average BERTScore Recall:     {avg_bert_r:.4f}")
    print(f"  Average BERTScore F1:         {avg_bert_f1:.4f}")
    print(f"")
    print(f"  Best topic by BERTScore F1:   {best_topic[0]} ({best_topic[1]['average_bertscore_f1']:.4f})")
    print(f"  Worst topic by BERTScore F1:  {worst_topic[0]} ({worst_topic[1]['average_bertscore_f1']:.4f})")
    print(f"")
    print(f"  BLEU Level Distribution:")
    print(f"    • Excellent (>=0.7):        {excellent} sentences")
    print(f"    • Good (0.5-0.7):           {good} sentences")
    print(f"    • Acceptable (0.3-0.5):     {acceptable} sentences")
    print(f"    • Poor (<0.3):              {poor} sentences")
    print(f"{'='*70}")

    # Topic breakdown table
    print(f"\n  EVALUATION BY TOPIC:")
    print(f"  {'Topic':24s} | {'Avg BLEU':>8} | {'BERTScore F1':>12} | {'Count':>5}")
    print(f"  {'-'*65}")
    for topic, data in sorted(topic_averages.items(), key=lambda x: -x[1]["average_bertscore_f1"]):
        print(f"  {topic:24s} | {data['average_bleu']:8.4f} | {data['average_bertscore_f1']:12.4f} | {data['count']:5d}")
    print()

    # Step 8: Save results to JSON
    output = {
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "translation_model": "openai/gpt-4o-mini via OpenRouter",
        "bertscore_model": BERTSCORE_MODEL,
        "total_sentences": total,
        "evaluated_sentences": len(valid_results),
        "average_bleu": avg_bleu,
        "corpus_bleu": corpus_bleu_score,
        "average_bertscore_precision": avg_bert_p,
        "average_bertscore_recall": avg_bert_r,
        "average_bertscore_f1": avg_bert_f1,
        "best_topic_bertscore_f1": {
            "topic": best_topic[0],
            "score": best_topic[1]["average_bertscore_f1"]
        },
        "worst_topic_bertscore_f1": {
            "topic": worst_topic[0],
            "score": worst_topic[1]["average_bertscore_f1"]
        },
        "bleu_level_distribution": {
            "excellent_gte_0.7": excellent,
            "good_0.5_0.7": good,
            "acceptable_0.3_0.5": acceptable,
            "poor_lt_0.3": poor
        },
        "topic_scores": topic_averages,
        "sentence_results": results
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[Step 8] Results saved to: {RESULTS_PATH}")
    print(f"\nDone! BLEU & BERTScore evaluation complete.")


if __name__ == "__main__":
    # Ensure NLTK punkt tokenizer is available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        import nltk
        nltk.download('punkt', quiet=True)
    except ImportError:
        print("ERROR: nltk is not installed. Run: pip install nltk")
        sys.exit(1)

    run_bleu_evaluation()
