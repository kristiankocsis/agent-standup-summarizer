"""
Evaluation script for Golden Dataset — 30 examples.

Scoring uses token overlap (Jaccard similarity) for semantic matching
instead of exact string comparison. Structured output is read directly
from the agent's state via result["structured_output"], not parsed from
markdown text.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import run_agent


# ── Data loading ───────────────────────────────────────────────────────────

def load_golden_dataset() -> list[dict]:
    path = Path(__file__).parent.parent / "data" / "golden_dataset" / "golden_examples.json"
    with open(path) as f:
        return json.load(f)


# ── Similarity helpers ─────────────────────────────────────────────────────

def _tokens(text: str) -> set[str]:
    """Lowercase word tokens from a string, stripping punctuation."""
    import re
    return set(re.findall(r"\b\w+\b", text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _best_match(needle: str, haystack: list[str]) -> float:
    """Best Jaccard score between needle and any item in haystack."""
    if not haystack:
        return 0.0
    return max(_jaccard(needle, h) for h in haystack)


# ── Per-field scoring ──────────────────────────────────────────────────────

JACCARD_THRESHOLD = 0.25  # minimum similarity to count as a match


def score_text_list(actual: list, expected: list) -> float:
    """
    Recall-oriented: for each expected item, find the best matching
    actual item. Score = fraction of expected items that have a match
    above threshold.
    """
    if not expected and not actual:
        return 1.0

    if not expected:
        # Expected empty but agent returned items — hallucination penalty
        return 0.7

    if not actual:
        return 0.0

    actual_str = [str(x) for x in actual]
    hits = sum(
        1 for exp in expected
        if _best_match(str(exp), actual_str) >= JACCARD_THRESHOLD
    )
    return hits / len(expected)


def score_blockers(actual: list, expected: list) -> float:
    """
    Match blocker objects by description similarity.
    Recall over expected blockers; also checks false-positive rate.
    """
    if not expected and not actual:
        return 1.0

    if not expected:
        # Agent hallucinated blockers — partial penalty
        return max(0.0, 1.0 - 0.2 * len(actual))

    if not actual:
        return 0.0

    actual_desc = [str(b.get("description", b)) for b in actual]
    hits = sum(
        1 for exp in expected
        if _best_match(str(exp.get("description", exp)), actual_desc) >= JACCARD_THRESHOLD
    )
    return hits / len(expected)


def score_actions(actual: list, expected: list) -> float:
    """
    Match action objects by task description similarity.
    """
    if not expected and not actual:
        return 1.0

    if not expected:
        return 0.7  # hallucinated actions

    if not actual:
        return 0.0

    actual_tasks = [str(a.get("task", a)) for a in actual]
    hits = sum(
        1 for exp in expected
        if _best_match(str(exp.get("task", exp)), actual_tasks) >= JACCARD_THRESHOLD
    )
    return hits / len(expected)


# ── Single-example evaluation ──────────────────────────────────────────────

def evaluate_example(example: dict) -> dict:
    expected = example["expected_output"]

    t0 = time.time()
    result = run_agent(example["input"])
    elapsed = time.time() - t0

    base = {
        "id": example["id"],
        "scenario": example.get("scenario", ""),
        "elapsed_s": round(elapsed, 2),
    }

    if result["status"] != "success":
        return {**base, "status": "agent_error", "error": result.get("error"), "score": 0.0}

    actual = result.get("structured_output")
    if actual is None:
        # Agent didn't call format_output — structural failure
        return {**base, "status": "no_structured_output", "score": 0.0}

    scores = {
        "done":        score_text_list(actual.get("done", []),        expected.get("done", [])),
        "in_progress": score_text_list(actual.get("in_progress", []), expected.get("in_progress", [])),
        "blockers":    score_blockers( actual.get("blockers", []),     expected.get("blockers", [])),
        "actions":     score_actions(  actual.get("actions", []),      expected.get("actions", [])),
    }
    overall = sum(scores.values()) / len(scores)

    return {
        **base,
        "status": "success",
        "scores": scores,
        "score": overall,
    }


# ── Full suite ─────────────────────────────────────────────────────────────

def run_eval_suite(limit: int | None = None):
    print("\n" + "=" * 70)
    print("GOLDEN DATASET EVALUATION  —  token-overlap (Jaccard) scoring")
    print("=" * 70)

    dataset = load_golden_dataset()
    if limit:
        dataset = dataset[:limit]

    print(f"Running {len(dataset)} examples ...\n")

    results = []
    total_elapsed = 0.0

    for i, example in enumerate(dataset, 1):
        print(f"[{i:>2}/{len(dataset)}] {example['id']:>3} {example.get('scenario', ''):<40}", end=" ", flush=True)

        r = evaluate_example(example)
        results.append(r)
        total_elapsed += r.get("elapsed_s", 0)

        if r["status"] == "success":
            s = r["score"]
            icon = "[OK]  " if s >= 0.75 else "[~]   " if s >= 0.45 else "[FAIL]"
            print(f"{icon} {s:.0%}  ({r['elapsed_s']:.1f}s)")
        else:
            print(f"[FAIL] {r['status']}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    ok = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    if ok:
        avg = sum(r["score"] for r in ok) / len(ok)
        scores_list = [r["score"] for r in ok]
        print(f"\n  Passed     : {len(ok)}/{len(results)}")
        print(f"  Avg score  : {avg:.1%}")
        print(f"  Range      : {min(scores_list):.1%} – {max(scores_list):.1%}")
        print(f"  Total time : {total_elapsed:.1f}s")

        print("\n  Category breakdown:")
        for cat in ("done", "in_progress", "blockers", "actions"):
            cat_scores = [r["scores"][cat] for r in ok if cat in r.get("scores", {})]
            if cat_scores:
                print(f"    {cat:<14} {sum(cat_scores)/len(cat_scores):.1%}")

    if failed:
        print(f"\n  Failed: {len(failed)}")
        for r in failed:
            print(f"    example {r['id']} ({r['scenario']}): {r['status']}")

    # ── Detailed per-example breakdown ────────────────────────────────────
    print("\n" + "-" * 70)
    print("DETAIL")
    print("-" * 70)
    for r in results:
        if r["status"] == "success":
            cats = "  ".join(f"{k}={v:.0%}" for k, v in r["scores"].items())
            print(f"  {r['id']:>3} {r['scenario']:<38} [{r['score']:.0%}]  {cats}")
        else:
            print(f"  {r['id']:>3} {r['scenario']:<38} [{r['status']}]")

    # ── Persist results ───────────────────────────────────────────────────
    out_path = Path(__file__).parent.parent / "data" / "eval_results.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "success": len(ok),
        "avg_score": sum(r["score"] for r in ok) / len(ok) if ok else 0,
        "total_elapsed_s": round(total_elapsed, 2),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults saved -> {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run golden dataset evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N examples")
    args = parser.parse_args()
    run_eval_suite(limit=args.limit)
