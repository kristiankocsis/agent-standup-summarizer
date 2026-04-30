"""
Evaluation script for Golden Dataset.

Runs the agent against 10 labeled examples and measures accuracy.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import run_agent


def load_golden_dataset():
    """Load golden dataset from JSON file."""
    dataset_path = Path(__file__).parent.parent / "data" / "golden_dataset" / "golden_examples.json"
    with open(dataset_path) as f:
        return json.load(f)


def parse_agent_output(output_text):
    """Extract JSON from agent output (handles markdown code blocks)."""
    # Remove markdown code blocks if present
    if "```json" in output_text:
        start = output_text.find("```json") + 7
        end = output_text.find("```", start)
        output_text = output_text[start:end].strip()
    elif "```" in output_text:
        start = output_text.find("```") + 3
        end = output_text.find("```", start)
        output_text = output_text[start:end].strip()

    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        return None


def compare_arrays(actual, expected, field_name):
    """Compare two arrays and return match score (0-1)."""
    if not expected:
        return 1.0 if not actual else 0.5

    if not actual:
        return 0.0

    # Simple overlap-based scoring
    actual_set = set(str(x).lower() for x in actual)
    expected_set = set(str(x).lower() for x in expected)

    if not expected_set:
        return 1.0

    overlap = len(actual_set & expected_set)
    score = overlap / len(expected_set)  # Recall-based
    return min(score, 1.0)


def compare_blockers(actual, expected):
    """Compare blocker objects (more lenient matching)."""
    if not expected:
        return 1.0 if not actual else 0.5

    if not actual:
        return 0.0 if expected else 1.0

    # Check if at least one blocker description matches
    actual_descriptions = set(str(b.get("description", "")).lower() for b in actual)
    expected_descriptions = set(str(b.get("description", "")).lower() for b in expected)

    overlap = len(actual_descriptions & expected_descriptions)
    expected_count = len(expected_descriptions)

    return overlap / expected_count if expected_count > 0 else 1.0


def evaluate_example(example):
    """Evaluate a single golden dataset example."""
    input_text = example["input"]
    expected_output = example["expected_output"]

    # Run agent
    result = run_agent(input_text)

    if result["status"] != "success":
        return {
            "id": example["id"],
            "status": "error",
            "error": result.get("error"),
            "score": 0.0
        }

    # Parse agent output
    actual_output = parse_agent_output(result["output"])

    if not actual_output:
        return {
            "id": example["id"],
            "status": "parse_error",
            "raw_output": result["output"][:200],
            "score": 0.0
        }

    # Compare outputs
    scores = {}

    # Compare done
    scores["done"] = compare_arrays(
        actual_output.get("done", []),
        expected_output.get("done", []),
        "done"
    )

    # Compare in_progress
    scores["in_progress"] = compare_arrays(
        actual_output.get("in_progress", []),
        expected_output.get("in_progress", []),
        "in_progress"
    )

    # Compare blockers
    scores["blockers"] = compare_blockers(
        actual_output.get("blockers", []),
        expected_output.get("blockers", [])
    )

    # Compare actions (lenient - just check if present)
    actual_actions = actual_output.get("actions", [])
    expected_actions = expected_output.get("actions", [])
    scores["actions"] = 1.0 if bool(actual_actions) == bool(expected_actions) else 0.5

    # Overall score (average)
    overall_score = sum(scores.values()) / len(scores)

    return {
        "id": example["id"],
        "status": "success",
        "scores": scores,
        "overall_score": overall_score,
        "tokens_used": result.get("tokens_used")
    }


def run_eval_suite():
    """Run complete evaluation suite."""
    print("\n" + "="*70)
    print("GOLDEN DATASET EVALUATION SUITE")
    print("="*70)

    dataset = load_golden_dataset()
    print(f"\nLoaded {len(dataset)} examples from golden dataset\n")

    results = []

    for i, example in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] Evaluating example {example['id']}...", end=" ", flush=True)

        result = evaluate_example(example)
        results.append(result)

        if result["status"] == "success":
            score = result["overall_score"]
            status_icon = "[OK]" if score > 0.7 else "[~]" if score > 0.4 else "[FAIL]"
            print(f"{status_icon} {score:.1%}")
        else:
            print(f"[FAIL] {result['status']}")

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    if successful:
        scores = [r["overall_score"] for r in successful]
        avg_score = sum(scores) / len(scores)

        print(f"\nSuccessful: {len(successful)}/{len(results)}")
        print(f"Average score: {avg_score:.1%}")
        print(f"Range: {min(scores):.1%} - {max(scores):.1%}")

        print("\nBreakdown by category:")
        for category in ["done", "in_progress", "blockers", "actions"]:
            cat_scores = [r["scores"][category] for r in successful if category in r["scores"]]
            if cat_scores:
                cat_avg = sum(cat_scores) / len(cat_scores)
                print(f"  {category}: {cat_avg:.1%}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  Example {r['id']}: {r['status']}")

    print("\n" + "="*70)

    # Detailed results
    print("\nDETAILED RESULTS:")
    print("-"*70)
    for r in results:
        if r["status"] == "success":
            print(f"\nExample {r['id']}: {r['overall_score']:.1%}")
            for cat, score in r["scores"].items():
                print(f"  {cat}: {score:.1%}")

    # Write results to file
    results_file = Path(__file__).parent.parent / "data" / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": str(Path(__file__).stat().st_mtime),
            "total_examples": len(results),
            "successful": len(successful),
            "average_score": sum(r["overall_score"] for r in successful) / len(successful) if successful else 0,
            "results": results
        }, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    run_eval_suite()
