"""
Semantic judge: asks a model whether an agent item means the same as an
expected golden item. Token overlap can't see synonyms, so it rejects correct
answers that use other words.

An experiment, not the official score: it did not pass calibration on unseen
pairs (see README). Both commands make paid API calls:
    python -m tests.semantic_judge --pairs data/golden_dataset/judge_calibration_3.json
    python -m tests.semantic_judge --rescore
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JUDGE_MODEL = "claude-haiku-5-5"
DATA_DIR = Path(__file__).parent.parent / "data"
CALIBRATION_PATH = DATA_DIR / "golden_dataset" / "judge_calibration.json"
RESULTS_PATH = DATA_DIR / "eval_results.json"
SEMANTIC_RESULTS_PATH = DATA_DIR / "eval_results_semantic.json"
PASS_AGREEMENT = 0.9
CATEGORIES = ("done", "in_progress", "blockers", "actions")

_PROMPT = """You are checking the output of a tool that summarizes standup meetings.

A is what the tool was expected to capture. B is what it wrote.
Would a Scrum Master be satisfied that B covers A?

Rules:
- Ignore a person's name before a colon, such as "Uma:".
- A missing detail such as place, method or purpose is fine if it stays clear what has to be done.
- A missing part of the task is fine if the main outcome remains. If the main outcome is missing, it is not a match. Approving is its own outcome, not an implied part of a review.
- Different words are fine if they mean the same work and the same status. A verb form alone, such as review and reviewing, does not matter.
- Different activities are not a match. Debugging finds a cause and may be passed on, while fixing repairs it. Addressing a problem expects it to be fixed, so address and fix match.
- A task still to do and work already finished are not a match. Use the section to read A: under done, A describes finished work even if it reads like a task name.
- How far a blocker reaches is not a mere detail. A blocker on one kind of work and a blocker on all related work are not a match.

Section: {section}
A: {expected}
B: {answer}

Answer with only YES or NO."""


class JudgeError(RuntimeError):
    """The judge gave no usable answer, so its verdict can't be counted."""


def _default_client():
    from anthropic import Anthropic
    from src.config import ANTHROPIC_API_KEY
    return Anthropic(api_key=ANTHROPIC_API_KEY)


def judge_pair(expected: str, answer: str, section: str | None = None, client=None, model: str = JUDGE_MODEL) -> bool:
    """Return True if the judge says the answer covers the expected item."""
    client = client or _default_client()
    prompt = _PROMPT.format(section=section or "not given", expected=expected, answer=answer)
    response = client.messages.create(
        model=model,
        max_tokens=2048,  # thinking tokens share this budget
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason != "end_turn":
        raise JudgeError(f"The judge did not finish (stop_reason: {response.stop_reason}).")
    text = next((b.text for b in response.content if b.type == "text"), "")
    match = re.fullmatch(r"\s*(yes|no)\W*", text, re.IGNORECASE)
    if not match:
        raise JudgeError(f"The judge gave an unclear answer: {text[:80]!r}")
    return match.group(1).lower() == "yes"


def agreement_report(pairs: list[dict], verdicts: list[bool]) -> dict:
    """Compare judge verdicts with the human labels in pairs."""
    total = len(pairs)
    agreed = sum(p["same_meaning"] == v for p, v in zip(pairs, verdicts))
    false_accepts = sum(v and not p["same_meaning"] for p, v in zip(pairs, verdicts))
    false_rejects = sum(p["same_meaning"] and not v for p, v in zip(pairs, verdicts))
    wrong_accepted = sum(v and p["kind"] in ("wrong", "hard") for p, v in zip(pairs, verdicts))
    return {
        "agreed": agreed,
        "total": total,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "wrong_pairs_accepted": wrong_accepted,
        "passed": agreed / total >= PASS_AGREEMENT and wrong_accepted == 0,
    }


def _texts(category: str, items: list) -> list[str]:
    field = {"blockers": "description", "actions": "task"}.get(category)
    return [str(i.get(field, i)) if field and isinstance(i, dict) else str(i) for i in items]


def score_category(category: str, actual: list, expected: list, matches) -> float:
    """Same rules as tests.eval_golden_dataset: only the matching differs."""
    if not expected and not actual:
        return 1.0
    if not expected:
        return max(0.0, 1.0 - 0.2 * len(actual)) if category == "blockers" else 0.7
    if not actual:
        return 0.0
    answers = _texts(category, actual)
    hits = sum(any(matches(e, a) for a in answers) for e in _texts(category, expected))
    return hits / len(expected)


def score_example(actual: dict, expected: dict, matches) -> dict:
    return {c: score_category(c, actual.get(c, []), expected.get(c, []), matches) for c in CATEGORIES}


def unmatched_items(actual: dict, expected: dict, matches) -> int:
    """Agent items that match no expected item. The score doesn't count these."""
    total = 0
    for category in CATEGORIES:
        targets = _texts(category, expected.get(category, []))
        total += sum(
            not any(matches(e, a) for e in targets)
            for a in _texts(category, actual.get(category, []))
        )
    return total


def run_rescore(runs: int = 2, workers: int = 5) -> None:
    """Judge every expected and agent item pair in the stored outputs."""
    from tests.eval_golden_dataset import load_golden_dataset

    baseline = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    examples = {e["id"]: e for e in load_golden_dataset()}
    rows = baseline["results"]
    sections = {
        (e, a): c
        for row in rows
        for c in CATEGORIES
        for e in _texts(c, examples[row["id"]]["expected_output"].get(c, []))
        for a in _texts(c, row["actual"].get(c, []))
    }
    pairs = sorted(sections)
    client = _default_client()
    print(f"unique pairs: {len(pairs)}, runs: {runs}")

    output = {"judge_model": JUDGE_MODEL, "agent_model": baseline["model"], "runs": []}
    for run in range(1, runs + 1):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(lambda p: _safe_judge(p, sections[p], client), pairs))
        errors = [o for o in outcomes if isinstance(o, str)]
        verdicts = {p: o is True for p, o in zip(pairs, outcomes)}
        matches = lambda e, a: verdicts[(e, a)]
        results = []
        for row in rows:
            expected = examples[row["id"]]["expected_output"]
            scores = score_example(row["actual"], expected, matches)
            results.append({
                "id": row["id"],
                "scenario": row.get("scenario", ""),
                "scores": scores,
                "score": sum(scores.values()) / len(scores),
                "unmatched_items": unmatched_items(row["actual"], expected, matches),
            })
        average = sum(r["score"] for r in results) / len(results)
        categories = {c: sum(r["scores"][c] for r in results) / len(results) for c in CATEGORIES}
        output["runs"].append({
            "avg_score": average,
            "category_avgs": categories,
            "unmatched_items": sum(r["unmatched_items"] for r in results),
            "judge_errors": errors,
            "results": results,
            "verdicts": [{"expected": e, "answer": a, "same": v} for (e, a), v in verdicts.items()],
        })
        SEMANTIC_RESULTS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"run {run}: avg {average:.4f} {categories} "
              f"unmatched {output['runs'][-1]['unmatched_items']} errors {len(errors)}")
    if runs > 1:
        first, second = output["runs"][0]["verdicts"], output["runs"][1]["verdicts"]
        print(f"pairs where the runs differ: {sum(x['same'] != y['same'] for x, y in zip(first, second))}/{len(pairs)}")


def _safe_judge(pair: tuple[str, str], section: str, client):
    """A failed call counts as no match and is reported, so one can't stop a paid run."""
    try:
        return judge_pair(pair[0], pair[1], section=section, client=client)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def run_calibration(runs: int = 2, model: str = JUDGE_MODEL, path: Path = CALIBRATION_PATH) -> None:
    pairs = json.loads(Path(path).read_text(encoding="utf-8"))
    client = _default_client()
    print(f"model: {model}")
    all_verdicts = []
    for run in range(1, runs + 1):
        verdicts = [
            judge_pair(p["expected"], p["answer"], section=p.get("category"), client=client, model=model)
            for p in pairs
        ]
        all_verdicts.append(verdicts)
        print(f"run {run}: {agreement_report(pairs, verdicts)}")
        for p, v in zip(pairs, verdicts):
            if p["same_meaning"] != v:
                print(f"  #{p['n']} {p['kind']} human={p['same_meaning']} judge={v} | "
                      f"{p['expected']} | {p['answer']}")
    if runs > 1:
        changed = sum(len(set(column)) > 1 for column in zip(*all_verdicts))
        print(f"pairs where the runs differ: {changed}/{len(pairs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the judge, or rescore stored outputs.")
    parser.add_argument("--rescore", action="store_true", help="score data/eval_results.json")
    parser.add_argument("--model", default=JUDGE_MODEL, help="model for calibration")
    parser.add_argument("--pairs", default=CALIBRATION_PATH, help="calibration file")
    args = parser.parse_args()
    if args.rescore:
        run_rescore()
    else:
        run_calibration(model=args.model, path=args.pairs)
