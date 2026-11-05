"""
Unit tests for the semantic judge, with fake model replies and no API calls.

Run directly:  python -m tests.test_semantic_judge
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env.test", override=True)

from tests.eval_golden_dataset import JACCARD_THRESHOLD, _jaccard, _strip_speaker, load_golden_dataset
from tests.semantic_judge import (
    RESULTS_PATH,
    JudgeError,
    agreement_report,
    judge_pair,
    score_example,
    unmatched_items,
)


def _client(text, stop_reason="end_turn", calls=None):
    def create(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        content = [SimpleNamespace(type="text", text=text)] if text is not None else []
        return SimpleNamespace(stop_reason=stop_reason, content=content)
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def _raises_judge_error(client):
    try:
        judge_pair("A", "B", client=client)
    except JudgeError:
        return True
    return False


def test_judge_reads_yes_and_no():
    assert judge_pair("A", "B", client=_client("YES")) is True
    assert judge_pair("A", "B", client=_client("no.")) is False


def test_judge_sends_both_items_to_the_model():
    calls = []
    judge_pair("Expected item", "Agent item", client=_client("YES", calls=calls))
    prompt = calls[0]["messages"][0]["content"]
    assert "Expected item" in prompt and "Agent item" in prompt


def test_judge_sends_the_section_and_the_rules_to_the_model():
    calls = []
    judge_pair("A", "B", section="done", client=_client("YES", calls=calls))
    prompt = calls[0]["messages"][0]["content"]
    assert "Section: done" in prompt
    assert "debugging" in prompt.lower()


def test_judge_rejects_an_unclear_answer():
    assert _raises_judge_error(_client("Maybe, it depends"))
    assert _raises_judge_error(_client(None))


def test_judge_rejects_an_unfinished_reply():
    assert _raises_judge_error(_client("YES", stop_reason="max_tokens"))
    assert _raises_judge_error(_client("NO", stop_reason="refusal"))


def _pair(kind, label):
    return {"kind": kind, "same_meaning": label}


def test_agreement_counts_errors_separately():
    pairs = [_pair("clear", True), _pair("near", False), _pair("wrong", False), _pair("hard", False)]
    report = agreement_report(pairs, [True, True, False, True])
    assert report["agreed"] == 2 and report["total"] == 4
    assert report["false_accepts"] == 2 and report["false_rejects"] == 0
    assert report["wrong_pairs_accepted"] == 1


def test_agreement_fails_below_ninety_percent():
    pairs = [_pair("clear", True)] * 10
    assert agreement_report(pairs, [True] * 9 + [False])["passed"] is True
    assert agreement_report(pairs, [True] * 8 + [False] * 2)["passed"] is False


def test_agreement_fails_when_a_wrong_pair_is_accepted():
    pairs = [_pair("clear", True)] * 9 + [_pair("wrong", False)]
    assert agreement_report(pairs, [True] * 10)["passed"] is False


def test_rescore_with_word_matching_repeats_the_stored_scores():
    # Same rules as today: swapping only the matcher back must change nothing.
    baseline = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    examples = {e["id"]: e for e in load_golden_dataset()}
    word_match = lambda exp, act: _jaccard(_strip_speaker(exp), _strip_speaker(act)) >= JACCARD_THRESHOLD
    for row in baseline["results"]:
        scores = score_example(row["actual"], examples[row["id"]]["expected_output"], word_match)
        for category, value in row["scores"].items():
            assert abs(scores[category] - value) < 1e-9, f"{row['id']} {category}: {scores[category]} != {value}"


def test_rescore_counts_agent_items_that_match_nothing():
    actual = {"done": ["Alice: API", "Bob: Extra"], "in_progress": [], "blockers": [], "actions": []}
    expected = {"done": ["API"], "in_progress": [], "blockers": [], "actions": []}
    same = lambda exp, act: exp in act
    assert unmatched_items(actual, expected, same) == 1


def test_judge_uses_the_requested_model():
    calls = []
    judge_pair("A", "B", client=_client("YES", calls=calls), model="claude-opus-5-5")
    assert calls[0]["model"] == "claude-opus-5-5"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"[OK]   {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {test.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
