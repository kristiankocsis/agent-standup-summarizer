"""
Unit tests for the eval scoring helpers (no API calls, run in <1s).

Run directly:  python -m tests.test_eval_scoring
Or via pytest: pytest tests/test_eval_scoring.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# Dummy credentials so the import chain (eval -> agent -> config) works on a
# fresh clone with no real .env. Scoring tests never call the API.
load_dotenv(Path(__file__).parent.parent / ".env.test", override=True)

from tests.eval_golden_dataset import (
    JACCARD_THRESHOLD,
    _jaccard,
    _stem,
    _strip_speaker,
    score_actions,
    score_blockers,
    score_text_list,
)


# ── Stemmer ────────────────────────────────────────────────────────────────

def test_stem_maps_inflected_forms_to_same_stem():
    pairs = [
        ("tests", "testing"),        # plural vs gerund
        ("polished", "polish"),      # past tense vs noun
        ("migrated", "migration"),   # past tense vs -ion noun
        ("queries", "query"),        # -ies plural
        ("sessions", "session"),     # chained suffixes must stay consistent
        ("deployed", "deploy"),
        ("passes", "passing"),
        ("cache", "caching"),        # trailing -e strip
        ("logs", "logging"),         # doubled-consonant collapse
        ("stopped", "stops"),
        ("live", "lives"),
    ]
    for a, b in pairs:
        assert _stem(a) == _stem(b), f"_stem({a!r})={_stem(a)!r} != _stem({b!r})={_stem(b)!r}"


def test_stem_never_strips_below_three_chars():
    assert _stem("is") == "is"
    assert _stem("was") == "was"
    for word in ("goes", "does", "used", "axes", "ties"):
        assert len(_stem(word)) >= 3, f"_stem({word!r}) = {_stem(word)!r}"


# ── Speaker prefix stripping ───────────────────────────────────────────────

def test_strip_speaker_removes_name_prefix():
    assert _strip_speaker("Alice: finished the auth API") == "finished the auth API"
    assert _strip_speaker("Greg (async): completed refactor") == "completed refactor"


def test_strip_speaker_keeps_plain_text():
    # No "Name:" shape at the start -> unchanged
    assert _strip_speaker("finished the auth API") == "finished the auth API"
    assert _strip_speaker("Architecture decision needed: JWT vs sessions") \
        == "Architecture decision needed: JWT vs sessions"


# ── Jaccard ────────────────────────────────────────────────────────────────

def test_jaccard_identical_and_disjoint():
    assert _jaccard("stress testing", "stress testing") == 1.0
    assert _jaccard("alpha bravo", "charlie delta") == 0.0
    assert _jaccard("", "") == 1.0
    assert _jaccard("alpha", "") == 0.0


def test_jaccard_matches_across_inflection():
    # This exact pair scored 0.20 (a miss) before stemming was added
    score = _jaccard(
        "running final stress tests on the migration",
        "stress testing on migrated system",
    )
    assert score >= JACCARD_THRESHOLD, f"expected match, got {score:.2f}"


# ── score_text_list ────────────────────────────────────────────────────────

def test_text_list_empty_cases():
    assert score_text_list([], []) == 1.0          # both empty -> perfect
    assert score_text_list(["extra"], []) == 0.7   # hallucination penalty
    assert score_text_list([], ["missed"]) == 0.0  # agent found nothing


def test_text_list_speaker_prefix_does_not_penalize():
    # Agent items carry "Name:" prefixes, golden labels don't
    actual = ["Ivan: polished the UI"]
    expected = ["UI polish for demo"]
    assert score_text_list(actual, expected) == 1.0


def test_text_list_threshold_boundary_counts_as_hit():
    # 1 shared token of 4 total = exactly 0.25 -> >= threshold -> hit
    assert JACCARD_THRESHOLD == 0.25
    assert score_text_list(["alpha bravo charlie delta"], ["alpha"]) == 1.0


def test_text_list_partial_recall():
    actual = ["Bob: wrapped up the database migration"]
    expected = ["Database migration", "Frontend forms development"]
    assert score_text_list(actual, expected) == 0.5


# ── score_blockers ─────────────────────────────────────────────────────────

def test_blockers_match_by_description():
    actual = [{"description": "Staging environment is down", "owner": "Alice", "severity": "high"}]
    expected = [{"description": "Staging environment down", "owner": "Alice", "severity": "high"}]
    assert score_blockers(actual, expected) == 1.0


def test_blockers_hallucination_penalty_scales():
    fake = {"description": "Invented blocker", "owner": "X", "severity": "low"}
    assert score_blockers([fake], []) == 0.8
    assert score_blockers([fake, fake], []) == 0.6
    assert score_blockers([], []) == 1.0


# ── score_actions ──────────────────────────────────────────────────────────

def test_actions_match_by_task():
    actual = [{"task": "Restore the staging environment", "owner": "DevOps", "deadline": "ASAP"}]
    expected = [{"task": "Restore staging environment", "owner": "DevOps", "deadline": "ASAP"}]
    assert score_actions(actual, expected) == 1.0


def test_actions_empty_cases():
    assert score_actions([], []) == 1.0
    assert score_actions([{"task": "extra"}], []) == 0.7
    assert score_actions([], [{"task": "missed"}]) == 0.0


# ── Plain runner (ASCII output for Windows cp1250 console) ─────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"[OK]   {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {test.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
