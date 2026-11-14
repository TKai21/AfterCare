"""The library evaluation script's metrics and its question file. Offline: the
script itself runs against the real index on demand, never in pytest."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_library import QUESTIONS, metrics  # noqa: E402


def test_metrics_hits_ranks_and_counts():
    result = metrics([["a", "b"], ["c"], []], [["b"], ["x"], []])
    assert result["hit@1"] == 0
    assert result["hit@5"] == 0.5
    assert result["mrr"] == 0.25
    assert result["no_answer_false_hits"] == 0
    assert result["answerable"] == 2
    assert result["no_answer"] == 1


def test_metrics_counts_any_result_on_a_no_answer_question_as_a_false_hit():
    result = metrics([["a"], ["b", "c"], []], [["a"], [], []])
    assert result["hit@1"] == 1
    assert result["hit@5"] == 1
    assert result["mrr"] == 1
    assert result["no_answer_false_hits"] == 1
    assert result["answerable"] == 1
    assert result["no_answer"] == 2


def test_metrics_hit_at_5_ignores_ranks_past_five():
    result = metrics([["a", "b", "c", "d", "e", "x"]], [["x"]])
    assert result["hit@5"] == 0
    assert result["mrr"] == pytest.approx(1 / 6)


def test_metrics_any_expected_id_counts_and_the_best_rank_wins():
    result = metrics([["a", "b", "c"]], [["c", "b"]])
    assert result["hit@1"] == 0
    assert result["hit@5"] == 1
    assert result["mrr"] == 0.5


def test_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        metrics([["a"]], [["a"], ["b"]])


def test_question_file_is_valid():
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    manifest = {entry["id"] for entry in json.loads((ROOT / "knowledge" / "manifest.json").read_text())}
    assert 30 <= len(questions) <= 40
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    for q in questions:
        assert set(q) == {"id", "question", "expected", "split"}, q["id"]
        assert q["split"] in ("tune", "test"), q["id"]
        assert isinstance(q["question"], str) and q["question"].strip(), q["id"]
        assert isinstance(q["expected"], list), q["id"]
        for source in q["expected"]:
            assert source in manifest or re.fullmatch(r"dailymed-rxcui-\d+", source), (q["id"], source)
    for split in ("tune", "test"):
        assert any(q["split"] == split and not q["expected"] for q in questions), split
    assert sum(not q["expected"] for q in questions) >= 6
    assert "\u2014" not in QUESTIONS.read_text(encoding="utf-8")
