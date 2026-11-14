"""The library evaluation script's metrics and its question file. Offline: the
script itself runs against the real index on demand, never in pytest."""
import json
import re
import sys
from contextlib import closing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import seed_demo  # noqa: E402
from eval_library import QUESTIONS, DegradedRerankError, copy_index, demo_label_records, metrics, search  # noqa: E402


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
    manifest = {entry["id"]: entry["topic"] for entry in
                json.loads((ROOT / "knowledge" / "manifest.json").read_text())}
    assert 90 <= len(questions) <= 110
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    for q in questions:
        assert set(q) <= {"id", "question", "expected", "split", "rxcuis"}, q["id"]
        assert {"id", "question", "expected", "split"} <= set(q), q["id"]
        assert q["split"] in ("tune", "test"), q["id"]
        assert isinstance(q["question"], str) and q["question"].strip(), q["id"]
        assert isinstance(q["expected"], list), q["id"]
        rxcuis = q.get("rxcuis", [])
        assert isinstance(rxcuis, list) and all(re.fullmatch(r"\d+", r) for r in rxcuis), q["id"]
        for source in q["expected"]:
            label = re.fullmatch(r"dailymed-rxcui-(\d+)", source)
            assert source in manifest or label, (q["id"], source)
            # A label can only come back for a patient who takes that drug.
            assert not label or label.group(1) in rxcuis, (q["id"], source)
    no_answer = [q for q in questions if not q["expected"]]
    assert 15 <= len(no_answer) <= 25
    for split in ("tune", "test"):
        assert any(q["split"] == split and not q["expected"] for q in questions), split
        # Near misses: a no-answer question for a patient whose drug set leaves out the drug asked about.
        assert any(q["split"] == split and not q["expected"] and "rxcuis" in q for q in questions), split
        assert any(q["split"] == split and any(e.startswith("dailymed-rxcui-") for e in q["expected"])
                   for q in questions), split
    # Every topic with a question appears in both splits.
    topics = {}
    for q in questions:
        for source in q["expected"]:
            topic = manifest.get(source, "Medications")
            topics.setdefault(topic, set()).add(q["split"])
    assert all(splits == {"tune", "test"} for splits in topics.values()), topics
    halves = [sum(q["split"] == split for q in questions) for split in ("tune", "test")]
    assert abs(halves[0] - halves[1]) <= 2
    assert "\u2014" not in QUESTIONS.read_text(encoding="utf-8")


async def test_demo_label_records_come_offline_in_the_approved_label_shape():
    records = await demo_label_records()
    assert [r["id"] for r in records] == [f"dailymed-rxcui-{x}" for x in ("2683", "7258", "77492", "5640")]
    assert {r["rxcui"] for r in records} == seed_demo.DEMO_RXCUIS  # the same drugs the seed gives the demo patients
    for r in records:
        assert r["rxcui"] == r["id"].removeprefix("dailymed-rxcui-")
        assert r["topic"] == "Medications"
        assert r["url"].startswith("https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=")
        assert r["title"].startswith("DailyMed label: ") and r["text"].strip() and len(r["sha256"]) == 64
    assert "colchicine" in records[0]["title"].lower()


def test_copy_index_copies_and_leaves_the_source_unchanged(tmp_path):
    import sqlite3
    source = tmp_path / "index.sqlite3"
    with closing(sqlite3.connect(source)) as db, db:
        db.execute("CREATE TABLE t (x)")
        db.execute("INSERT INTO t VALUES (1)")
    before = source.read_bytes()
    copy = tmp_path / "copy.sqlite3"
    copy_index(source, copy)
    with closing(sqlite3.connect(copy)) as db, db:
        assert db.execute("SELECT x FROM t").fetchall() == [(1,)]
        db.execute("INSERT INTO t VALUES (2)")
    assert source.read_bytes() == before


class FallingBackStore:
    """Counts a fallback on every rerank search, like a store whose reranker fails."""

    def __init__(self):
        self.reranker_failures = 0
        self.calls = []

    async def retrieve(self, query, rxcuis=(), mode=None):
        self.calls.append((query, list(rxcuis), mode))
        if mode == "rerank":
            self.reranker_failures += 1
        return []


async def test_search_passes_each_question_its_own_drug_set_and_detects_fallback():
    store = FallingBackStore()
    questions = [{"question": "a", "rxcuis": ["2683"]}, {"question": "b"}]
    assert await search(store, questions, "embedding") == [[], []]
    assert store.calls == [("a", ["2683"], "embedding"), ("b", [], "embedding")]
    with pytest.raises(DegradedRerankError, match="2 of 2"):
        await search(store, questions, "rerank")


def test_eval_index_path_resolves_without_creating_anything(tmp_path):
    from eval_library import existing_index_path
    missing = tmp_path / "sub" / "index.sqlite3"
    with pytest.raises(SystemExit) as exc:
        existing_index_path(str(missing))
    assert "does not exist" in str(exc.value) and str(missing) in str(exc.value)
    assert not missing.parent.exists()
    present = tmp_path / "index.sqlite3"
    present.write_bytes(b"")
    assert existing_index_path(str(present)) == present
