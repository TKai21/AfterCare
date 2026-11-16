import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import select

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_clone  # noqa: E402
import seed_demo  # noqa: E402
from db import make_engine, make_session_factory  # noqa: E402
from fake_llm import FakeLLM  # noqa: E402
from guidance import eligible, patient_text  # noqa: E402
from models import DischargeDocument, DoctorGuidance, ExtractedItem, User  # noqa: E402


def run(split, expect, kind, score=None, wrong=None):
    return {"split": split, "expect": expect, "kind": kind, "score": score, "wrong": wrong}


def test_loss_inputs_and_metrics():
    runs = [
        run("calibration", "answer", "answer", 0.9, False),
        run("calibration", "answer", "answer", 0.4, True),
        run("calibration", "escalate", "answer", 0.7, True),
        run("calibration", "escalate", "not_covered"),
        run("calibration", "answer", "not_covered"),
    ]
    scores, wrong = calibrate_clone.loss_inputs(runs)
    assert scores == [0.9, 0.4, 0.7, None, None]
    assert wrong == [False, True, True, False, False]
    m = calibrate_clone.metrics(runs, 0.5)
    assert m == {"n": 5, "risk": 0.2, "answer_rate": 0.4, "coverage": round(1 / 3, 4),
                 "wrong_answers": 1, "escalated": 3}
    assert calibrate_clone.metrics(runs, 0.0)["risk"] == 0.4


def test_calibrate_meets_the_bound_and_reports_test():
    cal = [run("calibration", "answer", "answer", 0.9, False)] * 15 + \
          [run("calibration", "escalate", "answer", 0.3, True)] * 4
    test = [run("test", "answer", "answer", 0.85, False), run("test", "escalate", "answer", 0.2, True)]
    result = calibrate_clone.calibrate(cal + test, 0.1)
    assert result["threshold"] == 0.9
    assert result["calibration"]["bound"] <= 0.1
    assert result["test"]["risk"] == 0.0 and result["test"]["answer_rate"] == 0.0
    assert result["test_without_gate"]["risk"] == 0.5
    try:
        calibrate_clone.calibrate(cal[:5], 0.1)
        raise AssertionError("expected a refusal")
    except ValueError as exc:
        assert "cannot certify" in str(exc)


def test_curve_and_svg_are_well_formed():
    test = [run("test", "answer", "answer", s, s < 0.5) for s in (0.2, 0.4, 0.6, 0.8)]
    points = calibrate_clone.curve(test)
    assert points[0]["threshold"] == 0.0 and points[-1]["answer_rate"] == 0.0
    svg = calibrate_clone.svg_curve(test, 0.6, 0.1)
    assert svg.startswith("<svg") and "t* = 0.60" in svg and "nan" not in svg


def test_question_set_is_well_formed():
    data = json.loads(calibrate_clone.QUESTIONS.read_text())["questions"]
    ids = [q["id"] for q in data]
    assert len(ids) == len(set(ids)) >= 90
    for q in data:
        assert q["patient"] in calibrate_clone.PASSWORDS
        assert q["doctor"] in ("dr.ting", "dr.renny", "dr.patel")
        assert q["expect"] in ("answer", "escalate") and q["split"] in ("calibration", "test")
        assert bool(q["facts"]) == (q["expect"] == "answer")
    for expect in ("answer", "escalate"):
        splits = [q["split"] for q in data if q["expect"] == expect]
        # Alternating splits, then one question relabeled after review (clone_review.json).
        assert abs(splits.count("calibration") - splits.count("test")) <= 2


async def test_every_history_entry_reaches_its_eval_patient(tmp_path):
    """Each fictional earlier advice entry passes the gate for the patient whose
    questions rely on it, so a miss in the evaluation is the clone's, not the data's."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'seed.db'}"
    await seed_demo.run_seed(url, FakeLLM(), True)
    engine = make_engine(url)
    patient_of = {"dr.renny": "demo", "dr.ting": "demo2", "dr.patel": "demo2"}
    try:
        async with make_session_factory(engine)() as s:
            users = {u.username: u for u in await s.scalars(select(User))}
            texts = {}
            for name in ("demo", "demo2"):
                doc = await s.scalar(select(DischargeDocument).where(
                    DischargeDocument.active_user_id == users[name].id))
                items = list(await s.scalars(select(ExtractedItem).where(ExtractedItem.document_id == doc.id)))
                texts[name] = patient_text(doc, items)
    finally:
        await engine.dispose()
    for e in json.loads(calibrate_clone.GUIDANCE.read_text())["entries"]:
        entry = DoctorGuidance(scope="general", drug_tags=e["drug_tags"], condition_tags=e["condition_tags"])
        assert eligible(entry, texts[patient_of[e["doctor"]]]), e["question"]


def test_apply_review_overrides_and_rejects_stale_entries():
    runs = [dict(run("test", "answer", "answer", 0.9, True), id="a", answer="text A"),
            dict(run("test", "escalate", "not_covered"), id="b", answer="")]
    out = calibrate_clone.apply_review(runs, [{"id": "a", "wrong": False, "note": "judge error", "answer": "text A"}])
    assert out[0]["wrong"] is False and out[0]["judge_wrong"] is True and out[0]["review"] == "judge error"
    assert runs[0]["wrong"] is True  # the input is not changed
    for stale in ({"id": "b", "wrong": True, "note": "", "answer": ""},
                  {"id": "zz", "wrong": True, "note": "", "answer": "x"}):
        with pytest.raises(ValueError):
            calibrate_clone.apply_review(runs, [stale])


def test_override_for_a_different_answer_is_stale():
    runs = [dict(run("test", "answer", "answer", 0.9, True), id="a", answer="new text")]
    with pytest.raises(ValueError, match=r"a.*stale.*different answer"):
        calibrate_clone.apply_review(runs, [{"id": "a", "wrong": False, "note": "n", "answer": "old text"}])


def test_override_without_an_answer_is_an_error():
    runs = [dict(run("test", "answer", "answer", 0.9, True), id="a", answer="text")]
    with pytest.raises(ValueError, match=r"a.*answer"):
        calibrate_clone.apply_review(runs, [{"id": "a", "wrong": False, "note": "n"}])


def test_no_review_flag_keeps_the_judge_verdicts(tmp_path):
    out = tmp_path / "out"
    assert calibrate_clone.main(["--from-runs", "--runs", str(ROOT / "eval" / "clone_runs.json"),
                                 "--no-review", "--out-dir", str(out)]) == 0
    assert json.loads((out / "clone_calibration.json").read_text())["review_overrides"] == 0


def test_review_files_match_their_saved_runs():
    pairs = [(calibrate_clone.RUNS, calibrate_clone.REVIEW),
             (calibrate_clone.ROOT / "eval" / "clone_runs_v1.json", calibrate_clone.ROOT / "eval" / "clone_review_v1.json")]
    for runs_path, review_path in pairs:
        if runs_path.is_file() and review_path.is_file():
            runs = json.loads(runs_path.read_text())["runs"]
            calibrate_clone.apply_review(runs, json.loads(review_path.read_text())["overrides"])


def test_out_dir_writes_there_only(tmp_path):
    eval_dir = ROOT / "eval"
    names = ("clone_runs.json", "clone_calibration.json", "clone_risk_coverage.svg")
    before = {n: (eval_dir / n).read_bytes() for n in names}
    out = tmp_path / "nested" / "out"
    code = calibrate_clone.main(["--from-runs", "--runs", str(eval_dir / "clone_runs.json"),
                                 "--out-dir", str(out)])
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == ["clone_calibration.json",
                                                    "clone_risk_coverage.svg"]
    assert (out / "clone_risk_coverage.svg").read_text().startswith("<svg")
    assert json.loads((out / "clone_calibration.json").read_text())["threshold"] is not None
    assert {n: (eval_dir / n).read_bytes() for n in names} == before


def test_refuses_an_empty_library_index(tmp_path, capsys, monkeypatch):
    from dataclasses import replace
    index = tmp_path / "index.sqlite3"
    real = calibrate_clone.get_settings
    monkeypatch.setattr(calibrate_clone, "get_settings", lambda: replace(real(), rag_db_path=str(index)))
    out = tmp_path / "out"
    assert calibrate_clone.main(["--out-dir", str(out)]) == 1
    err = capsys.readouterr().err
    assert str(index) in err and "0 documents" in err
    assert not out.exists()


def test_from_runs_without_a_runs_file_is_a_clear_error(tmp_path, capsys):
    out = tmp_path / "empty"
    assert calibrate_clone.main(["--from-runs", "--out-dir", str(out)]) == 2
    err = capsys.readouterr().err
    assert str(out / "clone_runs.json") in err and "--runs" in err
    assert not out.exists()


def test_the_final_message_lists_the_files_written(tmp_path, capsys):
    out = tmp_path / "out"
    assert calibrate_clone.main(["--from-runs", "--runs", str(ROOT / "eval" / "clone_runs.json"),
                                 "--out-dir", str(out)]) == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last == f"Wrote clone_calibration.json, clone_risk_coverage.svg in {out}"


def test_from_runs_help_names_the_runs_option():
    help_text = calibrate_clone.build_parser().format_help()
    assert "--runs" in help_text and "OUT_DIR/clone_runs.json" in " ".join(help_text.split())
    assert "--from-runs" in calibrate_clone.__doc__ and "--runs" in calibrate_clone.__doc__


def test_library_index_check_builds_no_model_client(tmp_path, monkeypatch):
    import llm
    from dataclasses import replace

    def refuse(*args, **kwargs):
        raise AssertionError("no model client is needed to count the index's documents")

    monkeypatch.setattr(llm, "LLMClient", refuse)
    real = calibrate_clone.get_settings
    monkeypatch.setattr(calibrate_clone, "get_settings",
                        lambda: replace(real(), rag_db_path=str(tmp_path / "index.sqlite3")))
    assert "0 documents" in calibrate_clone.library_index_problem()
