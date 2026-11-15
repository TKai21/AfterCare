import json
import sys
from pathlib import Path

from sqlalchemy import select

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import eval_clone  # noqa: E402
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
    scores, wrong = eval_clone.loss_inputs(runs)
    assert scores == [0.9, 0.4, 0.7, None, None]
    assert wrong == [False, True, True, False, False]
    m = eval_clone.metrics(runs, 0.5)
    assert m == {"n": 5, "risk": 0.2, "answer_rate": 0.4, "coverage": round(1 / 3, 4),
                 "wrong_answers": 1, "escalated": 3}
    assert eval_clone.metrics(runs, 0.0)["risk"] == 0.4


def test_calibrate_meets_the_bound_and_reports_test():
    cal = [run("calibration", "answer", "answer", 0.9, False)] * 15 + \
          [run("calibration", "escalate", "answer", 0.3, True)] * 4
    test = [run("test", "answer", "answer", 0.85, False), run("test", "escalate", "answer", 0.2, True)]
    result = eval_clone.calibrate(cal + test, 0.1)
    assert result["threshold"] == 0.9
    assert result["calibration"]["bound"] <= 0.1
    assert result["test"]["risk"] == 0.0 and result["test"]["answer_rate"] == 0.0
    assert result["test_without_gate"]["risk"] == 0.5
    try:
        eval_clone.calibrate(cal[:5], 0.1)
        raise AssertionError("expected a refusal")
    except ValueError as exc:
        assert "cannot certify" in str(exc)


def test_curve_and_svg_are_well_formed():
    test = [run("test", "answer", "answer", s, s < 0.5) for s in (0.2, 0.4, 0.6, 0.8)]
    points = eval_clone.curve(test)
    assert points[0]["threshold"] == 0.0 and points[-1]["answer_rate"] == 0.0
    svg = eval_clone.svg_curve(test, 0.6, 0.1)
    assert svg.startswith("<svg") and "t* = 0.60" in svg and "nan" not in svg


def test_question_set_is_well_formed():
    data = json.loads(eval_clone.QUESTIONS.read_text())["questions"]
    ids = [q["id"] for q in data]
    assert len(ids) == len(set(ids)) >= 90
    for q in data:
        assert q["patient"] in eval_clone.PASSWORDS
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
    for e in json.loads(eval_clone.GUIDANCE.read_text())["entries"]:
        entry = DoctorGuidance(scope="general", drug_tags=e["drug_tags"], condition_tags=e["condition_tags"])
        assert eligible(entry, texts[patient_of[e["doctor"]]]), e["question"]


def test_apply_review_overrides_and_rejects_stale_entries():
    runs = [dict(run("test", "answer", "answer", 0.9, True), id="a"),
            dict(run("test", "escalate", "not_covered"), id="b")]
    out = eval_clone.apply_review(runs, [{"id": "a", "wrong": False, "note": "judge error"}])
    assert out[0]["wrong"] is False and out[0]["judge_wrong"] is True and out[0]["review"] == "judge error"
    assert runs[0]["wrong"] is True  # the input is not changed
    for stale in ({"id": "b", "wrong": True, "note": ""}, {"id": "zz", "wrong": True, "note": ""}):
        try:
            eval_clone.apply_review(runs, [stale])
            raise AssertionError("expected a stale review error")
        except ValueError:
            pass


def test_review_file_matches_the_saved_runs():
    if not eval_clone.RUNS.is_file():
        return
    runs = json.loads(eval_clone.RUNS.read_text())["runs"]
    eval_clone.apply_review(runs, json.loads(eval_clone.REVIEW.read_text())["overrides"])
