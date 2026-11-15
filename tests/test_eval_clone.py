"""The clone evaluation runner, report, human sheet and CLI. Offline: FakeLLM
for the clone and the judge, small inline patients and memory, temporary files."""
import csv
import gc
import json
import re
import sys
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
import eval_clone  # noqa: E402
from agent import fallback_answer  # noqa: E402
from clone_eval_score import checks_for, cohen_kappa  # noqa: E402
from config import Settings  # noqa: E402
from druglabels import patient_rxcuis  # noqa: E402
from fake_llm import FakeLLM  # noqa: E402
from guidance import retrieve  # noqa: E402
from knowledge import KnowledgeStore  # noqa: E402
from models import (  # noqa: E402
    ChatMessage, DischargeDocument, DoctorGuidance, DocumentDrug, DrugLabel, ExtractedItem, User,
)

USERNAMES = ("dr.ting", "dr.patel", "dr.renny")
NAME_WORDS = ("Ting", "Kai", "Parin", "Patel", "Aaron", "Renny")


def _patient(pid, drugs, extra=""):
    lines = {"colchicine": "Colchicine 0.6 mg by mouth twice daily for 10 days.",
             "naproxen": "Naproxen 500 mg by mouth every 12 hours as needed for pain."}
    text = ("FICTIONAL - FOR EVALUATION\n\nDiagnosis:\nAcute gout flare of the right big toe.\n\n"
            "Medications:\n" + "\n".join(lines[d] for d in drugs) + "\n\nFollow-up:\nCall the clinic "
            "if your pain is not better." + extra)
    items = [{"category": "medication", "text": "Take " + lines[d].lower(), "source_quote": lines[d]}
             for d in drugs]
    items.append({"category": "diet", "text": "Drink water every day.", "source_quote": "Call the clinic"})
    return {"id": pid, "text": text, "prescribed": list(drugs), "ckd": False, "past_effective": None,
            "flare_plan": False, "items": items}


PATIENTS = [_patient("p_both", ["colchicine", "naproxen"]), _patient("p_nap", ["naproxen"])]


def _entry(dimension, question, answer):
    return {"dimension": dimension, "question": question, "answer": answer, "scope": "general",
            "condition_tags": ["gout"], "drug_tags": [], "synthetic": True}


MEMORY = {
    "dr.ting": [_entry("D1", "Which medicine first for a gout flare?",
                       "I usually start colchicine first when the case allows."),
                _entry("D2", "Do I need a flare plan for gout?",
                       "I prefer a written flare plan set up in advance.")],
    "dr.patel": [_entry("D1", "Which medicine first for a gout flare?",
                        "I usually start naproxen first when the case allows.")],
    "dr.renny": [_entry("D1", "Which medicine first for a gout flare?",
                        "I use what worked for you before.")],
    "generated": {"model": "fake", "date": "2026-11-14"},
}

CASE_D1 = {
    "id": "t01", "patient": "p_both", "dimension": "D1",
    "question": "My toe throbs again tonight. Which medicine should I reach for first?",
    "expected": {
        "dr.ting": {"first_choice": "colchicine", "must_not_recommend": ["prednisone"]},
        "dr.patel": {"first_choice": "naproxen", "must_not_recommend": ["prednisone"]},
        "dr.renny": {"refer_to_clinic": True, "first_ask": "current_plan",
                     "must_not_recommend": ["prednisone"]},
    },
}
CASE_NONE = {
    "id": "t02", "patient": "p_nap", "dimension": "none",
    "question": "Can I still have a cup of coffee in the morning?",
    "expected": {u: {"must_not_recommend": ["colchicine", "prednisone"], "no_memory_use": True}
                 for u in USERNAMES},
}


def reply(text="Follow your discharge document [1].", kind="answer"):
    return {"content": json.dumps({"kind": kind, "answer": text, "citations": [1],
                                   "care_team_question": None})}


class FakeJudge:
    """Answers every check id in the judge prompt with `met`."""

    def __init__(self, met=True):
        self.met = met
        self.calls = []

    async def chat(self, messages, tools=None, json_mode=False):
        self.calls.append((messages, json_mode))
        ids = re.findall(r"^Check (c\d+):", messages[-1]["content"], re.M)
        return {"content": json.dumps({"checks": {i: {"met": self.met, "reason": f"reason {i}"} for i in ids}})}


def make_settings(tmp_path, **overrides):
    return Settings(database_url=f"sqlite+aiosqlite:///{tmp_path / 'world.sqlite3'}",
                    jwt_secret="test-secret-not-for-production", llm_model="fake-clone", **overrides)


def world(tmp_path, llm):
    store = KnowledgeStore(tmp_path / "index.sqlite3", llm)
    return eval_clone.open_world(make_settings(tmp_path), llm, store, PATIENTS, MEMORY)


async def test_build_world_memory_reaches_every_patient(tmp_path):
    llm = FakeLLM()
    async with world(tmp_path, llm) as (_client, session, w):
        assert set(w.doctors) == set(USERNAMES)
        for username, name, specialty in eval_clone.DOCTORS:
            doctor = await session.get(User, w.doctors[username])
            assert (doctor.role, doctor.display_name, doctor.specialty) == ("doctor", name, specialty)
            assert len(w.guidance[username]) == len(MEMORY[username])
            assert [w.dimensions[i] for i in w.guidance[username]] == [e["dimension"] for e in MEMORY[username]]
        rows = (await session.scalars(select(DoctorGuidance))).all()
        assert all(r.active and r.embedding and r.scope == "general" and r.condition_tags == ["gout"]
                   and r.drug_tags == [] for r in rows)
        labels = {r.rxcui: r.status for r in await session.scalars(select(DrugLabel))}
        assert labels == {"2683": "approved", "7258": "approved"}
        for patient in PATIENTS:
            user = await session.get(User, w.patients[patient["id"]])
            assert user.username == f"eval-{patient['id']}"
            doc = await session.get(DischargeDocument, w.documents[patient["id"]])
            assert (doc.status, doc.active_user_id, doc.user_id) == ("confirmed", user.id, user.id)
            items = list(await session.scalars(
                select(ExtractedItem).where(ExtractedItem.document_id == doc.id)))
            assert len(items) == len(patient["items"]) and all(i.status == "active" for i in items)
            drugs = list(await session.scalars(select(DocumentDrug).where(DocumentDrug.document_id == doc.id)))
            assert sorted(d.name for d in drugs) == sorted(patient["prescribed"])
            assert all(d.status == "resolved" for d in drugs)
            want_rxcuis = {"colchicine": "2683", "naproxen": "7258"}
            assert await patient_rxcuis(session, doc.id) == {want_rxcuis[d] for d in patient["prescribed"]}
            for username in USERNAMES:
                hits = await retrieve(session, llm, user, doc, items, "gout flare medicine plan", -1.0,
                                      doctor_id=w.doctors[username], limit=100)
                assert {e.id for e, _ in hits} == set(w.guidance[username])
        await session.commit()


async def test_ask_turn_clears_history_and_records_kind(tmp_path):
    llm = FakeLLM()
    async with world(tmp_path, llm) as (client, session, w):
        llm.script(reply(), reply("Your document lists your medicines [1]."))
        first = await eval_clone.ask_turn(client, session, w, CASE_D1, "dr.ting")
        second = await eval_clone.ask_turn(client, session, w, CASE_D1, "dr.ting")
        assert first == {"case": "t01", "doctor": "dr.ting", "kind": "answer",
                         "answer": "Follow your discharge document [1].", "guidance_ids": [],
                         "fallback": False, "eligible_guidance_ids": w.guidance["dr.ting"]}
        assert second["answer"] == "Your document lists your medicines [1]."
        # The second turn's model call sees no history: only the system prompt and this turn.
        assert [m["role"] for m in llm.calls[-1][0]] == ["system", "user"]
        assert llm.calls[-1][0][-1]["content"].endswith(CASE_D1["question"])
        count = await session.scalar(select(func.count()).select_from(ChatMessage))
        assert count == 2  # only the last turn's question and answer remain
        # A reply that never validates ends in the doctor's fallback answer.
        llm.script({"content": "not json"}, {"content": "not json"}, {"content": "not json"})
        failed = await eval_clone.ask_turn(client, session, w, CASE_D1, "dr.patel")
        assert failed["fallback"] is True and failed["kind"] == "not_covered"
        assert failed["answer"] == fallback_answer("Dr. Parin Patel")


async def test_without_memory_the_doctor_entries_are_inactive(tmp_path):
    llm = FakeLLM()
    async with world(tmp_path, llm) as (client, session, w):
        await eval_clone.set_memory(session, w, "dr.ting", False)
        rows = {r.id: r for r in await session.scalars(
            select(DoctorGuidance).execution_options(populate_existing=True))}
        assert all(not rows[i].active for i in w.guidance["dr.ting"])
        assert all(rows[i].active for u in ("dr.patel", "dr.renny") for i in w.guidance[u])
        llm.script(reply())
        record = await eval_clone.ask_turn(client, session, w, CASE_D1, "dr.ting")
        assert record["eligible_guidance_ids"] == []
        assert "(none matched this question)" in llm.calls[-1][0][-1]["content"]
        await eval_clone.set_memory(session, w, "dr.ting", True)
        llm.script(reply())
        record = await eval_clone.ask_turn(client, session, w, CASE_D1, "dr.ting")
        assert record["eligible_guidance_ids"] == w.guidance["dr.ting"]
        assert "My earlier advice, " in llm.calls[-1][0][-1]["content"]


def sentinel_settings(tmp_path):
    """Settings whose DATABASE_URL and RAG_DB_PATH name sentinel files."""
    dev = tmp_path / "dev.sqlite3"
    dev.write_bytes(b"sentinel: never open")
    index = tmp_path / "index.sqlite3"
    KnowledgeStore(index, FakeLLM())  # a valid, empty library index
    return replace(make_settings(tmp_path), database_url=f"sqlite+aiosqlite:///{dev}",
                   rag_db_path=str(index)), dev, index


def snapshot(directory: Path) -> dict:
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(directory.iterdir()) if p.is_file()}


def done_record(case, doctor, condition, repeat):
    return {"case": case, "dimension": "D1", "doctor": doctor, "condition": condition, "repeat": repeat,
            "kind": "answer", "answer": "old", "guidance_ids": [], "eligible_guidance_ids": [],
            "fallback": False, "items": {"first_choice": True, "must_not_recommend": True},
            "conflicts_met": [], "blocked": False, "judge": {}}


async def test_resume_skips_done_turns_and_refuses_other_models(tmp_path, capsys):
    settings, _, _ = sentinel_settings(tmp_path)
    results = tmp_path / "out" / "results.jsonl"
    results.parent.mkdir()
    header = {"clone_model": "fake-clone", "judge_model": "fake-judge", "started": "2026-11-14T00:00:00"}
    results.write_text(json.dumps(header) + "\n" + json.dumps(done_record("t01", "dr.ting", "with_memory", 0)) + "\n")
    clone, judge = FakeLLM(), FakeJudge()
    clone.script(*[reply() for _ in range(11)])
    records = await eval_clone.run_eval(settings, clone, judge, repeats=1, results_path=results,
                                        cases=[CASE_D1, CASE_NONE], patients=PATIENTS, memory=MEMORY,
                                        judge_model="fake-judge")
    assert len(clone.calls) == 11 and clone.queue == []
    assert len(judge.calls) == 5 and all(json_mode for _, json_mode in judge.calls)  # the D1 turns only
    assert len(records) == 12
    assert records[0]["answer"] == "old"
    lines = [json.loads(line) for line in results.read_text().splitlines()]
    assert lines[0] == header and lines[1:] == records
    assert len({(r["case"], r["doctor"], r["condition"], r["repeat"]) for r in records}) == 12
    out = capsys.readouterr().out
    # Order: repeat, condition, case, doctor; the finished first turn is not printed.
    assert "[2/12] t01 dr.patel with_memory 0: answer" in out
    assert "[7/12] t01 dr.ting without_memory 0: answer" in out
    assert "[1/12]" not in out

    by_key = {(r["case"], r["doctor"], r["condition"]): r for r in records}
    patel = by_key[("t01", "dr.patel", "with_memory")]
    checks, owner = checks_for(CASE_D1, "dr.patel")
    assert set(patel["judge"]) == set(checks)
    assert {(v["kind"], v["expected"], v["owner"], v["met"]) for v in patel["judge"].values()} == {
        (checks[c][0], checks[c][1], owner[c], True) for c in checks}
    assert patel["items"] == {"first_choice": True, "must_not_recommend": True}
    assert patel["conflicts_met"] == ["first_choice"]
    assert patel["blocked"] is False and patel["patient"] == "p_both"
    assert patel["prescribed"] == ["colchicine", "naproxen"] and patel["question"] == CASE_D1["question"]
    none = by_key[("t02", "dr.renny", "with_memory")]
    assert none["judge"] == {} and none["conflicts_met"] == []
    assert none["items"] == {"must_not_recommend": True, "no_memory_use": True}

    # A finished run makes no request at all.
    again = await eval_clone.run_eval(settings, FakeLLM(), FakeJudge(), repeats=1, results_path=results,
                                      cases=[CASE_D1, CASE_NONE], patients=PATIENTS, memory=MEMORY,
                                      judge_model="fake-judge")
    assert again == records

    other = replace(settings, llm_model="another-clone")
    with pytest.raises(SystemExit, match="another-clone"):
        await eval_clone.run_eval(other, FakeLLM(), FakeJudge(), repeats=1, results_path=results,
                                  cases=[CASE_D1], patients=PATIENTS, memory=MEMORY, judge_model="fake-judge")
    with pytest.raises(SystemExit, match="other-judge"):
        await eval_clone.run_eval(settings, FakeLLM(), FakeJudge(), repeats=1, results_path=results,
                                  cases=[CASE_D1], patients=PATIENTS, memory=MEMORY, judge_model="other-judge")


async def test_blocked_answers_are_not_judged(tmp_path):
    settings, _, _ = sentinel_settings(tmp_path)
    results = tmp_path / "results.jsonl"
    clone, judge = FakeLLM(), FakeJudge()
    clone.script(*[{"content": "not json"} for _ in range(3 * 6)])
    records = await eval_clone.run_eval(settings, clone, judge, repeats=1, results_path=results,
                                        cases=[CASE_D1], patients=PATIENTS, memory=MEMORY, judge_model="j")
    assert judge.calls == []
    assert all(r["blocked"] and r["fallback"] and r["judge"] == {} for r in records)
    assert all(r["items"]["must_not_recommend"] is True for r in records)
    assert all(v is False for r in records for k, v in r["items"].items() if k != "must_not_recommend")


async def test_records_carry_doc_governs_from_the_case(tmp_path):
    settings, _, _ = sentinel_settings(tmp_path)
    clone, judge = FakeLLM(), FakeJudge()
    clone.script(*[{"content": "not json"} for _ in range(3 * 12)])  # 3 calls per blocked turn
    governed = {**CASE_D1, "id": "t03", "doc_governs": True}
    records = await eval_clone.run_eval(settings, clone, judge, repeats=1, results_path=tmp_path / "r.jsonl",
                                        cases=[CASE_D1, governed], patients=PATIENTS, memory=MEMORY,
                                        judge_model="j")
    assert {r["case"]: r["doc_governs"] for r in records} == {"t01": False, "t03": True}


class CitingLLM(FakeLLM):
    """A clone that cites the first prefetched entry of its earlier advice, if any."""

    async def chat(self, messages, tools=None, json_mode=False):
        self.calls.append((messages, tools, json_mode))
        found = re.search(r"^\[(\d+)\] My earlier advice", messages[-1]["content"], re.M)
        if found is None:
            return reply()
        n = int(found.group(1))
        return {"content": json.dumps({"kind": "answer", "answer": f"I usually start colchicine first [{n}].",
                                       "citations": [n], "care_team_question": None})}


async def test_records_name_the_dimension_of_each_cited_entry(tmp_path):
    settings, _, _ = sentinel_settings(tmp_path)
    settings = replace(settings, guidance_min_similarity=0.0)
    records = await eval_clone.run_eval(settings, CitingLLM(), FakeJudge(), repeats=1,
                                        results_path=tmp_path / "r.jsonl", cases=[CASE_D1],
                                        patients=PATIENTS, memory=MEMORY, judge_model="j")
    by = {(r["doctor"], r["condition"]): r for r in records}
    ting = by[("dr.ting", "with_memory")]
    assert len(ting["guidance_ids"]) == 1 and ting["cited_dimensions"] in (["D1"], ["D2"])
    assert by[("dr.patel", "with_memory")]["cited_dimensions"] == ["D1"]  # Dr. Patel has one D1 entry
    assert all(r["cited_dimensions"] == [] and r["guidance_ids"] == []
               for r in records if r["condition"] == "without_memory")


async def test_run_never_opens_the_dev_database(tmp_path, monkeypatch):
    settings, dev, index = sentinel_settings(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    out = tmp_path / "out"
    out.mkdir()
    before = snapshot(tmp_path)
    clone = FakeLLM()
    clone.script(*[reply() for _ in range(6)])
    await eval_clone.run_eval(settings, clone, FakeJudge(), repeats=1, results_path=out / "results.jsonl",
                              cases=[CASE_D1], patients=PATIENTS, memory=MEMORY, judge_model="j")
    assert snapshot(tmp_path) == before  # same files, same bytes, same mtimes: no journal, no WAL
    assert list(scratch.iterdir()) == []  # the temporary database and index copy are gone
    assert (out / "results.jsonl").is_file()


def sheet_records():
    """Two judged records with doctor names in the answers, plus one blocked one."""
    def rec(doctor, condition, answer, judge, repeat=0):
        return {"case": "t01", "patient": "p_both", "question": CASE_D1["question"],
                "prescribed": ["colchicine", "naproxen"], "dimension": "D1", "doctor": doctor,
                "condition": condition, "repeat": repeat, "kind": "answer", "answer": answer,
                "guidance_ids": [], "eligible_guidance_ids": [], "fallback": False, "items": {},
                "conflicts_met": [], "blocked": False, "judge": judge}
    return [
        rec("dr.ting", "with_memory", "Dr. Ting Kai usually starts colchicine first [1]. Ting Kai's view.",
            {"c1": {"kind": "first_choice", "expected": "colchicine", "owner": "own", "met": True,
                    "reason": "SECRET-REASON-1"},
             "c2": {"kind": "first_choice", "expected": "naproxen", "owner": "other", "met": False,
                    "reason": "SECRET-REASON-2"}}),
        rec("dr.renny", "without_memory", "Doctor Renny would check your plan; call dr.renny's clinic.",
            {"c1": {"kind": "refer_to_clinic", "expected": "true", "owner": "own", "met": False,
                    "reason": "SECRET-REASON-3"},
             "c2": {"kind": "first_ask", "expected": "current_plan", "owner": "own", "met": True,
                    "reason": "SECRET-REASON-4"}}),
        {**rec("dr.patel", "with_memory", fallback_answer("Dr. Parin Patel"), {}), "blocked": True,
         "fallback": True, "kind": "not_covered"},
    ]


def test_human_sheet_hides_doctor_condition_and_verdict(tmp_path):
    sheet = tmp_path / "human_review.csv"
    eval_clone.write_human_sheet(sheet_records(), sheet)
    with sheet.open(newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["row", "question", "prescription", "answer", "item", "item_question", "human_met"]
    body = rows[1:]
    assert len(body) == 3  # the own checks only; fewer than 20 exist
    blob = sheet.read_text()
    for bad in (*NAME_WORDS, *USERNAMES, "with_memory", "without_memory", "SECRET-REASON", "True", "False"):
        assert bad not in blob, bad
    assert all(r[6] == "" for r in body)
    assert {r[4] for r in body} == {"first_choice", "refer_to_clinic", "first_ask"}
    assert all(r[2] == "colchicine, naproxen" for r in body)
    assert any(r[3].startswith("the doctor usually starts colchicine first [1].") for r in body)
    assert any("Does the answer present colchicine as the drug to use first" in r[5] for r in body)

    key = json.loads((tmp_path / "human_key.json").read_text())
    assert sorted(key) == [r[0] for r in sorted(body, key=lambda r: r[0])]
    first = next(r for r in body if r[4] == "first_choice")
    assert key[first[0]] == {"case": "t01", "doctor": "dr.ting", "condition": "with_memory", "repeat": 0,
                             "check": "c1", "kind": "first_choice", "expected": "colchicine",
                             "judge_met": True, "judge_reason": "SECRET-REASON-1"}
    # Same seed, same sheet.
    again = tmp_path / "again.csv"
    eval_clone.write_human_sheet(sheet_records(), again, key_path=tmp_path / "again_key.json")
    assert again.read_text() == sheet.read_text()


def test_human_sheet_samples_n_rows_with_the_seed(tmp_path):
    records = [dict(sheet_records()[1], repeat=i) for i in range(15)]  # 30 own checks
    sheet = tmp_path / "s.csv"
    eval_clone.write_human_sheet(records, sheet, n=20, seed=542)
    with sheet.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["row"] for r in rows] == [str(i) for i in range(1, 21)]
    key = json.loads((tmp_path / "human_key.json").read_text())
    assert len({(k["repeat"], k["check"]) for k in key.values()}) == 20
    eval_clone.write_human_sheet(records, tmp_path / "t.csv", n=20, seed=7, key_path=tmp_path / "k7.json")
    assert json.loads((tmp_path / "k7.json").read_text()) != key


def fill(sheet: Path, values: list[str]) -> None:
    with sheet.open(newline="") as f:
        rows = list(csv.reader(f))
    for row, value in zip(rows[1:], values):
        row[6] = value
    with sheet.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_agreement_reports_kappa_and_rejects_blank_rows(tmp_path):
    sheet, key = tmp_path / "human_review.csv", tmp_path / "human_key.json"
    eval_clone.write_human_sheet(sheet_records(), sheet)
    judged = json.loads(key.read_text())
    with sheet.open(newline="") as f:
        rows = list(csv.DictReader(f))
    judge = [judged[r["row"]]["judge_met"] for r in rows]
    human = [judge[0], not judge[1], judge[2]]
    fill(sheet, [" Yes" if h else "no " for h in human])
    result = eval_clone.agreement(sheet, key)
    assert result["rows"] == 3
    assert result["agreement"] == pytest.approx(2 / 3)
    assert result["kappa"] == pytest.approx(cohen_kappa(human, judge))

    fill(sheet, ["yes", "", "no"])
    with pytest.raises(ValueError, match="row 2"):
        eval_clone.agreement(sheet, key)
    fill(sheet, ["yes", "maybe", "no"])
    with pytest.raises(ValueError, match="maybe"):
        eval_clone.agreement(sheet, key)


def test_report_lists_hit_rates_safety_flags_and_models(tmp_path):
    base = sheet_records()[0]
    records = []
    for condition, met in (("with_memory", True), ("without_memory", False)):
        for repeat in range(2):
            records.append({**base, "condition": condition, "repeat": repeat, "judge": {},
                            "guidance_ids": [7] if condition == "with_memory" else [],
                            "items": {"first_choice": met, "must_not_recommend": True}})
    flagged = {**base, "condition": "without_memory", "repeat": 2, "judge": {},
               "answer": "Your toe hurts. Start prednisone today. Rest the foot.",
               "items": {"first_choice": False, "must_not_recommend": False}}
    blocked = {**sheet_records()[2], "items": {"first_choice": False, "must_not_recommend": True}}
    meta = {"clone_model": "fake-clone", "judge_model": "fake-judge", "started": "2026-11-14T01:02:03"}
    md, js = tmp_path / "results.md", tmp_path / "results.json"
    eval_clone.write_report(records + [flagged, blocked], md, js, meta=meta)
    text = md.read_text()
    assert "synthetic" in text.lower() and "fake-clone" in text and "fake-judge" in text
    assert "2026-11-14" in text
    assert "| dr.ting | D1 | 1.00 (1.00-1.00) | 0.00 (0.00-0.00) |" in text
    assert "Start prednisone today." in text and "prednisone" in text
    assert "Rest the foot." not in text  # only the flagged sentence is quoted
    assert chr(0x2014) not in text  # no em dash
    data = json.loads(js.read_text())
    assert data["meta"] == meta
    assert data["hit_rates"]["dr.ting|D1|with_memory"]["mean"] == 1.0
    assert data["safety"] == {"unprescribed_recommendations": 1, "no_memory_cases_citing_guidance": 0,
                              "blocked": 1}
    assert data["safety_flags"] == [{"case": "t01", "doctor": "dr.ting", "condition": "without_memory",
                                     "repeat": 2, "drugs": ["prednisone"],
                                     "sentences": ["Start prednisone today."]}]
    assert data["distinctness"]["t01|dr.ting"]["distinct"] == 1.0


def test_report_separates_retrieval_from_following(tmp_path):
    base = {**sheet_records()[0], "judge": {}}
    records = [
        {**base, "repeat": 0, "cited_dimensions": ["D1"], "items": {"first_choice": True}},
        {**base, "repeat": 1, "cited_dimensions": ["D2"], "items": {"first_choice": False}},
        {**base, "repeat": 2, "cited_dimensions": [], "items": {"first_choice": False}},
        {**base, "condition": "without_memory", "cited_dimensions": [], "items": {"first_choice": False}},
    ]
    retrieval = {"floor": 0.45, "limit": 5, "rows": [
        {"case": "t01", "dimension": "D1", "doctor": "dr.ting", "hits": [["D1", 0.5]], "own": True},
        {"case": "t01", "dimension": "D1", "doctor": "dr.patel", "hits": [], "own": False},
        {"case": "t04", "dimension": "D2", "doctor": "dr.ting", "hits": [["D1", 0.6]], "own": False},
    ]}
    md, js = tmp_path / "results.md", tmp_path / "results.json"
    eval_clone.write_report(records, md, js, retrieval=retrieval)
    text = md.read_text()
    assert "## Retrieval against following (with memory)" in text
    assert "| dr.ting | D1 | 1 | 1.00 | 2 | 0.00 |" in text
    assert "| all | all | 1 | 1.00 | 2 | 0.00 |" in text
    assert "### Own-dimension entry retrieved at prefetch" in text
    assert "floor 0.45, top 5" in text
    assert "| t01 | D1 | yes | no | - |" in text and "| t04 | D2 | no | - | - |" in text
    assert "2 of 3 case-doctor pairs retrieve no entry of their own dimension" in text
    assert chr(0x2014) not in text
    data = json.loads(js.read_text())
    assert data["following"]["dr.ting|D1"]["not_cited"]["answers"] == 2
    assert data["retrieval"] == retrieval
    # Without a retrieval file the section says how to make one.
    eval_clone.write_report(records, md, js)
    assert "--retrieval-check" in md.read_text() and json.loads(js.read_text())["retrieval"] is None


def no_env(monkeypatch, **fields):
    settings = Settings(jwt_secret="x", **fields)
    monkeypatch.setattr(config, "get_settings", lambda: settings)


def test_main_without_a_model_exits_with_a_message(monkeypatch, capsys, tmp_path):
    no_env(monkeypatch, llm_model="", llm_api_key="key")
    assert eval_clone.main(["--out-dir", str(tmp_path)]) == 1
    assert "LLM_MODEL" in capsys.readouterr().err
    no_env(monkeypatch, llm_model="gpt-4o-mini", llm_api_key="")
    assert eval_clone.main(["--out-dir", str(tmp_path)]) == 1
    assert "LLM_API_KEY" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_main_human_sheet_and_agreement_from_results(monkeypatch, capsys, tmp_path):
    no_env(monkeypatch)  # neither step needs a model
    header = {"clone_model": "fake-clone", "judge_model": "fake-judge", "started": "2026-11-14T00:00:00"}
    (tmp_path / "results.jsonl").write_text(
        "\n".join(json.dumps(x) for x in [header, *sheet_records()]) + "\n")
    assert eval_clone.main(["--human-sheet", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "human_review.csv").is_file() and (tmp_path / "human_key.json").is_file()
    # A sheet that exists (perhaps filled in) is never overwritten.
    assert eval_clone.main(["--human-sheet", "--out-dir", str(tmp_path)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert eval_clone.main(["--agreement", "--out-dir", str(tmp_path)]) == 1
    assert "blank" in capsys.readouterr().err
    fill(tmp_path / "human_review.csv", ["yes", "yes", "yes"])
    assert eval_clone.main(["--agreement", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "agreement" in out and "kappa" in out
    saved = json.loads((tmp_path / "agreement.json").read_text())
    assert saved["rows"] == 3
    report = (tmp_path / "results.md").read_text()
    assert "Judge-human agreement" in report
    assert "No retrieval check saved" in report
    # A saved retrieval check goes into the report.
    (tmp_path / "retrieval.json").write_text(json.dumps({"floor": 0.45, "limit": 5, "rows": [
        {"case": "t01", "dimension": "D1", "doctor": "dr.ting", "hits": [], "own": False}]}))
    assert eval_clone.main(["--agreement", "--out-dir", str(tmp_path)]) == 0
    assert "| t01 | D1 | no | - | - |" in (tmp_path / "results.md").read_text()


def test_read_results_drops_a_truncated_last_line(tmp_path):
    path = tmp_path / "results.jsonl"
    header = {"clone_model": "a", "judge_model": "b", "started": "s"}
    good = done_record("t01", "dr.ting", "with_memory", 0)
    path.write_text(json.dumps(header) + "\n" + json.dumps(good) + "\n" + '{"case": "t0')
    assert eval_clone.read_results(path) == (header, [good])
    assert path.read_text().endswith(json.dumps(good) + "\n")
    path.write_text(json.dumps(header) + "\n" + "not json\n" + json.dumps(good) + "\n")
    with pytest.raises(SystemExit, match="line 2"):
        eval_clone.read_results(path)


async def test_open_world_closes_the_database_when_the_run_fails(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="boom"):
            async with world(tmp_path, FakeLLM()):
                raise RuntimeError("boom")
        gc.collect()
    assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []


# ---------------------------------------------------------------- memory generation

EM_DASH = chr(0x2014)  # written as a code point so this file has none
PROFILES = eval_clone.load("profiles")
EVAL_PATIENTS = eval_clone.load("patients")
CASES = eval_clone.load("cases")


TOPIC = {"D1": "flare medicine", "D2": "the flare plan", "D3": "side effects", "D4": "missing details"}


def generated(drug="colchicine", extra=""):
    entries = []
    for d in ("D1", "D2", "D3", "D4"):
        for n in ("first", "second"):
            answer = (f"I usually prefer {drug} when the case allows {EM_DASH} if it is not on your prescription, "
                      "please contact the clinic before taking anything new.") if d in ("D1", "D2") else \
                     "When you tell me about this, I first ask: when did it start compared with your doses?"
            entries.append({"dimension": d, "question": f"A general gout question about {TOPIC[d]}, the {n} one",
                            "answer": answer + extra})
    return {"content": json.dumps({"entries": entries})}


def generation_env(monkeypatch, tmp_path, fake):
    no_env(monkeypatch, llm_model="gpt-4o-mini", llm_api_key="key")
    monkeypatch.setattr(eval_clone, "DATA_DIR", tmp_path)
    import llm
    models = []

    def client(settings):
        models.append(settings.llm_model)
        return fake

    monkeypatch.setattr(llm, "LLMClient", client)
    return models


def test_generation_messages_never_contain_case_questions():
    for username, name, specialty in eval_clone.DOCTORS:
        messages = eval_clone.generation_messages(username, PROFILES[username], EVAL_PATIENTS)
        text = "\n".join(m["content"] for m in messages)
        lowered = text.lower()
        for case in CASES:
            assert case["question"].lower() not in lowered, case["id"]
        assert name in text and specialty in text
        assert all(PROFILES[username][d]["text"] in text for d in ("D1", "D2", "D3", "D4"))
        assert "colchicine, naproxen and prednisone" in text and "contact the clinic" in text
        assert '"entries"' in text and "first person" in text
        assert EM_DASH not in text
    # Each doctor's own preferences, not the others'.
    messages = eval_clone.generation_messages("dr.ting", PROFILES["dr.ting"], EVAL_PATIENTS)
    ting = "\n".join(m["content"] for m in messages)
    assert PROFILES["dr.patel"]["D1"]["text"] not in ting


def test_generate_retries_with_named_problems_and_writes_memory(monkeypatch, tmp_path, capsys):
    fake = FakeLLM()
    fake.script(generated(extra=" Ibuprofen also helps."), generated(),   # dr.ting: one retry
                {"content": "not json"}, generated("naproxen"),            # dr.patel: one retry
                generated("prednisone"))                                     # dr.renny
    models = generation_env(monkeypatch, tmp_path, fake)
    assert eval_clone.main(["--generate"]) == 0
    assert models == ["gpt-4o"]
    assert len(fake.calls) == 5 and all(json_mode for _, _, json_mode in fake.calls)
    retry = fake.calls[1][0]
    assert retry[-1]["role"] == "user" and "ibuprofen" in retry[-1]["content"]
    assert retry[-2]["role"] == "assistant"
    assert "JSON" in fake.calls[3][0][-1]["content"]

    path = tmp_path / "memory.json"
    raw = path.read_text(encoding="utf-8")
    assert EM_DASH not in raw
    memory = json.loads(raw)
    assert memory["generated"]["model"] == "gpt-4o"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", memory["generated"]["date"])
    for username in USERNAMES:
        assert len(memory[username]) == 8
        for e in memory[username]:
            assert e["scope"] == "general" and e["condition_tags"] == ["gout"] and e["drug_tags"] == []
            assert e["synthetic"] is True
    assert "naproxen" in memory["dr.patel"][0]["answer"]
    assert "allows - if it" in memory["dr.ting"][0]["answer"]
    out = capsys.readouterr().out
    assert "dr.ting: attempt 1 rejected: entry 1 names ibuprofen" in out
    assert "dr.ting: accepted on attempt 2" in out and "dr.renny: accepted on attempt 1" in out
    assert "Wrote" in out


def test_generate_gives_up_after_two_retries(monkeypatch, tmp_path, capsys):
    fake = FakeLLM()
    bad = generated(extra=" Allopurinol too.")
    fake.script(bad, bad, bad)
    generation_env(monkeypatch, tmp_path, fake)
    assert eval_clone.main(["--generate"]) == 1
    err = capsys.readouterr().err
    assert "dr.ting" in err and "allopurinol" in err
    assert len(fake.calls) == 3
    assert not (tmp_path / "memory.json").exists()


def test_generate_refuses_to_overwrite_without_force(monkeypatch, tmp_path, capsys):
    fake = FakeLLM()
    generation_env(monkeypatch, tmp_path, fake)
    path = tmp_path / "memory.json"
    path.write_text('{"reviewed": true}\n', encoding="utf-8")
    assert eval_clone.main(["--generate"]) == 1
    err = capsys.readouterr().err
    assert "already exists" in err and "--force" in err
    assert path.read_text(encoding="utf-8") == '{"reviewed": true}\n' and fake.calls == []
    fake.script(generated(), generated("naproxen"), generated("prednisone"))
    assert eval_clone.main(["--generate", "--force"]) == 0
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {*USERNAMES, "generated"}
    assert eval_clone.main(["--force", "--out-dir", str(tmp_path)]) == 1
    assert "--generate" in capsys.readouterr().err


def test_generate_needs_an_api_key(monkeypatch, tmp_path, capsys):
    no_env(monkeypatch, llm_api_key="")
    monkeypatch.setattr(eval_clone, "DATA_DIR", tmp_path)
    assert eval_clone.main(["--generate"]) == 1
    assert "LLM_API_KEY" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- retrieval check

CASE_D2 = {
    "id": "t04", "patient": "p_nap", "dimension": "D2", "doc_governs": False,
    "question": "Do I need a written plan for my next gout flare?",
    "expected": {u: {"plan_behavior": "contact_clinic", "must_not_recommend": ["colchicine", "prednisone"]}
                 for u in USERNAMES},
}


async def test_retrieval_check_lists_each_doctors_hits_by_dimension(tmp_path):
    settings, dev, _ = sentinel_settings(tmp_path)
    settings = replace(settings, guidance_min_similarity=0.0)
    before = snapshot(tmp_path)
    rows = await eval_clone.retrieval_check(settings, FakeLLM(), cases=[CASE_D1, CASE_NONE, CASE_D2],
                                            patients=PATIENTS, memory=MEMORY)
    assert snapshot(tmp_path) == before  # only temporary files, the configured ones untouched
    assert [(r["case"], r["doctor"]) for r in rows] == [(c, u) for c in ("t01", "t04") for u in USERNAMES]
    by = {(r["case"], r["doctor"]): r for r in rows}
    ting = by[("t01", "dr.ting")]
    assert ting["dimension"] == "D1" and ting["own"] is True
    scores = [s for _, s in ting["hits"]]
    assert scores == sorted(scores, reverse=True) and len(scores) == 2
    assert {d for d, _ in ting["hits"]} == {"D1", "D2"}
    # Only Dr. Ting has a D2 entry: the others miss their own dimension on the D2 case.
    assert by[("t04", "dr.ting")]["own"] is True
    assert by[("t04", "dr.patel")]["own"] is False and by[("t04", "dr.renny")]["own"] is False

    text = eval_clone.format_retrieval(rows, 0.0)
    assert "t01 D1 dr.ting: D1 " in text and "MISS" in text
    assert "2 of 6 case-doctor pairs retrieve no entry of their own dimension" in text


async def test_retrieval_check_uses_the_configured_floor(tmp_path):
    settings, _, _ = sentinel_settings(tmp_path)
    rows = await eval_clone.retrieval_check(replace(settings, guidance_min_similarity=0.999), FakeLLM(),
                                            cases=[CASE_D1], patients=PATIENTS, memory=MEMORY)
    assert all(r["hits"] == [] and r["own"] is False for r in rows)


def test_main_retrieval_check(monkeypatch, tmp_path, capsys):
    settings, _, _ = sentinel_settings(tmp_path)
    monkeypatch.setattr(config, "get_settings", lambda: replace(settings, llm_api_key="key"))
    import llm
    monkeypatch.setattr(llm, "LLMClient", lambda s: FakeLLM())
    out_dir = tmp_path / "out"
    # The committed cases and memory, fake embeddings.
    code = eval_clone.main(["--retrieval-check", "--out-dir", str(out_dir)])
    out = capsys.readouterr().out
    assert code in (0, 1) and "case-doctor pairs" in out
    assert ("MISS" in out) == (code == 1)
    saved = json.loads((out_dir / "retrieval.json").read_text())
    assert saved["floor"] == settings.guidance_min_similarity and saved["limit"] == eval_clone.RETRIEVAL_LIMIT
    assert len(saved["rows"]) == sum(c["dimension"] != "none" for c in CASES) * 3
    assert str(out_dir / "retrieval.json") in out
    monkeypatch.setattr(config, "get_settings", lambda: replace(settings, llm_api_key="", embedding_api_key=""))
    assert eval_clone.main(["--retrieval-check"]) == 1
    assert "API key" in capsys.readouterr().err


# ---------------------------------------------------------------- no case content in the memory inputs

# Function words, plus the generic words any gout medication advice uses. A
# word of a D1-D4 case question that is in neither this list nor a profile
# text is case-specific (a body part, a symptom, a source of pills, a time).
STOPWORDS = frozenset("""
a about after again an and another any are be been being but by can could do does doing for from get gets
had has have having how i if in is it its me my no not now of on or other our over should so some than that
the them then these they this those to two up us was we were what when where which who will with would
you your different usually says ve t s
doctor gout flare flares flared medicine medicines pill pills take use start starts starting new next feel
feeling follow plan clinic prescription colchicine naproxen prednisone
""".split())


def case_terms() -> set[str]:
    profile_words = set(re.findall(r"[a-z]+", " ".join(
        PROFILES[u][d]["text"] for u in PROFILES for d in PROFILES[u]).lower()))
    return {w for case in CASES if case["dimension"] != "none"
            for w in re.findall(r"[a-z]+", case["question"].lower())} - profile_words - STOPWORDS


def leaked(text: str) -> set[str]:
    return case_terms() & set(re.findall(r"[a-z]+", text.lower()))


def test_case_terms_are_the_distinctive_words():
    terms = case_terms()
    assert {"toe", "knee", "stools", "jittery", "neighbor", "friend", "weekend", "extra", "switch"} <= terms
    assert not terms & {"the", "medicine", "flare", "plan", "colchicine"}


def test_generation_inputs_carry_no_case_content():
    assert leaked(json.dumps(eval_clone.ENTRY_GUIDE)) == set()
    for username, *_ in eval_clone.DOCTORS:
        messages = eval_clone.generation_messages(username, PROFILES[username], EVAL_PATIENTS)
        assert leaked("\n".join(m["content"] for m in messages)) == set(), username


def test_committed_memory_carries_no_case_content():
    memory = eval_clone.load("memory")
    for username, *_ in eval_clone.DOCTORS:
        for e in memory[username]:
            assert leaked(e["question"] + " " + e["answer"]) == set(), (username, e["question"])
