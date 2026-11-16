"""The guidance floor and margin sweep: linking, scoring and the choice. Offline."""
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import sweep_guidance_floor as sweep  # noqa: E402
from db import init_db, make_engine, make_session_factory  # noqa: E402
from fake_llm import FakeLLM  # noqa: E402
from guidance import embed_entry  # noqa: E402
from models import DischargeDocument, DoctorGuidance, User  # noqa: E402

ENTRIES = [
    {"doctor": "dr.a", "question": "Can I drink coffee?", "answer": "You can have 1 cup of coffee in the morning."},
    {"doctor": "dr.a", "question": "Is pink urine normal?", "answer": "Pink urine is common while the stent is in."},
    {"doctor": "dr.b", "question": "Coffee?", "answer": "Two cups of coffee are fine."},
]


def test_needed_entries_match_the_facts_in_the_doctors_own_entries():
    q = {"doctor": "dr.a", "facts": ["1 cup of coffee in the morning"]}
    assert sweep.needed_entries(q, ENTRIES) == ["Can I drink coffee?"]
    assert sweep.needed_entries({"doctor": "dr.a", "facts": ["at bedtime"]}, ENTRIES) == []  # a document fact
    assert sweep.needed_entries({"doctor": "dr.b", "facts": ["1 cup of coffee in the morning"]}, ENTRIES) == []


def test_a_tie_accepts_every_tied_entry():
    both = [{"doctor": "d", "question": "one", "answer": "call the clinic"},
            {"doctor": "d", "question": "two", "answer": "Please call the clinic."}]
    assert sorted(sweep.needed_entries({"doctor": "d", "facts": ["call the clinic"]}, both)) == ["one", "two"]


def rows_fixture():
    # Answerable question 1: its rule scores 0.20 below a best of 0.40. Answerable question 2: 0.35 is the best.
    # The escalate question has four mediocre rules.
    return [
        {"id": "a1", "expect": "answer", "needed": ["n1"],
         "hits": [("x", 0.40), ("y", 0.30), ("n1", 0.20)]},
        {"id": "a2", "expect": "answer", "needed": ["n2"], "hits": [("n2", 0.35), ("z", 0.10)]},
        {"id": "e1", "expect": "escalate", "needed": [],
         "hits": [("p", 0.30), ("q", 0.25), ("r", 0.22), ("s", 0.12)]},
    ]


def test_evaluate_counts_kept_rules_and_escalate_rules():
    wide = sweep.evaluate(rows_fixture(), 0.10, 0.30)
    assert (wide["kept"], wide["answerable"], wide["escalate_rules"]) == (2, 2, 4.0)
    tight = sweep.evaluate(rows_fixture(), 0.10, 0.10)
    assert tight["kept"] == 1  # a1's rule is 0.20 below the best
    old = sweep.evaluate(rows_fixture(), 0.45, None)
    assert (old["kept"], old["escalate_rules"]) == (0, 0.0)


def test_choose_never_exceeds_the_escalate_cap_even_when_recall_is_higher():
    table = sweep.sweep(rows_fixture())
    recall_max = max(table, key=lambda r: (r["kept"], -r["escalate_rules"]))
    assert recall_max["kept"] == 2 and recall_max["escalate_rules"] >= 3.0
    chosen = sweep.choose(table, cap=2.0)
    assert chosen["escalate_rules"] <= 2.0 and chosen["kept"] < 2
    assert sweep.choose(table, cap=4.0)["kept"] == 2


def test_choose_with_no_pair_within_the_cap_raises():
    with pytest.raises(ValueError):
        sweep.choose([{"floor": 0.1, "margin": 0.1, "kept": 2, "escalate_rules": 3.0}], cap=1.0)


def test_choose_prefers_the_higher_floor_then_the_smaller_margin_on_a_tie():
    table = [
        {"floor": 0.20, "margin": 0.10, "kept": 2, "escalate_rules": 1.0},
        {"floor": 0.30, "margin": 0.20, "kept": 2, "escalate_rules": 1.0},
        {"floor": 0.30, "margin": 0.10, "kept": 2, "escalate_rules": 1.0},
        {"floor": 0.40, "margin": 0.05, "kept": 1, "escalate_rules": 0.0},
    ]
    assert sweep.choose(table, cap=1.0) == table[2]


@pytest_asyncio.fixture
async def session():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    async with make_session_factory(engine)() as s:
        yield s
    await engine.dispose()


async def test_score_questions_returns_every_gated_hit_with_its_score(session):
    llm = FakeLLM()
    users = [User(username=n, password_hash="x") for n in ("dr.a", "demo")]
    session.add_all(users)
    await session.flush()
    session.add(DischargeDocument(user_id=users[1].id, text="gout colchicine", active_user_id=users[1].id))
    await session.flush()
    for question, answer, drug in (("colchicine diarrhea", "reduce the dose", "colchicine"),
                                   ("warfarin kale", "keep kale steady", "warfarin")):
        e = DoctorGuidance(doctor_id=users[0].id, question=question, answer=answer, scope="general",
                           drug_tags=[drug], condition_tags=[], active=True)
        assert await embed_entry(llm, e)
        session.add(e)
    await session.flush()
    doc_id = await session.scalar(select(DischargeDocument.id))
    questions = [{"id": "q1", "patient": "demo", "doctor": "dr.a", "expect": "answer",
                  "question": "colchicine diarrhea", "facts": ["reduce the dose"]},
                 {"id": "q2", "patient": "demo", "doctor": "dr.a", "expect": "escalate",
                  "question": "colchicine diarrhea", "facts": []}]
    entries = [{"doctor": "dr.a", "question": "colchicine diarrhea", "answer": "reduce the dose"}]
    rows = await sweep.score_questions(session, llm, questions, entries,
                                       {"demo": users[1].id}, {"demo": doc_id})
    assert [r["hits"][0][0] for r in rows] == ["colchicine diarrhea"] * 2  # warfarin's entry is gated out
    assert len(rows[0]["hits"]) == 1 and rows[0]["hits"][0][1] > 0.5
    assert rows[0]["needed"] == ["colchicine diarrhea"] and rows[1]["needed"] == []
