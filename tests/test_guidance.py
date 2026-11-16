import os

import pytest
import pytest_asyncio
from sqlalchemy import text

from config import Settings, get_settings
from db import init_db, make_engine, make_session_factory
from fake_llm import FakeLLM
from guidance import embed_entry, eligible, patient_text, retrieve, tag_names
from llm import LLMError
from models import DischargeDocument, DoctorGuidance, ExtractedItem, User


def entry(**kw) -> DoctorGuidance:
    base = dict(
        doctor_id=1, question="q", answer="a", scope="general",
        drug_tags=[], condition_tags=[], active=True,
    )
    base.update(kw)
    return DoctorGuidance(**base)


def test_tag_names_splits_and_strips():
    assert tag_names("colchicine / Colcrys") == ["colchicine", "Colcrys"]
    assert tag_names(" a // b / ") == ["a", "b"]
    assert tag_names("") == []


def test_gate_requires_every_drug_and_one_condition():
    e = entry(drug_tags=["colchicine / Colcrys"], condition_tags=["gout"])
    assert eligible(e, "Take Colcrys 0.6 mg daily for a gout flare.")
    assert not eligible(e, "Patient history of gout.")
    assert not eligible(e, "Take colchicine 0.6 mg daily.")
    two = entry(drug_tags=["colchicine", "allopurinol"], condition_tags=["gout"])
    assert not eligible(two, "colchicine for gout")
    assert eligible(two, "colchicine and allopurinol for gout")


def test_gate_condition_any_of_and_drug_only():
    e = entry(drug_tags=["colchicine"], condition_tags=["gout", "pseudogout"])
    assert eligible(e, "colchicine for pseudogout")
    assert eligible(entry(drug_tags=["colchicine"]), "colchicine 0.6 mg")
    assert eligible(entry(condition_tags=["gout"]), "gout flare")


def test_gate_whole_words():
    assert not eligible(entry(condition_tags=["gout"]), "gouty arthritis history")
    assert not eligible(entry(drug_tags=["ace inhibitor"]), "replace inhibitor")
    assert eligible(entry(drug_tags=["ace inhibitor"]), "Take an ACE   inhibitor.")


def test_untagged_or_patient_scope_never_eligible():
    assert not eligible(entry(), "anything gout")
    assert not eligible(entry(scope="patient", drug_tags=["colchicine"]), "colchicine")
    assert not eligible(entry(drug_tags=["/"]), "colchicine")


def test_patient_text_skips_rejected_items():
    doc = DischargeDocument(user_id=1, text="Discharge for gout")
    items = [
        ExtractedItem(document_id=1, category="medication", text="colchicine", source_quote="q", status="active"),
        ExtractedItem(document_id=1, category="medication", text="warfarin", source_quote="q", status="rejected"),
    ]
    text = patient_text(doc, items)
    assert "gout" in text and "colchicine" in text and "warfarin" not in text


@pytest_asyncio.fixture
async def session():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    async with make_session_factory(engine)() as s:
        yield s
    await engine.dispose()


async def seed(session):
    users = [User(username=n, password_hash="x") for n in ("doc", "a", "b")]
    session.add_all(users)
    await session.flush()
    docs = [DischargeDocument(user_id=users[1].id, text="gout colchicine") for _ in range(2)]
    session.add_all(docs)
    await session.flush()
    return users, docs


async def add(session, llm, **kw):
    e = entry(**kw)
    assert await embed_entry(llm, e)
    session.add(e)
    await session.flush()
    return e


async def test_retrieve_scope_and_document(session):
    llm = FakeLLM()
    (doc, a, b), (d1, d2) = await seed(session)
    e = await add(
        session, llm, doctor_id=doc.id, scope="patient", patient_id=a.id,
        document_id=d1.id, question="Can I drink coffee?", answer="Yes, coffee is fine.",
    )
    q = "coffee"
    got = await retrieve(session, llm, a, d1, [], q, floor=0.1, margin=1.0, doctor_id=doc.id)
    assert [g.id for g, _ in got] == [e.id]
    assert got[0][1] > 0.1
    assert await retrieve(session, llm, b, d1, [], q, floor=0.1, margin=1.0, doctor_id=doc.id) == []
    assert await retrieve(session, llm, a, d2, [], q, floor=0.1, margin=1.0, doctor_id=doc.id) == []


async def test_retrieve_general_uses_gate_and_ranks(session):
    llm = FakeLLM()
    (doc, a, b), (d1, d2) = await seed(session)
    good = await add(session, llm, doctor_id=doc.id, drug_tags=["colchicine"],
                     question="colchicine diarrhea", answer="reduce dose")
    await add(session, llm, doctor_id=doc.id, drug_tags=["warfarin"],
              question="colchicine diarrhea", answer="reduce dose")
    got = await retrieve(session, llm, a, d1, [], "colchicine diarrhea", floor=0.1, margin=1.0, doctor_id=doc.id)
    assert [g.id for g, _ in got] == [good.id]
    assert await retrieve(session, llm, a, d1, [], "unrelated zebra", floor=0.5, margin=1.0, doctor_id=doc.id) == []


async def test_retrieve_orders_best_first_and_limits(session):
    llm = FakeLLM()
    (doc, a, b), (d1, d2) = await seed(session)
    far = await add(session, llm, doctor_id=doc.id, drug_tags=["colchicine"],
                    question="colchicine diarrhea water", answer="more words here")
    near = await add(session, llm, doctor_id=doc.id, drug_tags=["colchicine"],
                     question="colchicine diarrhea", answer="")
    got = await retrieve(session, llm, a, d1, [], "colchicine diarrhea", floor=0.0, margin=1.0, doctor_id=doc.id)
    assert [g.id for g, _ in got] == [near.id, far.id]
    assert got[0][1] >= got[1][1]
    assert len(await retrieve(session, llm, a, d1, [], "colchicine diarrhea",
                              floor=0.0, margin=1.0, doctor_id=doc.id, limit=1)) == 1


async def test_retrieve_skips_retired_unembedded_and_other_model(session):
    llm = FakeLLM()
    (doc, a, b), (d1, d2) = await seed(session)
    common = dict(doctor_id=doc.id, scope="patient", patient_id=a.id, document_id=d1.id,
                  question="coffee", answer="coffee")
    await add(session, llm, active=False, **common)
    session.add(entry(**common))
    old = await add(session, llm, **common)
    old.embedding_model = "old"
    await session.flush()
    assert await retrieve(session, llm, a, d1, [], "coffee", floor=0.0, margin=1.0, doctor_id=doc.id) == []


async def test_retrieve_skips_mismatched_vector_length(session):
    llm = FakeLLM()
    (doc, a, b), (d1, d2) = await seed(session)
    e = await add(session, llm, doctor_id=doc.id, scope="patient", patient_id=a.id,
                  document_id=d1.id, question="coffee", answer="coffee")
    e.embedding = [1.0, 0.0]
    await session.flush()
    assert await retrieve(session, llm, a, d1, [], "coffee", floor=0.0, margin=1.0, doctor_id=doc.id) == []


class FailingLLM(FakeLLM):
    async def embed(self, texts):
        raise LLMError("down")


async def test_embed_entry_failure_returns_false():
    e = entry()
    assert await embed_entry(FailingLLM(), e) is False
    assert e.embedding is None
    assert await embed_entry(FakeLLM(), e) is True
    assert e.embedding and e.embedding_model == FakeLLM().embedding_model


async def test_failed_embedding_is_sql_null_and_skipped(session):
    (doc, a, b), (d1, d2) = await seed(session)
    e = entry(doctor_id=doc.id, scope="patient", patient_id=a.id, document_id=d1.id,
              question="coffee", answer="coffee")
    assert await embed_entry(FailingLLM(), e) is False
    session.add(e)
    await session.flush()
    row = (await session.execute(
        text("select embedding is null from doctor_guidance where id = :i"), {"i": e.id}
    )).one()
    assert row[0] == 1
    assert await retrieve(session, FakeLLM(), a, d1, [], "coffee", floor=0.0, margin=1.0, doctor_id=doc.id) == []


async def test_retrieve_only_own_doctor(session):
    llm = FakeLLM()
    (doc, a, b), (d1, d2) = await seed(session)
    other = User(username="other-doc", password_hash="x", role="doctor")
    session.add(other)
    await session.flush()
    common = dict(drug_tags=["colchicine"], condition_tags=["gout"],
                  question="colchicine diarrhea", answer="reduce dose")
    mine = await add(session, llm, doctor_id=doc.id, **common)
    theirs = await add(session, llm, doctor_id=other.id, **common)
    own = await add(session, llm, doctor_id=other.id, scope="patient", patient_id=a.id,
                    document_id=d1.id, question="colchicine diarrhea", answer="reduce dose")
    got = await retrieve(session, llm, a, d1, [], "colchicine diarrhea", floor=0.1, margin=1.0, doctor_id=doc.id)
    assert [g.id for g, _ in got] == [mine.id]
    got = await retrieve(session, llm, a, d1, [], "colchicine diarrhea", floor=0.1, margin=1.0, doctor_id=other.id)
    assert sorted(g.id for g, _ in got) == sorted([theirs.id, own.id])


async def test_retrieve_requires_a_doctor_id(session):
    (doc, a, b), (d1, d2) = await seed(session)
    with pytest.raises(TypeError):
        await retrieve(session, FakeLLM(), a, d1, [], "coffee", floor=0.0, margin=1.0)


class VectorLLM(FakeLLM):
    """Every query embeds to the unit x vector, so an entry's score is the x of its own unit vector."""
    def __init__(self):
        super().__init__("vec")

    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


async def scored_entries(session, scores):
    llm = VectorLLM()
    (doc, a, _), (d1, _) = await seed(session)
    for i, score in enumerate(scores):
        session.add(entry(doctor_id=doc.id, drug_tags=["colchicine"], question=f"q{i}", answer=f"a{i}",
                          embedding=[score, (1 - score ** 2) ** 0.5], embedding_model="vec"))
    await session.flush()
    return llm, doc, a, d1


async def test_margin_drops_hits_far_below_the_best(session):
    llm, doc, a, d1 = await scored_entries(session, [0.40, 0.38, 0.20])
    got = await retrieve(session, llm, a, d1, [], "q", floor=0.1, margin=0.1, doctor_id=doc.id)
    assert [round(s, 2) for _, s in got] == [0.40, 0.38]


async def test_floor_still_applies(session):
    llm, doc, a, d1 = await scored_entries(session, [0.08, 0.07])
    assert await retrieve(session, llm, a, d1, [], "q", floor=0.1, margin=0.1, doctor_id=doc.id) == []


async def test_margin_is_measured_from_the_best_even_when_it_is_the_only_hit_above_the_floor(session):
    llm, doc, a, d1 = await scored_entries(session, [0.30, 0.25, 0.05])
    got = await retrieve(session, llm, a, d1, [], "q", floor=0.1, margin=0.2, doctor_id=doc.id)
    assert [round(s, 2) for _, s in got] == [0.30, 0.25]


async def test_habit_rules_are_never_retrieved(session):
    from models import GuidanceRule
    llm = VectorLLM()
    (doc, a, _), (d1, _) = await seed(session)
    e = entry(doctor_id=doc.id, drug_tags=["colchicine"], embedding=[0.1, 0.99], embedding_model="vec",
              rules_status="ready")
    session.add(e)
    await session.flush()
    session.add(GuidanceRule(guidance_id=e.id, doctor_id=doc.id, position=0, statement="habit", quote="habit", kind="habit",
                             embedding=[1.0, 0.0], embedding_model="vec"))
    session.add(GuidanceRule(guidance_id=e.id, doctor_id=doc.id, position=1, statement="advice", quote="advice", kind="advice",
                             embedding=[0.5, 0.87], embedding_model="vec"))
    await session.flush()
    got = await retrieve(session, llm, a, d1, [], "q", floor=0.0, margin=1.0, doctor_id=doc.id)
    assert [u.rule.statement for u, _ in got] == ["advice"]


@pytest.fixture
def clean_settings_cache():
    saved = dict(os.environ)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    os.environ.clear()
    os.environ.update(saved)


def test_old_env_name_sets_the_floor(monkeypatch, clean_settings_cache):
    monkeypatch.delenv("GUIDANCE_FLOOR", raising=False)
    monkeypatch.setenv("GUIDANCE_MIN_SIMILARITY", "0.33")
    assert get_settings().guidance_floor == 0.33
    get_settings.cache_clear()
    monkeypatch.setenv("GUIDANCE_FLOOR", "0.2")  # the new name wins
    assert get_settings().guidance_floor == 0.2


def test_margin_and_top_k_come_from_the_environment(monkeypatch, clean_settings_cache):
    monkeypatch.setenv("GUIDANCE_MARGIN", "0.07")
    monkeypatch.setenv("GUIDANCE_TOP_K", "4")
    settings = get_settings()
    assert settings.guidance_margin == 0.07 and settings.guidance_top_k == 4
    assert Settings().guidance_top_k == 6


async def _entry_with_rules(session, rules, **kw):
    from models import GuidanceRule
    (doc, a, _), (d1, _) = await seed(session)
    e = entry(doctor_id=doc.id, drug_tags=["colchicine"], embedding=[0.1, 0.99], embedding_model="vec", **kw)
    session.add(e)
    await session.flush()
    for i, extra in enumerate(rules):
        session.add(GuidanceRule(guidance_id=e.id, doctor_id=doc.id, position=i, statement=f"r{i}", quote=f"r{i}",
                                 embedding=[1.0, 0.0], embedding_model="vec", **extra))
    await session.flush()
    return doc, a, d1


async def test_ready_entry_with_only_habit_rules_adds_nothing(session):
    doc, a, d1 = await _entry_with_rules(session, [{"kind": "habit", "habit_type": "ask"}], rules_status="ready")
    assert await retrieve(session, VectorLLM(), a, d1, [], "q", floor=0.0, margin=1.0, doctor_id=doc.id) == []


async def test_ready_entry_with_only_suppressed_rules_adds_nothing(session):
    doc, a, d1 = await _entry_with_rules(session, [{"suppressed": True}], rules_status="ready")
    assert await retrieve(session, VectorLLM(), a, d1, [], "q", floor=0.0, margin=1.0, doctor_id=doc.id) == []


async def test_entry_not_ready_still_scores_whole(session):
    doc, a, d1 = await _entry_with_rules(session, [{"kind": "habit"}], rules_status="failed")
    got = await retrieve(session, VectorLLM(), a, d1, [], "q", floor=0.0, margin=1.0, doctor_id=doc.id)
    assert len(got) == 1


async def _ready_entry(session, rows):
    from models import GuidanceRule
    (doc, a, _), (d1, _) = await seed(session)
    e = entry(doctor_id=doc.id, drug_tags=["colchicine"], embedding=[0.1, 0.99], embedding_model="vec",
              rules_status="ready")
    session.add(e)
    await session.flush()
    for i, extra in enumerate(rows):
        session.add(GuidanceRule(guidance_id=e.id, doctor_id=doc.id, position=i, statement=f"r{i}",
                                 quote=f"r{i}", embedding=[1.0, 0.0], **extra))
    await session.flush()
    return doc, a, d1


async def test_stale_suppressed_habit_does_not_send_the_entry_to_whole_scoring(session):
    doc, a, d1 = await _ready_entry(session, [
        {"embedding_model": "vec"},
        {"embedding_model": "old", "kind": "habit", "habit_type": "ask", "suppressed": True},
    ])
    got = await retrieve(session, VectorLLM(), a, d1, [], "q", floor=0.0, margin=1.0, doctor_id=doc.id)
    assert [u.rule.statement for u, _ in got] == ["r0"]


async def test_ready_entry_with_only_a_stale_suppressed_advice_row_adds_nothing(session):
    doc, a, d1 = await _ready_entry(session, [{"embedding_model": "old", "suppressed": True}])
    assert await retrieve(session, VectorLLM(), a, d1, [], "q", floor=0.0, margin=1.0, doctor_id=doc.id) == []


async def test_stale_live_advice_still_falls_back_to_whole_entry(session):
    doc, a, d1 = await _ready_entry(session, [{"embedding_model": "old"}])
    got = await retrieve(session, VectorLLM(), a, d1, [], "q", floor=0.0, margin=1.0, doctor_id=doc.id)
    assert len(got) == 1 and got[0][0].id is not None and not hasattr(got[0][0], "rule")
