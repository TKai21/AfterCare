import pytest
import pytest_asyncio

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
    got = await retrieve(session, llm, a, d1, [], q, 0.1)
    assert [g.id for g, _ in got] == [e.id]
    assert got[0][1] > 0.1
    assert await retrieve(session, llm, b, d1, [], q, 0.1) == []
    assert await retrieve(session, llm, a, d2, [], q, 0.1) == []


async def test_retrieve_general_uses_gate_and_ranks(session):
    llm = FakeLLM()
    (doc, a, b), (d1, d2) = await seed(session)
    good = await add(session, llm, doctor_id=doc.id, drug_tags=["colchicine"],
                     question="colchicine diarrhea", answer="reduce dose")
    await add(session, llm, doctor_id=doc.id, drug_tags=["warfarin"],
              question="colchicine diarrhea", answer="reduce dose")
    got = await retrieve(session, llm, a, d1, [], "colchicine diarrhea", 0.1)
    assert [g.id for g, _ in got] == [good.id]
    assert await retrieve(session, llm, a, d1, [], "unrelated zebra", 0.5) == []


async def test_retrieve_orders_best_first_and_limits(session):
    llm = FakeLLM()
    (doc, a, b), (d1, d2) = await seed(session)
    far = await add(session, llm, doctor_id=doc.id, drug_tags=["colchicine"],
                    question="colchicine diarrhea water", answer="more words here")
    near = await add(session, llm, doctor_id=doc.id, drug_tags=["colchicine"],
                     question="colchicine diarrhea", answer="")
    got = await retrieve(session, llm, a, d1, [], "colchicine diarrhea", 0.0)
    assert [g.id for g, _ in got] == [near.id, far.id]
    assert got[0][1] >= got[1][1]
    assert len(await retrieve(session, llm, a, d1, [], "colchicine diarrhea", 0.0, limit=1)) == 1


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
    assert await retrieve(session, llm, a, d1, [], "coffee", 0.0) == []


async def test_retrieve_skips_mismatched_vector_length(session):
    llm = FakeLLM()
    (doc, a, b), (d1, d2) = await seed(session)
    e = await add(session, llm, doctor_id=doc.id, scope="patient", patient_id=a.id,
                  document_id=d1.id, question="coffee", answer="coffee")
    e.embedding = [1.0, 0.0]
    await session.flush()
    assert await retrieve(session, llm, a, d1, [], "coffee", 0.0) == []


class FailingLLM(FakeLLM):
    async def embed(self, texts):
        raise LLMError("down")


async def test_embed_entry_failure_returns_false():
    e = entry()
    assert await embed_entry(FailingLLM(), e) is False
    assert e.embedding is None
    assert await embed_entry(FakeLLM(), e) is True
    assert e.embedding and e.embedding_model == FakeLLM().embedding_model
