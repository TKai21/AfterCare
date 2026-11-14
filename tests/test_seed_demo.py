import sys
from pathlib import Path

import httpx
from sqlalchemy import func, select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import seed_demo  # noqa: E402
from db import make_engine, make_session_factory  # noqa: E402
from fake_llm import FakeLLM  # noqa: E402
from guidance import eligible, patient_text  # noqa: E402
from models import (  # noqa: E402
    CareTeamQuestion, DischargeDocument, DoctorGuidance, ExtractedItem, User,
)


async def snapshot(url):
    engine = make_engine(url)
    try:
        async with make_session_factory(engine)() as s:
            users = {u.username: u for u in await s.scalars(select(User))}
            guidance = list(await s.scalars(select(DoctorGuidance).order_by(DoctorGuidance.id)))
            questions = list(await s.scalars(select(CareTeamQuestion).order_by(CareTeamQuestion.id)))
            docs = {d.user_id: d for d in await s.scalars(select(DischargeDocument))}
            counts = {
                "users": await s.scalar(select(func.count()).select_from(User)),
                "docs": len(docs),
            }
            demo2_items = list(await s.scalars(select(ExtractedItem).where(
                ExtractedItem.document_id == docs[users["demo2"].id].id)))
            return users, guidance, questions, docs, counts, demo2_items
    finally:
        await engine.dispose()


async def test_seed_is_idempotent_and_creates_the_demo_history(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'seed.db'}"
    llm = FakeLLM()
    await seed_demo.run_seed(url, llm, confirm=False)
    await seed_demo.run_seed(url, llm, confirm=False)
    users, guidance, questions, docs, counts, _ = await snapshot(url)

    assert counts == {"users": 5, "docs": 2}
    for username, name in (("dr.ting", "Dr. Ting Kai"), ("dr.renny", "Dr. Aaron Renny"),
                           ("dr.patel", "Dr. Parin Patel")):
        assert users[username].role == "doctor"
        assert users[username].display_name == name
    assert docs[users["demo"].id].status == "pending_review"
    assert docs[users["demo2"].id].status == "confirmed"

    general = [g for g in guidance if g.scope == "general"]
    patient = [g for g in guidance if g.scope == "patient"]
    assert len(general) == 3 and len(patient) == 1
    (g,) = [g for g in general if g.doctor_id == users["dr.ting"].id]
    assert g.drug_tags == ["colchicine / Colcrys"] and g.condition_tags == ["gout"]
    assert "0.6 mg" in g.answer and g.embedding is not None and g.active
    assert patient[0].patient_id == users["demo2"].id
    assert patient[0].document_id == docs[users["demo2"].id].id
    assert patient[0].embedding is not None

    (answered,) = [q for q in questions if q.status == "answered"]
    assert answered.user_id == users["demo2"].id
    assert answered.answered_by == users["dr.ting"].id
    assert answered.doctor_id == users["dr.ting"].id
    assert answered.source == "patient"  # asked directly, as POST /api/care-team saves it
    assert patient[0].source_question_id == answered.id
    open_qs = [q for q in questions if q.status == "open"]
    assert len(questions) == 4 and len(open_qs) == 3
    # Exactly one open question per doctor, each from a demo patient's own document.
    assert sorted(q.doctor_id for q in open_qs) == sorted(
        users[u].id for u in ("dr.ting", "dr.renny", "dr.patel"))
    for q in open_qs:
        assert q.answer is None
        assert q.document_id == docs[q.user_id].id
    (ting_open,) = [q for q in open_qs if q.doctor_id == users["dr.ting"].id]
    assert ting_open.user_id == users["demo"].id
    # Dr. Renny's open question is about something his own entry does not cover.
    (renny_open,) = [q for q in open_qs if q.doctor_id == users["dr.renny"].id]
    (renny_entry,) = [g for g in general if g.doctor_id == users["dr.renny"].id]
    assert renny_open.question == "Can I drive while the stent is in?"
    assert "driv" not in renny_entry.answer.lower()
    assert "driv" not in docs[users["demo"].id].text.lower()


async def test_every_doctor_has_a_general_entry_eligible_for_a_demo_patient(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'seed.db'}"
    await seed_demo.run_seed(url, FakeLLM(), confirm=True)
    users, guidance, _, docs, _, demo2_items = await snapshot(url)
    demo_doc, demo2_doc = docs[users["demo"].id], docs[users["demo2"].id]
    assert demo_doc.status == "confirmed"
    for username in ("dr.ting", "dr.renny", "dr.patel"):
        entries = [g for g in guidance
                   if g.scope == "general" and g.doctor_id == users[username].id]
        assert entries, username
        assert any(eligible(g, patient_text(d, items)) for g in entries
                   for d, items in ((demo2_doc, demo2_items), (demo_doc, []))), username
    # The colchicine entry still reaches only the gout patient.
    (ting,) = [g for g in guidance if g.scope == "general" and g.doctor_id == users["dr.ting"].id]
    assert eligible(ting, patient_text(demo2_doc, demo2_items))
    assert not eligible(ting, demo_doc.text)


async def test_seed_without_a_model_saves_unsearchable_entries(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'seed.db'}"
    await seed_demo.run_seed(url, None, confirm=False)
    _, guidance, _, _, _, _ = await snapshot(url)
    assert len(guidance) == 4 and all(g.embedding is None for g in guidance)


async def test_reseed_leaves_other_users_and_their_guidance_alone(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'seed.db'}"
    await seed_demo.run_seed(url, FakeLLM(), confirm=False)
    engine = make_engine(url)
    async with make_session_factory(engine)() as s:
        ting = await s.scalar(select(User).where(User.username == "dr.ting"))
        other = User(username="bob", password_hash="x")
        s.add(other)
        await s.flush()
        s.add(DoctorGuidance(doctor_id=ting.id, question="q", answer="a", scope="patient",
                             patient_id=other.id))
        await s.commit()
    await engine.dispose()
    await seed_demo.run_seed(url, FakeLLM(), confirm=False)
    users, guidance, _, _, _, _ = await snapshot(url)
    assert "bob" in users
    assert len(guidance) == 5


async def test_seed_records_demo_drugs_offline(tmp_path, monkeypatch):
    """The seed resolves labels from recorded fixtures: a guard fails any real request."""
    import druglabels
    from models import DocumentDrug, DrugLabel

    real_init = druglabels.LabelClient.__init__

    def guarded(self, transport=None):
        # Only a fixture MockTransport is allowed; anything else would hit the network.
        assert isinstance(transport, httpx.MockTransport)
        real_init(self, transport)

    monkeypatch.setattr(druglabels.LabelClient, "__init__", guarded)
    url = f"sqlite+aiosqlite:///{tmp_path / 'labels.db'}"
    await seed_demo.run_seed(url, FakeLLM(), confirm=True)

    engine = make_engine(url)
    try:
        async with make_session_factory(engine)() as s:
            users = {u.username: u for u in await s.scalars(select(User))}
            docs = {d.user_id: d for d in await s.scalars(select(DischargeDocument))}

            async def drugs(username):
                doc = docs[users[username].id]
                return sorted(d.name.lower() for d in await s.scalars(
                    select(DocumentDrug).where(DocumentDrug.document_id == doc.id,
                                               DocumentDrug.status == "resolved")))

            assert await drugs("demo2") == ["colchicine", "naproxen"]
            assert await drugs("demo") == ["ibuprofen", "tamsulosin"]
            labels = list(await s.scalars(select(DrugLabel)))
            assert {l.ingredient.lower() for l in labels} == {
                "colchicine", "naproxen", "ibuprofen", "tamsulosin"}
            assert {l.status for l in labels} == {"pending"}
    finally:
        await engine.dispose()
