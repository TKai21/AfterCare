import sys
from pathlib import Path

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

    assert counts == {"users": 3, "docs": 2}
    assert users["dr.ting"].role == "doctor"
    assert users["dr.ting"].display_name == "Dr. Ting Kai"
    assert docs[users["demo"].id].status == "pending_review"
    assert docs[users["demo2"].id].status == "confirmed"

    general = [g for g in guidance if g.scope == "general"]
    patient = [g for g in guidance if g.scope == "patient"]
    assert len(general) == 1 and len(patient) == 1
    (g,) = general
    assert g.drug_tags == ["colchicine / Colcrys"] and g.condition_tags == ["gout"]
    assert "0.6 mg" in g.answer and g.embedding is not None and g.active
    assert patient[0].patient_id == users["demo2"].id
    assert patient[0].document_id == docs[users["demo2"].id].id
    assert patient[0].embedding is not None

    by_user = {u: [q for q in questions if q.user_id == users[u].id] for u in ("demo", "demo2")}
    (answered,) = by_user["demo2"]
    assert answered.status == "answered" and answered.answered_by == users["dr.ting"].id
    assert patient[0].source_question_id == answered.id
    (open_q,) = by_user["demo"]
    assert open_q.status == "open" and open_q.answer is None
    assert len(questions) == 2


async def test_seeded_general_entry_reaches_only_the_gout_patient(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'seed.db'}"
    await seed_demo.run_seed(url, FakeLLM(), confirm=True)
    users, guidance, _, docs, _, demo2_items = await snapshot(url)
    (general,) = [g for g in guidance if g.scope == "general"]
    assert eligible(general, patient_text(docs[users["demo2"].id], demo2_items))
    assert not eligible(general, docs[users["demo"].id].text)
    assert docs[users["demo"].id].status == "confirmed"


async def test_seed_without_a_model_saves_unsearchable_entries(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'seed.db'}"
    await seed_demo.run_seed(url, None, confirm=False)
    _, guidance, _, _, _, _ = await snapshot(url)
    assert len(guidance) == 2 and all(g.embedding is None for g in guidance)


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
    assert len(guidance) == 3
