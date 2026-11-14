"""Dev-only demo seed. Never imported by the app (it lives outside backend/).

Creates, or resets, the user `demo` / `demo-password-1` in the database named by
DATABASE_URL (from the project .env), with one PENDING fictional kidney-stones
discharge document and its extracted items, so the review screen can be seen
without a language model. It also seeds the doctor clone demo: the three
doctors (`dr.ting`, `dr.renny`, `dr.patel`, all `doctor-password-1`), the patient
`demo2` / `demo-password-2` (gout case, always confirmed), and a small guidance
history (see seed_guidance). Every source_quote is a verbatim substring of one
document paragraph; the script checks this before writing anything.

    .venv/bin/python scripts/seed_demo.py            # pending review
    .venv/bin/python scripts/seed_demo.py --confirm  # confirmed, checklist generated

--confirm runs the confirm route's own function, so the checklist is generated
exactly as when the patient presses Confirm. Resetting deletes everything the
demo users own (documents, checklist, conversation, care-team questions) and the
guidance tied to them, and leaves every other user alone.
"""
import argparse
import asyncio
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "demo"))

from sqlalchemy import delete, or_, select  # noqa: E402

from auth import hash_password  # noqa: E402
from config import get_settings  # noqa: E402
from guidance import embed_entry  # noqa: E402
from llm import LLMClient  # noqa: E402
from db import init_db, make_engine, make_session_factory  # noqa: E402
from documents import paragraphs  # noqa: E402
from models import (  # noqa: E402
    CareTeamQuestion, ChatMessage, ChecklistCompletion, ChecklistItem,
    DischargeDocument, DoctorGuidance, DocumentFile, ExtractedItem, User,
)
import routes_documents  # noqa: E402
from generate_demo import load_case, long_date, render_pdf, render_text  # noqa: E402

USERNAME = "demo"
PASSWORD = "demo-password-1"
USERNAME2 = "demo2"
PASSWORD2 = "demo-password-2"
DOCTOR = "dr.ting"
DOCTOR_PASSWORD = "doctor-password-1"
DOCTOR_NAME = "Dr. Ting Kai"
DOCTORS = [  # (username, display name); all share DOCTOR_PASSWORD
    (DOCTOR, DOCTOR_NAME),
    ("dr.renny", "Dr. Aaron Renny"),
    ("dr.patel", "Dr. Parin Patel"),
]

# Everything below is INVENTED for the demo. Dr. Ting Kai is fictional and none of
# this is medical advice; the numbers are plausible but made up.
GENERAL_QUESTION = "During a gout flare, when can I take more colchicine if my toe still hurts?"
GENERAL_ANSWER = (
    "If your toe is still very painful after your colchicine dose, you may take one extra "
    "0.6 mg tablet after 1 hour, then no more that day. Do not take more than that, and "
    "stop and call the clinic if you get diarrhea."
)
GENERAL_DRUG_TAGS = ["colchicine / Colcrys"]
GENERAL_CONDITION_TAGS = ["gout"]
PATIENT_QUESTION = "Can I take my naproxen on an empty stomach in the morning?"
PATIENT_ANSWER = (
    "No - take your naproxen with breakfast or a snack, every time. On an empty stomach it "
    "can upset your stomach. If you still get stomach pain or black stools, call the clinic."
)
RENNY_QUESTION = "My stent makes me feel like I have to pee all the time. Is there anything that helps?"
RENNY_ANSWER = (
    "Stent discomfort and urgency are common. Take your tamsulosin 0.4 mg every night at "
    "bedtime as prescribed, because it relaxes the ureter. Drink plenty of water, and stay "
    "away from caffeine in the afternoon. Call the clinic if you cannot urinate or you "
    "have a fever of 101 F or higher."
)
RENNY_DRUG_TAGS = ["tamsulosin"]
RENNY_CONDITION_TAGS = ["ureteral stone"]
PATEL_QUESTION = "Do I have to eat before I take naproxen for my gout pain?"
PATEL_ANSWER = (
    "Yes - take naproxen 500 mg with a meal or a snack, and no more than every 12 hours. "
    "Stop and call the clinic if you notice black stools or stomach pain."
)
PATEL_DRUG_TAGS = ["naproxen"]
PATEL_CONDITION_TAGS = ["gout"]
OPEN_QUESTION = "Is it normal to see a little blood in my urine while the stent is in?"
OPEN_REASON = "The patient asked something the discharge document does not answer."
# Neither Dr. Renny's entry nor the kidney-stone document covers driving.
RENNY_OPEN_QUESTION = "Can I drive while the stent is in?"
PATEL_OPEN_QUESTION = "Is it safe to have a beer after my gout flare settles?"


def demo_text(discharge: date) -> str:
    """Same document as demo/out/kidney_stones.txt (one source: demo/cases)."""
    return render_text(load_case("kidney_stones"), discharge)


NO_SCHEDULE = {"times_per_day": None, "times_of_day": [], "duration_days": None}


def build_items(discharge: date) -> list[tuple[str, str, str, dict]]:
    """(category, plain-language text, verbatim source_quote, details)."""
    visit = long_date(discharge + timedelta(days=7))
    return [
        ("medication", "Take tamsulosin 0.4 mg by mouth once a day at bedtime for 14 days.",
         "Tamsulosin 0.4 mg by mouth once daily at bedtime for 14 days.",
         {"name": "Tamsulosin", "dose": "0.4 mg", "route": "by mouth",
          "frequency_text": "once daily at bedtime", "times_of_day": ["bedtime"],
          "times_per_day": 1, "interval_hours": None, "as_needed": False,
          "duration_days": 14, "instructions": "at bedtime"}),
        ("medication", "Take ibuprofen 600 mg with food, up to every 8 hours, only if you need it for pain.",
         "Ibuprofen 600 mg by mouth every 8 hours as needed for pain.",
         {"name": "Ibuprofen", "dose": "600 mg", "route": "by mouth",
          "frequency_text": "every 8 hours as needed", "times_of_day": [],
          "times_per_day": None, "interval_hours": 8, "as_needed": True,
          "duration_days": None, "instructions": "with food"}),
        ("diet", "Drink enough fluids to make at least 2.5 liters of urine a day.",
         "drink enough fluids to make at least 2.5 liters of urine a day", {}),
        ("diet", "Avoid high-oxalate foods such as nuts, spinach, and rhubarb.",
         "avoid high-oxalate foods such as nuts, spinach, and rhubarb", {}),
        ("diet", "Keep salt (sodium) under 2,300 mg a day.",
         "limit sodium to less than 2,300 mg per day", {}),
        ("activity", "Do not lift anything heavier than 10 pounds until your stent is removed.",
         "No heavy lifting over 10 pounds until your stent is removed.", {}),
        ("wound_or_device_care", "Showers are fine. No baths or swimming while the stent is in.",
         "You may shower; no baths or swimming while the stent is in place.", dict(NO_SCHEDULE)),
        ("monitoring", "Strain all of your urine and save any stone pieces for your follow-up visit.",
         "Strain all of your urine through the strainer provided and save any stone pieces",
         dict(NO_SCHEDULE)),
        ("monitoring", "Check your temperature every evening for 7 days.",
         "Check your temperature every evening for 7 days.",
         {"times_per_day": 1, "times_of_day": ["evening"], "duration_days": 7}),
        ("follow_up", f"See Dr. Priya Raman at the Urology Clinic on {visit} to remove the stent.",
         f"See Dr. Priya Raman, Urology Clinic on {visit} for stent removal.",
         {"who": "Dr. Priya Raman, Urology Clinic", "when_text": visit,
          "date": (discharge + timedelta(days=7)).isoformat(), "days_after_discharge": 7}),
        ("follow_up", f"See your primary care provider on {long_date(discharge + timedelta(days=14))} "
         "to talk about a 24-hour urine test.",
         f"See your primary care provider on {long_date(discharge + timedelta(days=14))} "
         "for a discussion of a 24-hour urine test.",
         {"who": "Primary care provider", "when_text": long_date(discharge + timedelta(days=14)),
          "date": (discharge + timedelta(days=14)).isoformat(), "days_after_discharge": 14}),
        ("warning_signs", "Call 911 or go to the ER for a fever of 101 F or higher with chills.",
         "a fever of 101 F (38.3 C) or higher with chills", {}),
        ("warning_signs", "Call 911 or go to the ER if you cannot keep fluids down.",
         "you cannot keep fluids down", {}),
        ("warning_signs", "Call 911 or go to the ER for severe pain that your medicine does not control.",
         "severe pain that your medicine does not control", {}),
        ("warning_signs", "Call 911 or go to the ER for chest pain, trouble breathing, or fainting.",
         "chest pain, trouble breathing, or fainting", {}),
        ("warning_signs", "Call the clinic if you cannot urinate.",
         "you cannot urinate", {}),
        ("warning_signs", "Call the clinic if you see heavy bleeding or clots.",
         "you see heavy bleeding or clots", {}),
    ]


def build_gout_items() -> list[tuple[str, str, str, dict]]:
    """Items for demo2's gout document (quotes are verbatim lines of the gout case)."""
    return [
        ("medication", "Take colchicine 0.6 mg by mouth twice a day for 10 days.",
         "Colchicine 0.6 mg by mouth twice daily for 10 days.",
         {"name": "Colchicine", "dose": "0.6 mg", "route": "by mouth",
          "frequency_text": "twice daily", "times_of_day": [], "times_per_day": 2,
          "interval_hours": None, "as_needed": False, "duration_days": 10,
          "instructions": None}),
        ("medication", "Take naproxen 500 mg with food, up to every 12 hours, only if you need it for pain.",
         "Naproxen 500 mg by mouth every 12 hours as needed for pain.",
         {"name": "Naproxen", "dose": "500 mg", "route": "by mouth",
          "frequency_text": "every 12 hours as needed", "times_of_day": [],
          "times_per_day": None, "interval_hours": 12, "as_needed": True,
          "duration_days": None, "instructions": "with food"}),
        ("diet", "Drink at least 2 liters of water a day.",
         "You should drink at least 2 liters of water a day.", {}),
        ("diet", "Avoid beer and liquor, and limit wine to one glass on any day.",
         "Please avoid beer and liquor, and limit wine to one glass on any day.", {}),
        ("activity", "Rest and raise your right foot when you can for the next 3 days.",
         "Rest and raise your right foot when you can for the next 3 days.", {}),
        ("warning_signs", "Call 911 or go to the ER for chest pain, trouble breathing, or fainting.",
         "chest pain, trouble breathing, or fainting", {}),
        ("warning_signs", "Call the clinic if you have diarrhea, vomiting, or black stools.",
         "you have diarrhea, vomiting, or black stools", {}),
    ]


# Shown in the "could not be verified" list: the document never says this.
UNVERIFIED = [{"category": "diet", "text": "Avoid all dairy products."}]


async def reset_user(session, username: str, password: str) -> User:
    """Delete the named user with everything they own and the guidance tied to
    them, then create them again. Other users are never touched."""
    user = await session.scalar(select(User).where(User.username == username))
    if user is not None:
        # Explicit deletes: SQLite does not enforce ON DELETE CASCADE by default.
        doc_ids = select(DischargeDocument.id).where(DischargeDocument.user_id == user.id)
        item_ids = select(ChecklistItem.id).where(ChecklistItem.document_id.in_(doc_ids))
        question_ids = select(CareTeamQuestion.id).where(CareTeamQuestion.user_id == user.id)
        await session.execute(
            delete(DoctorGuidance).where(or_(
                DoctorGuidance.patient_id == user.id,
                DoctorGuidance.source_question_id.in_(question_ids),
            ))
        )
        await session.execute(
            delete(ChecklistCompletion).where(ChecklistCompletion.checklist_item_id.in_(item_ids))
        )
        await session.execute(delete(ChecklistItem).where(ChecklistItem.document_id.in_(doc_ids)))
        await session.execute(delete(ExtractedItem).where(ExtractedItem.document_id.in_(doc_ids)))
        await session.execute(delete(ChatMessage).where(ChatMessage.document_id.in_(doc_ids)))
        await session.execute(delete(DocumentFile).where(DocumentFile.document_id.in_(doc_ids)))
        await session.execute(delete(CareTeamQuestion).where(CareTeamQuestion.user_id == user.id))
        await session.execute(delete(DischargeDocument).where(DischargeDocument.user_id == user.id))
        await session.delete(user)
        await session.flush()
    user = User(username=username, password_hash=hash_password(password))
    session.add(user)
    await session.flush()
    return user


async def add_document(session, user: User, case: str, filename: str, text: str,
                       discharge: date, items, unverified) -> DischargeDocument:
    doc = DischargeDocument(
        user_id=user.id,
        active_user_id=user.id,
        status="pending_review",
        filename=filename,
        text=text,
        discharge_date=discharge,
        unverified=unverified,
    )
    session.add(doc)
    await session.flush()
    session.add_all(
        ExtractedItem(document_id=doc.id, category=c, text=t, source_quote=q,
                      details=d, status="active")
        for c, t, q, d in items
    )
    # The same document as a PDF, shown beside the instructions like an upload.
    session.add(DocumentFile(document_id=doc.id, content_type="application/pdf",
                             data=render_pdf(load_case(case), discharge)))
    return doc


async def ensure_doctors(session) -> dict[str, User]:
    """The three demo doctors by username, created if missing. Existing ones keep
    their history."""
    doctors = {}
    for username, name in DOCTORS:
        doctor = await session.scalar(select(User).where(User.username == username))
        if doctor is None:
            doctor = User(username=username, password_hash=hash_password(DOCTOR_PASSWORD),
                          role="doctor", display_name=name)
            session.add(doctor)
        else:
            doctor.password_hash = hash_password(DOCTOR_PASSWORD)
            doctor.role = "doctor"
            doctor.display_name = name
        doctors[username] = doctor
    await session.flush()
    return doctors


async def seed_guidance(session, llm, doctors: dict[str, User], demo: User, demo2: User,
                        doc: DischargeDocument, doc2: DischargeDocument) -> bool:
    """The doctors' fictional history. Dr. Ting: one general entry (colchicine timing
    in a gout flare), one patient-specific entry for demo2 from the question demo2
    asked him directly (answered), and one open question from `demo`. Dr. Renny: a
    general kidney-stone entry and an open question from `demo`. Dr. Patel: a general naproxen/gout entry and an open
    question from `demo2`. Returns True when every entry got an embedding."""
    doctor = doctors[DOCTOR]
    renny, patel = doctors["dr.renny"], doctors["dr.patel"]
    general_questions = (GENERAL_QUESTION, RENNY_QUESTION, PATEL_QUESTION)
    await session.execute(delete(DoctorGuidance).where(
        DoctorGuidance.doctor_id.in_([d.id for d in doctors.values()]),
        DoctorGuidance.scope == "general",
        DoctorGuidance.question.in_(general_questions),
    ))
    answered = CareTeamQuestion(
        user_id=demo2.id, document_id=doc2.id, question=PATIENT_QUESTION,
        reason="Asked by the patient", source="patient",
        status="answered", answer=PATIENT_ANSWER, answered_by=doctor.id, doctor_id=doctor.id,
        answered_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    session.add(answered)
    session.add(CareTeamQuestion(
        user_id=demo.id, document_id=doc.id, question=OPEN_QUESTION, reason=OPEN_REASON,
        source="agent", doctor_id=doctor.id,
    ))
    session.add(CareTeamQuestion(
        user_id=demo.id, document_id=doc.id, question=RENNY_OPEN_QUESTION, reason=OPEN_REASON,
        source="agent", doctor_id=renny.id,
    ))
    session.add(CareTeamQuestion(
        user_id=demo2.id, document_id=doc2.id, question=PATEL_OPEN_QUESTION, reason=OPEN_REASON,
        source="agent", doctor_id=patel.id,
    ))
    await session.flush()
    entries = [
        DoctorGuidance(
            doctor_id=doctor.id, question=GENERAL_QUESTION, answer=GENERAL_ANSWER,
            scope="general", drug_tags=GENERAL_DRUG_TAGS, condition_tags=GENERAL_CONDITION_TAGS,
        ),
        DoctorGuidance(
            doctor_id=doctor.id, question=PATIENT_QUESTION, answer=PATIENT_ANSWER,
            scope="patient", patient_id=demo2.id, document_id=doc2.id,
            source_question_id=answered.id,
        ),
        DoctorGuidance(
            doctor_id=renny.id, question=RENNY_QUESTION, answer=RENNY_ANSWER,
            scope="general", drug_tags=RENNY_DRUG_TAGS, condition_tags=RENNY_CONDITION_TAGS,
        ),
        DoctorGuidance(
            doctor_id=patel.id, question=PATEL_QUESTION, answer=PATEL_ANSWER,
            scope="general", drug_tags=PATEL_DRUG_TAGS, condition_tags=PATEL_CONDITION_TAGS,
        ),
    ]
    session.add_all(entries)
    await session.flush()
    embedded = True
    if llm is not None:
        for entry in entries:
            embedded = await embed_entry(llm, entry) and embedded
    else:
        embedded = False
    return embedded


async def run_seed(database_url: str, llm, confirm: bool) -> bool:
    """Seed `database_url`. `llm` computes the guidance embeddings (None leaves the
    entries unsearchable). Returns True when every entry was embedded."""
    discharge = date.today() - timedelta(days=2)
    text = demo_text(discharge)
    gout_text = render_text(load_case("gout"), discharge)
    items, gout_items = build_items(discharge), build_gout_items()
    for doc_text, doc_items in ((text, items), (gout_text, gout_items)):
        paras = paragraphs(doc_text)
        for _, _, quote, _ in doc_items:
            if not any(quote in p for p in paras):
                raise SystemExit(f"Quote is not inside one paragraph: {quote!r}")

    engine = make_engine(database_url)
    try:
        await init_db(engine)
        async with make_session_factory(engine)() as session:
            user = await reset_user(session, USERNAME, PASSWORD)
            doc = await add_document(session, user, "kidney_stones", "kidney_stones_demo.pdf",
                                     text, discharge, items, UNVERIFIED)
            user2 = await reset_user(session, USERNAME2, PASSWORD2)
            doc2 = await add_document(session, user2, "gout", "gout_demo.pdf",
                                      gout_text, discharge, gout_items, [])
            doctors = await ensure_doctors(session)
            embedded = await seed_guidance(session, llm, doctors, user, user2, doc, doc2)
            await session.commit()
            if confirm:
                await routes_documents.confirm(user=user, session=session)
            await routes_documents.confirm(user=user2, session=session)
        state = "confirmed" if confirm else "pending review"
        print(f"Seeded user '{USERNAME}' with a {state} kidney-stones document ({len(items)} items).")
        print(f"Seeded user '{USERNAME2}' with a confirmed gout document ({len(gout_items)} items).")
        print("Seeded doctors " + ", ".join(f"'{u}'" for u, _ in DOCTORS)
              + " with their guidance entries and one open question each.")
        if not embedded:
            print("The guidance entries are saved but not searchable (no embedding model "
                  "answered). Log in at /doctor and use 'Retry embedding' on each entry.")
        return embedded
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the AfterCare demo users (dev only).")
    parser.add_argument("--confirm", action="store_true",
                        help="also confirm the kidney-stones document, generating the checklist")
    confirm = parser.parse_args().confirm
    settings = get_settings()
    llm = LLMClient(settings) if settings.llm_model else None
    asyncio.run(run_seed(settings.database_url, llm, confirm))


if __name__ == "__main__":
    main()
