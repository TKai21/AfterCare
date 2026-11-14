"""The doctor's side: the question queue, answering, and managing guidance."""
import json
import re
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import case, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from auth import current_doctor
from db import get_session
from doctors import doctor_labels
from documents import paragraphs
from extraction import normalize
from guidance import embed_entry
from llm import LLMError
from models import (
    CareTeamQuestion,
    ChatMessage,
    DischargeDocument,
    DoctorGuidance,
    User,
)
from routes_agent import recent_history
from routes_care_team import question_view
from routes_documents import load_items
from voice import voice_views

router = APIRouter()

ANSWERED_LIMIT = 50
GENERAL_TAG_ERROR = "General advice needs at least one drug or condition tag."

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class AnswerBody(BaseModel):
    answer: Text
    general: bool = False
    saved_question: Optional[Text] = None
    drug_tags: list[str] = Field(default_factory=list, max_length=20)
    condition_tags: list[str] = Field(default_factory=list, max_length=20)


class SuggestBody(BaseModel):
    question: Annotated[str, StringConstraints(max_length=4000)]
    answer: Annotated[str, StringConstraints(max_length=4000)]


class GuidancePatch(BaseModel):
    question: Optional[Text] = None
    answer: Optional[Text] = None
    scope: Optional[Literal["general", "patient"]] = None
    drug_tags: Optional[list[str]] = Field(default=None, max_length=20)
    condition_tags: Optional[list[str]] = Field(default=None, max_length=20)
    active: Optional[bool] = None


def clean_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    for tag in tags:
        tag = tag.strip()
        if tag and tag not in result:
            result.append(tag)
    return result


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _llm(request: Request):
    return request.app.state.llm


async def doctor_names(session: AsyncSession, questions) -> dict[int, str]:
    return await doctor_labels(
        session, {i for q in questions for i in (q.answered_by, q.doctor_id)}
    )


async def document_context(
    session: AsyncSession, document_id: int | None, doctor_id: int | None
) -> dict:
    context: dict = {"document_paragraphs": None, "confirmed_items": [], "conversation": []}
    doc = await session.get(DischargeDocument, document_id) if document_id else None
    if doc is None:
        return context
    items = await load_items(session, doc)
    return {
        "document_paragraphs": paragraphs(doc.text or ""),
        "confirmed_items": [
            {"category": i.category, "text": i.text} for i in items if i.status != "rejected"
        ],
        "conversation": await recent_history(session, doc.id, doctor_id),
    }


@router.get("/api/doctor/questions")
async def list_questions(
    user: User = Depends(current_doctor), session: AsyncSession = Depends(get_session)
):
    mine_or_unassigned = or_(
        CareTeamQuestion.doctor_id == user.id, CareTeamQuestion.doctor_id.is_(None)
    )
    open_rows = list(await session.scalars(
        select(CareTeamQuestion)
        .where(
            CareTeamQuestion.status == "open",
            CareTeamQuestion.answer.is_(None),
            mine_or_unassigned,
        )
        .order_by(CareTeamQuestion.id)
    ))
    answered_rows = list(await session.scalars(
        select(CareTeamQuestion)
        .where(CareTeamQuestion.answer.is_not(None), mine_or_unassigned)
        .order_by(CareTeamQuestion.answered_at.desc(), CareTeamQuestion.id.desc())
        .limit(ANSWERED_LIMIT)
    ))
    rows = open_rows + answered_rows
    patients = {
        u.id: u
        for u in await session.scalars(select(User).where(User.id.in_({q.user_id for q in rows})))
    } if rows else {}
    names = await doctor_names(session, rows)
    voices = await voice_views(session, [q.id for q in rows])
    # The conversation shown is the one with the question's own doctor's clone.
    contexts: dict[tuple[int | None, int | None], dict] = {}
    for q in rows:
        key = (q.document_id, q.doctor_id)
        if key not in contexts:
            contexts[key] = await document_context(session, q.document_id, q.doctor_id)

    def item(q: CareTeamQuestion) -> dict:
        patient = patients[q.user_id]
        view = question_view(q, names, voices)
        view["patient"] = {"id": patient.id, "username": patient.username}
        view["context"] = contexts[(q.document_id, q.doctor_id)]
        return view

    return {"open": [item(q) for q in open_rows], "answered": [item(q) for q in answered_rows]}


async def cited_counts(session: AsyncSession, doctor_ids: set[int]) -> dict[tuple, int]:
    """How many clone answers cited each entry, keyed by (doctor id, entry id).
    Only the entry's own doctor's clone conversations count."""
    counts: dict[tuple, int] = {}
    if not doctor_ids:
        return counts
    rows = await session.execute(
        select(ChatMessage.doctor_id, ChatMessage.payload).where(
            ChatMessage.role == "assistant", ChatMessage.doctor_id.in_(doctor_ids)
        )
    )
    for doctor_id, payload in rows:
        ids = payload.get("guidance_ids") if isinstance(payload, dict) else None
        if isinstance(ids, list):
            for gid in set(i for i in ids if isinstance(i, int)):
                counts[(doctor_id, gid)] = counts.get((doctor_id, gid), 0) + 1
    return counts


async def guidance_views(
    session: AsyncSession, llm, entries: list[DoctorGuidance]
) -> list[dict]:
    patient_ids = {e.patient_id for e in entries if e.patient_id}
    usernames: dict[int, str] = {}
    active_docs: dict[int, int] = {}
    if patient_ids:
        usernames = {
            u.id: u.username
            for u in await session.scalars(select(User).where(User.id.in_(patient_ids)))
        }
        active_docs = {
            uid: did
            for uid, did in (await session.execute(
                select(DischargeDocument.active_user_id, DischargeDocument.id).where(
                    DischargeDocument.active_user_id.in_(patient_ids)
                )
            )).all()
        }
    cited = await cited_counts(session, {e.doctor_id for e in entries})
    model = getattr(llm, "embedding_model", None) if llm is not None else None
    return [
        {
            "id": e.id,
            "question": e.question,
            "answer": e.answer,
            "scope": e.scope,
            "patient": usernames.get(e.patient_id) if e.patient_id else None,
            "drug_tags": e.drug_tags or [],
            "condition_tags": e.condition_tags or [],
            "active": bool(e.active),
            "searchable": (
                llm is not None and e.embedding is not None and e.embedding_model == model
            ),
            "stale": (
                e.scope == "patient"
                and e.patient_id is not None
                and (e.document_id is None or e.document_id != active_docs.get(e.patient_id))
            ),
            "cited_count": cited.get((e.doctor_id, e.id), 0),
            "created_at": e.created_at.isoformat() if e.created_at else None,
            "updated_at": e.updated_at.isoformat() if e.updated_at else None,
        }
        for e in entries
    ]


async def guidance_view(session: AsyncSession, llm, entry: DoctorGuidance) -> dict:
    return (await guidance_views(session, llm, [entry]))[0]


@router.post("/api/doctor/questions/{question_id}/answer")
async def answer_question(
    question_id: int,
    body: AnswerBody,
    request: Request,
    doctor: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    llm = _llm(request)
    q = await session.get(CareTeamQuestion, question_id)
    if q is None or q.doctor_id not in (None, doctor.id):
        raise HTTPException(status_code=404, detail="Question not found")
    if q.source == "patient_voice":
        raise HTTPException(status_code=409, detail="Reply to this voice question with a voice clip.")
    drug_tags, condition_tags = clean_tags(body.drug_tags), clean_tags(body.condition_tags)
    if body.general and not drug_tags and not condition_tags:
        raise HTTPException(status_code=422, detail=GENERAL_TAG_ERROR)
    # Only one concurrent answer can match "answer IS NULL" and win.
    result = await session.execute(
        update(CareTeamQuestion)
        .where(CareTeamQuestion.id == question_id, CareTeamQuestion.answer.is_(None))
        .values(
            answer=body.answer,
            answered_by=doctor.id,
            doctor_id=func.coalesce(CareTeamQuestion.doctor_id, doctor.id),
            answered_at=utcnow(),
            status=case((CareTeamQuestion.status == "resolved", "resolved"), else_="answered"),
        )
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        await session.rollback()
        raise HTTPException(status_code=409, detail="This question already has an answer.")
    entry = DoctorGuidance(
        doctor_id=doctor.id,
        question=(body.saved_question or q.question) if body.general else q.question,
        answer=body.answer,
        scope="general" if body.general else "patient",
        patient_id=None if body.general else q.user_id,
        document_id=None if body.general else q.document_id,
        source_question_id=q.id,
        drug_tags=drug_tags if body.general else [],
        condition_tags=condition_tags if body.general else [],
    )
    session.add(entry)
    await session.commit()
    if llm is not None:
        await embed_entry(llm, entry)
        await session.commit()
    await session.refresh(q)
    await session.refresh(entry)
    return {
        "question": question_view(q, await doctor_names(session, [q])),
        "guidance": await guidance_view(session, llm, entry),
    }


SUGGEST_PROMPT = """You help a doctor tag a piece of medical advice so it is only reused for patients it applies to.
Read the question and the doctor's answer. List the drugs and the medical conditions they are about.
Rules:
- Use full drug names and full condition names exactly as written in the text (for example "metoprolol", "atrial fibrillation"), never generic word fragments such as "blocker".
- Only list names that appear in the text. Do not add related names that are not written there.
- Reply with JSON only: {"drug_tags": [...], "condition_tags": [...]}. Use empty lists when there are none."""


def _occurs(name: str, text: str) -> bool:
    needle = normalize(name)
    return bool(needle) and re.search(
        r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", text
    ) is not None


def _filter_names(raw, text: str) -> list[str]:
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for name in raw:
        if isinstance(name, str) and _occurs(name, text) and normalize(name) not in seen:
            seen.add(normalize(name))
            result.append(name.strip())
    return result


@router.post("/api/doctor/suggest-tags")
async def suggest_tags(
    body: SuggestBody, request: Request, doctor: User = Depends(current_doctor)
):
    llm = _llm(request)
    if llm is None:
        raise HTTPException(status_code=503, detail="No model is configured.")
    messages = [
        {"role": "system", "content": SUGGEST_PROMPT},
        {"role": "user", "content": f"Question:\n{body.question}\n\nAnswer:\n{body.answer}"},
    ]
    try:
        message = await llm.chat(messages, json_mode=True)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail="The model service failed.") from exc
    try:
        data = json.loads(message.get("content") or "")
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="The model returned an unreadable reply.") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="The model returned an unreadable reply.")
    text = normalize(body.question + "\n" + body.answer)
    return {
        "drug_tags": _filter_names(data.get("drug_tags"), text),
        "condition_tags": _filter_names(data.get("condition_tags"), text),
    }


@router.get("/api/doctor/guidance")
async def list_guidance(
    request: Request,
    doctor: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    entries = list(await session.scalars(
        select(DoctorGuidance)
        .where(DoctorGuidance.doctor_id == doctor.id)
        .order_by(DoctorGuidance.id.desc())
    ))
    return {"entries": await guidance_views(session, _llm(request), entries)}


async def get_entry(
    session: AsyncSession, guidance_id: int, doctor: User
) -> DoctorGuidance:
    entry = await session.get(DoctorGuidance, guidance_id)
    if entry is None or entry.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="Guidance not found")
    return entry


@router.patch("/api/doctor/guidance/{guidance_id}")
async def patch_guidance(
    guidance_id: int,
    body: GuidancePatch,
    request: Request,
    doctor: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    llm = _llm(request)
    entry = await get_entry(session, guidance_id, doctor)
    sent = body.model_fields_set
    scope = body.scope if "scope" in sent and body.scope else entry.scope
    drug_tags = clean_tags(body.drug_tags) if body.drug_tags is not None else list(entry.drug_tags or [])
    condition_tags = (
        clean_tags(body.condition_tags) if body.condition_tags is not None
        else list(entry.condition_tags or [])
    )
    if scope == "general" and not drug_tags and not condition_tags:
        raise HTTPException(status_code=422, detail=GENERAL_TAG_ERROR)
    if scope == "patient" and entry.patient_id is None:
        raise HTTPException(
            status_code=422, detail="This advice was not written for one patient."
        )
    text_changed = (
        (body.question is not None and body.question != entry.question)
        or (body.answer is not None and body.answer != entry.answer)
    )
    if body.question is not None:
        entry.question = body.question
    if body.answer is not None:
        entry.answer = body.answer
    entry.scope = scope
    entry.drug_tags = drug_tags
    entry.condition_tags = condition_tags
    if body.active is not None:
        entry.active = body.active
    if text_changed:
        entry.embedding = None
        entry.embedding_model = None
    await session.commit()
    if text_changed and llm is not None:
        await embed_entry(llm, entry)
        await session.commit()
    await session.refresh(entry)
    return await guidance_view(session, llm, entry)


@router.post("/api/doctor/guidance/{guidance_id}/embed")
async def embed_guidance(
    guidance_id: int,
    request: Request,
    doctor: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    llm = _llm(request)
    if llm is None:
        raise HTTPException(status_code=503, detail="No model is configured.")
    entry = await get_entry(session, guidance_id, doctor)
    await embed_entry(llm, entry)
    await session.commit()
    await session.refresh(entry)
    return await guidance_view(session, llm, entry)
