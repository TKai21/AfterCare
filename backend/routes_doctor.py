"""The doctor's side: the question queue, answering, and managing guidance."""
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import case, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from auth import current_doctor
from db import get_session
from doctors import doctor_label, doctor_labels
from documents import paragraphs
from drafts import build_context, draft_view, record_send, stats, write_draft
from druglabels import patient_rxcuis
from extraction import normalize
from guidance import doctor_habits
from llm import LLMError
from models import (
    AnswerDraft,
    CareTeamQuestion,
    ChatMessage,
    DischargeDocument,
    DoctorGuidance,
    GuidanceConflict,
    GuidanceRule,
    User,
)
from closing import doctor_warning_terms
from routes_agent import NOT_ANSWERED, recent_history
from routes_care_team import mergeable, normalize_question, question_view
from routes_documents import load_items
from rules import clear_rules, index_entry, is_suppressed
from safety import match_terms, warning_terms
from voice import voice_views

router = APIRouter()

ANSWERED_LIMIT = 50

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class AnswerBody(BaseModel):
    answer: Text
    general: bool = False
    saved_question: Optional[Text] = None
    drug_tags: list[str] = Field(default_factory=list, max_length=20)
    condition_tags: list[str] = Field(default_factory=list, max_length=20)
    draft_id: Optional[int] = None  # the clone draft the answer started from


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


async def first_message_warnings(session: AsyncSession, doctor_id: int,
                                 rows: list[CareTeamQuestion]) -> dict[int, list[str]]:
    """For each waiting question, the warning terms its original message
    matches: the doctor's live warning habits and the question's document's
    warning terms, as close_blocker reads them. Computed on read, so it always
    reflects the doctor's current habits (such a case can never close)."""
    if not rows:
        return {}
    doctor_terms = await doctor_warning_terms(session, doctor_id)
    document_terms: dict[Optional[int], list[str]] = {}
    found: dict[int, list[str]] = {}
    for q in rows:
        if q.document_id not in document_terms:
            doc = await session.get(DischargeDocument, q.document_id) if q.document_id else None
            document_terms[q.document_id] = (
                warning_terms(await load_items(session, doc)) if doc is not None else [])
        terms = list(dict.fromkeys([*doctor_terms, *document_terms[q.document_id]]))
        found[q.id] = match_terms(q.question, terms)
    return found


@router.get("/api/doctor/questions")
async def list_questions(
    request: Request, user: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    cutoff = utcnow() - timedelta(hours=request.app.state.settings.clarify_timeout_hours)
    # A question the patient left unanswered too long becomes the doctor's.
    timed_out = await session.execute(
        update(CareTeamQuestion)
        .where(CareTeamQuestion.status == "waiting_on_patient",
               CareTeamQuestion.doctor_id == user.id,
               # The clock restarts with each round the clone asks.
               func.coalesce(CareTeamQuestion.last_asked_at, CareTeamQuestion.created_at) < cutoff)
        .values(status="open", reason=NOT_ANSWERED, stopped_by=NOT_ANSWERED)
    )
    if timed_out.rowcount:
        await session.commit()
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
    ))
    # The same question asked again is one card: the newest row, asked N times.
    groups: dict[tuple, list[CareTeamQuestion]] = {}
    for q in open_rows:
        key = (q.user_id, q.document_id, q.doctor_id,
               normalize_question(q.question) if mergeable(q) else q.id)
        groups.setdefault(key, []).append(q)
    asked_counts: dict[int, int] = {}
    recency: dict[int, datetime] = {}
    open_rows = []
    for members in groups.values():
        card = max(members, key=lambda r: r.id)
        asked_counts[card.id] = sum(r.asked_count or 1 for r in members)
        recency[card.id] = max(
            t for r in members for t in (r.created_at, r.last_asked_at) if t is not None)
        open_rows.append(card)
    open_rows.sort(key=lambda q: (recency[q.id], q.id), reverse=True)
    waiting_rows = list(await session.scalars(
        select(CareTeamQuestion)
        .where(CareTeamQuestion.status == "waiting_on_patient",
               CareTeamQuestion.doctor_id == user.id)
        .order_by(CareTeamQuestion.created_at.desc(), CareTeamQuestion.id.desc())
    ))
    answered_rows = list(await session.scalars(
        select(CareTeamQuestion)
        .where(CareTeamQuestion.answer.is_not(None), mine_or_unassigned)
        .order_by(CareTeamQuestion.answered_at.desc(), CareTeamQuestion.id.desc())
        .limit(ANSWERED_LIMIT)
    ))
    rows = open_rows + waiting_rows + answered_rows
    first_warnings = await first_message_warnings(session, user.id, waiting_rows)
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

    drafts = {
        d.question_id: d for d in await session.scalars(
            select(AnswerDraft).where(
                AnswerDraft.question_id.in_([q.id for q in open_rows]),
                AnswerDraft.doctor_id == user.id,
                AnswerDraft.sent_answer.is_(None),
            ).order_by(AnswerDraft.id)
        )
    } if open_rows else {}

    def item(q: CareTeamQuestion) -> dict:
        patient = patients[q.user_id]
        view = question_view(q, names, voices)
        view["asked_count"] = asked_counts.get(q.id, view["asked_count"])
        view["closed_by_clone"] = (
            q.status == "resolved" and q.answer is not None and q.answered_by is None
            and q.asked is not None)
        if q.answer is None and q.status == "open":
            view["draft"] = draft_view(drafts.get(q.id))
        view["patient"] = {"id": patient.id, "username": patient.username}
        view["first_warning"] = first_warnings.get(q.id, [])
        view["context"] = contexts[(q.document_id, q.doctor_id)]
        return view

    return {"open": [item(q) for q in open_rows], "waiting": [item(q) for q in waiting_rows],
            "answered": [item(q) for q in answered_rows]}


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


def habit_view(rule: GuidanceRule, question: Optional[str]) -> dict:
    """One habit as My guidance shows it, with the entry it was read from
    (`guidance_id`, and that entry's `question`) so the doctor can open it."""
    return {"id": rule.id, "habit_type": rule.habit_type, "statement": rule.statement,
            "quote": rule.quote, "warning_terms": list(rule.warning_terms or []),
            "guidance_id": rule.guidance_id, "question": question}


async def rules_and_conflicts(
    session: AsyncSession, entries: list[DoctorGuidance]
) -> tuple[dict[int, list[GuidanceRule]], dict[int, list[GuidanceRule]], dict[int, list[dict]]]:
    """Each entry's advice rules, its habits (suppressed rows are in neither),
    and the open conflicts on either side of its rules, as the doctor sees them
    from that entry. A conflict whose other entry is retired, or either of
    whose rules is suppressed, is not shown: retired advice reaches no patient."""
    ids = [e.id for e in entries]
    rules: dict[int, list[GuidanceRule]] = {}
    habits: dict[int, list[GuidanceRule]] = {}
    if not ids:
        return rules, habits, {}
    for r in await session.scalars(
        select(GuidanceRule)
        .where(GuidanceRule.guidance_id.in_(ids), GuidanceRule.suppressed.is_(False))
        .order_by(GuidanceRule.guidance_id, GuidanceRule.position, GuidanceRule.id)
    ):
        (habits if r.kind == "habit" else rules).setdefault(r.guidance_id, []).append(r)
    rule_ids = [r.id for rs in rules.values() for r in rs]
    if not rule_ids:
        return rules, habits, {}
    found = list(await session.scalars(
        select(GuidanceConflict).where(
            GuidanceConflict.status == "open",
            or_(GuidanceConflict.rule_id.in_(rule_ids),
                GuidanceConflict.other_rule_id.in_(rule_ids)),
        ).order_by(GuidanceConflict.id)
    ))
    side_ids = {c.rule_id for c in found} | {c.other_rule_id for c in found if c.other_rule_id}
    side = {
        r.id: (r, e) for r, e in (await session.execute(
            select(GuidanceRule, DoctorGuidance)
            .join(DoctorGuidance, DoctorGuidance.id == GuidanceRule.guidance_id)
            .where(GuidanceRule.id.in_(side_ids), GuidanceRule.suppressed.is_(False))
        )).all()
    } if side_ids else {}
    mine = set(ids)
    out: dict[int, list[dict]] = {}
    for c in found:
        pairs = [(c.rule_id, c.other_rule_id)]
        if c.other_rule_id is not None:
            pairs.append((c.other_rule_id, c.rule_id))
        for own_id, other_id in pairs:
            if own_id not in side:
                continue
            own_rule, own_entry = side[own_id]
            if own_entry.id not in mine:
                continue
            if other_id is None:
                other = {"kind": "label", "title": c.label_title, "url": c.label_url,
                         "quote": c.label_quote}
            else:
                if other_id not in side:
                    continue
                other_rule, other_entry = side[other_id]
                if not other_entry.active:
                    continue
                other = {"kind": "advice", "guidance_id": other_entry.id,
                         "question": other_entry.question, "statement": other_rule.statement,
                         "quote": other_rule.quote}
            out.setdefault(own_entry.id, []).append({
                "id": c.id, "explanation": c.explanation,
                "rule": {"statement": own_rule.statement, "quote": own_rule.quote},
                "other": other,
            })
    return rules, habits, out


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
    rules, habits, conflicts = await rules_and_conflicts(session, entries)
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
            "rules_status": e.rules_status,
            "rules": [{"id": r.id, "statement": r.statement, "quote": r.quote}
                      for r in rules.get(e.id, [])],
            "habits": [habit_view(h, e.question) for h in habits.get(e.id, [])],
            "conflicts": conflicts.get(e.id, []),
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
    # The same question asked again is answered with this card.
    ids = [q.id]
    if mergeable(q):
        twins = await session.scalars(
            select(CareTeamQuestion).where(
                CareTeamQuestion.id != q.id,
                CareTeamQuestion.user_id == q.user_id,
                CareTeamQuestion.status == "open",
                CareTeamQuestion.answer.is_(None),
                CareTeamQuestion.source != "patient_voice",
                CareTeamQuestion.document_id == q.document_id
                if q.document_id is not None else CareTeamQuestion.document_id.is_(None),
                CareTeamQuestion.doctor_id == q.doctor_id
                if q.doctor_id is not None else CareTeamQuestion.doctor_id.is_(None),
            )
        )
        wanted = normalize_question(q.question)
        ids += [t.id for t in twins if normalize_question(t.question) == wanted]
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
    if len(ids) > 1:
        await session.execute(
            update(CareTeamQuestion)
            .where(CareTeamQuestion.id.in_(ids[1:]), CareTeamQuestion.answer.is_(None))
            .values(
                answer=body.answer,
                answered_by=doctor.id,
                doctor_id=func.coalesce(CareTeamQuestion.doctor_id, doctor.id),
                answered_at=utcnow(),
                status="answered",
            )
            .execution_options(synchronize_session=False)
        )
    if body.draft_id is not None:
        draft = await session.get(AnswerDraft, body.draft_id)
        if (draft is not None and draft.question_id == q.id and draft.doctor_id == doctor.id
                and draft.sent_answer is None):
            record_send(draft, body.answer, utcnow())
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
        await index_entry(session, llm, request.app.state.store, entry)
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
    questions = {e.id: e.question for e in entries}
    habits_by_type: dict[str, list[dict]] = {"ask": [], "close": [], "warning": []}
    for habit in await doctor_habits(session, doctor.id):
        if habit.habit_type in habits_by_type:
            habits_by_type[habit.habit_type].append(
                habit_view(habit, questions.get(habit.guidance_id)))
    return {
        "entries": await guidance_views(session, _llm(request), entries),
        "habits_by_type": habits_by_type,
    }


@router.delete("/api/doctor/guidance/rules/{rule_id}", status_code=204)
async def suppress_habit(
    rule_id: int,
    doctor: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    """Remove a habit the clone learned. The row is kept `suppressed` so a
    re-read of the answer does not bring it back. Advice is changed by editing
    the answer, not here."""
    rule = await session.get(GuidanceRule, rule_id)
    if rule is None or rule.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="Habit not found")
    if rule.kind != "habit":
        raise HTTPException(
            status_code=422, detail="Advice is changed by editing the answer, not deleted here."
        )
    # The same habit written in several answers is shown once, so it goes
    # everywhere: every live habit of the same type with the same normalised
    # statement or quote, the keys build_rules keeps out of any later read.
    muted = {(rule.habit_type, normalize(rule.statement)), (rule.habit_type, normalize(rule.quote))}
    for twin in await session.scalars(select(GuidanceRule).where(
        GuidanceRule.doctor_id == doctor.id, GuidanceRule.kind == "habit",
        GuidanceRule.suppressed.is_(False),
    )):
        if twin.id == rule.id or is_suppressed(twin.habit_type, twin.statement, twin.quote, muted):
            twin.suppressed = True
    await session.commit()
    return Response(status_code=204)


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
        # The old vector and rules describe the old text: never search them.
        entry.embedding = None
        entry.embedding_model = None
        entry.rules_status = None
        await clear_rules(session, entry.id)
    await session.commit()
    if text_changed and llm is not None:
        await index_entry(session, llm, request.app.state.store, entry)
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
    await index_entry(session, llm, request.app.state.store, entry)
    await session.refresh(entry)
    return await guidance_view(session, llm, entry)


@router.post("/api/doctor/conflicts/{conflict_id}/dismiss")
async def dismiss_conflict(
    conflict_id: int,
    doctor: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    conflict = await session.get(GuidanceConflict, conflict_id)
    if conflict is None or conflict.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="Conflict not found")
    conflict.status = "dismissed"
    await session.commit()
    return {"id": conflict.id, "status": conflict.status}


@router.post("/api/doctor/questions/{question_id}/draft")
async def draft_answer(
    question_id: int,
    request: Request,
    doctor: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    llm = _llm(request)
    q = await session.get(CareTeamQuestion, question_id)
    if q is None or q.doctor_id not in (None, doctor.id):
        raise HTTPException(status_code=404, detail="Question not found")
    if q.answer is not None:
        raise HTTPException(status_code=409, detail="This question already has an answer.")
    if llm is None:
        raise HTTPException(status_code=503, detail="No model is configured.")
    name = doctor_label(doctor)
    try:
        ctx = await build_context(
            session, llm, request.app.state.store, q, doctor, name,
            request.app.state.settings,
            await patient_rxcuis(session, q.document_id),
        )
        draft = await write_draft(session, llm, q, doctor, name, ctx)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail="The model service failed.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="The model returned an unusable draft.") from exc
    await session.commit()
    await session.refresh(draft)
    return draft_view(draft)


@router.get("/api/doctor/draft-stats")
async def draft_stats(
    doctor: User = Depends(current_doctor), session: AsyncSession = Depends(get_session)
):
    return await stats(session, doctor.id)
