"""Doctor review of FDA drug labels: approving one adds it to the education library."""
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from auth import current_doctor
from db import get_session
from doctors import doctor_labels
from druglabels import fetch_label, label_record, resolve_document_drug
from models import DischargeDocument, DocumentDrug, DrugLabel, ExtractedItem, User

logger = logging.getLogger(__name__)

router = APIRouter()

CLOSED_LABEL_DAYS = 30
STORE_UNAVAILABLE = "The education library is not available right now."
ADD_FAILED = "Could not add the label to the library. Try again."


class CandidateBody(BaseModel):
    setid: str


class ReviewBody(BaseModel):
    """The setid of the label text the doctor read, so a review never applies to
    text that replaced it in the meantime (another doctor picked another candidate)."""
    setid: str


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def _medication_rows(session: AsyncSession, conditions: list) -> list:
    """(DocumentDrug, medication line, username) for the rows matching `conditions`."""
    return list(await session.execute(
        select(DocumentDrug, ExtractedItem.text, User.username)
        .join(ExtractedItem, ExtractedItem.id == DocumentDrug.extracted_item_id)
        .join(DischargeDocument, DischargeDocument.id == DocumentDrug.document_id)
        .join(User, User.id == DischargeDocument.user_id)
        .where(*conditions)
        .order_by(DocumentDrug.id)
        .execution_options(populate_existing=True)  # jobs update rows from other sessions
    ))


async def label_views(session: AsyncSession, labels: list[DrugLabel]) -> list[dict]:
    """Views of the labels, loading sources and reviewer names in bulk."""
    if not labels:
        return []
    rxcuis = {label.rxcui for label in labels}
    sources: dict[str, list[dict]] = {rxcui: [] for rxcui in rxcuis}
    for drug, medication, username in await _medication_rows(session, [DocumentDrug.rxcui.in_(rxcuis)]):
        entry = {"medication": medication, "username": username}
        if entry not in sources[drug.rxcui]:
            sources[drug.rxcui].append(entry)
    reviewers = await doctor_labels(session, [label.reviewed_by for label in labels])
    return [
        {
            "id": label.id,
            "rxcui": label.rxcui,
            "ingredient": label.ingredient,
            "status": label.status,
            "setid": label.setid,
            "title": label.title,
            "url": label.url,
            "labeler": next(
                (c.get("labeler") for c in label.candidates or [] if c.get("setid") == label.setid), None),
            "published": next(
                (c.get("published") for c in label.candidates or [] if c.get("setid") == label.setid), None),
            "text": label.text,
            "error": label.error,
            "candidates": label.candidates or [],
            "sources": sources[label.rxcui],
            "reviewed_by_name": reviewers.get(label.reviewed_by),
            "reviewed_at": label.reviewed_at.isoformat() if label.reviewed_at else None,
        }
        for label in labels
    ]


async def _fresh_view(session: AsyncSession, label_id: int) -> dict:
    label = await session.get(DrugLabel, label_id, populate_existing=True)
    return (await label_views(session, [label]))[0]


async def _get_label(session: AsyncSession, label_id: int) -> DrugLabel:
    label = await session.get(DrugLabel, label_id)
    if label is None:
        raise HTTPException(status_code=404, detail="Label not found")
    return label


async def _claim(session: AsyncSession, model, row_id: int, allowed, detail: str, extra=None) -> None:
    """Move the row to `fetching` only if it is still in one of `allowed`; else 409."""
    condition = model.status.in_(allowed) if isinstance(allowed, (list, tuple, set)) else model.status == allowed
    result = await session.execute(
        update(model).where(model.id == row_id, condition)
        .values(status="fetching", error=None)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        await session.rollback()
        raise HTTPException(status_code=409, detail=detail)
    await session.commit()


@router.get("/api/doctor/labels")
async def list_labels(doctor: User = Depends(current_doctor), session: AsyncSession = Depends(get_session)):
    cutoff = utcnow() - timedelta(days=CLOSED_LABEL_DAYS)
    labels = list(await session.scalars(
        select(DrugLabel)
        .where(or_(DrugLabel.status.notin_(("approved", "rejected")),
                   DrugLabel.reviewed_at.is_(None), DrugLabel.reviewed_at >= cutoff))
        .order_by(DrugLabel.id)
        .execution_options(populate_existing=True)
    ))
    labels.sort(key=lambda label: label.status != "pending")  # stable: pending first
    unresolved = [
        {"id": drug.id, "medication": medication, "username": username,
         "status": drug.status, "error": drug.error}
        for drug, medication, username in await _medication_rows(
            session, [DocumentDrug.status.in_(("unresolved", "failed"))])
    ]
    return {"labels": await label_views(session, labels), "unresolved": unresolved}


NOT_WAITING = "This label is not waiting for review."
CHANGED = "This label changed. Review it again."


async def _review_conflict(session: AsyncSession, label_id: int, setid: str) -> HTTPException:
    """The 409 for a review whose conditional update matched no row."""
    await session.rollback()
    label = await session.get(DrugLabel, label_id, populate_existing=True)
    changed = label is not None and label.status == "pending" and label.setid != setid
    return HTTPException(status_code=409, detail=CHANGED if changed else NOT_WAITING)


@router.post("/api/doctor/labels/{label_id}/approve")
async def approve_label(
    label_id: int,
    body: ReviewBody,
    request: Request,
    doctor: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    store = request.app.state.store
    label = await _get_label(session, label_id)
    if store is None:
        raise HTTPException(status_code=503, detail=STORE_UNAVAILABLE)
    doctor_id = doctor.id
    # Approve first, conditionally, so a concurrent reject loses before any text
    # reaches the library. A failed add is undone below.
    result = await session.execute(
        update(DrugLabel)
        .where(DrugLabel.id == label_id, DrugLabel.status == "pending", DrugLabel.setid == body.setid)
        .values(status="approved", reviewed_by=doctor_id, reviewed_at=utcnow())
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        raise await _review_conflict(session, label_id, body.setid)
    await session.commit()
    await session.refresh(label)
    try:
        await store.add(label_record(label))
    except Exception:
        logger.exception("Adding drug label %s to the library failed", label_id)
        await session.execute(
            update(DrugLabel).where(DrugLabel.id == label_id, DrugLabel.status == "approved")
            .values(status="pending", reviewed_by=None, reviewed_at=None)
            .execution_options(synchronize_session=False)
        )
        await session.commit()
        raise HTTPException(status_code=502, detail=ADD_FAILED)
    return await _fresh_view(session, label_id)


@router.post("/api/doctor/labels/{label_id}/reject")
async def reject_label(
    label_id: int,
    body: ReviewBody,
    doctor: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    await _get_label(session, label_id)
    result = await session.execute(
        update(DrugLabel)
        .where(DrugLabel.id == label_id, DrugLabel.status == "pending", DrugLabel.setid == body.setid)
        .values(status="rejected", reviewed_by=doctor.id, reviewed_at=utcnow())
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        raise await _review_conflict(session, label_id, body.setid)
    await session.commit()
    return await _fresh_view(session, label_id)


@router.post("/api/doctor/labels/{label_id}/candidate")
async def pick_candidate(
    label_id: int,
    body: CandidateBody,
    request: Request,
    doctor: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    label = await _get_label(session, label_id)
    # The setid is user input: it must be one the row offered, before any outside call.
    if body.setid not in {c.get("setid") for c in label.candidates or []}:
        raise HTTPException(status_code=422, detail="Choose one of the listed labels.")
    await _claim(session, DrugLabel, label_id, ("pending", "rejected", "no_label", "failed"),
                 "This label cannot be changed right now.")
    await fetch_label(request.app.state.session_factory, request.app.state.label_client, label_id, body.setid)
    return await _fresh_view(session, label_id)


@router.post("/api/doctor/labels/{label_id}/retry")
async def retry_label(
    label_id: int,
    request: Request,
    doctor: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    label = await _get_label(session, label_id)
    chosen = label.setid  # the label last chosen, else (None) the newest candidate
    await _claim(session, DrugLabel, label_id, ("failed", "no_label"), "Only a failed lookup can be retried.")
    await fetch_label(request.app.state.session_factory, request.app.state.label_client, label_id, chosen)
    return await _fresh_view(session, label_id)


@router.post("/api/doctor/document-drugs/{drug_id}/retry")
async def retry_document_drug(
    drug_id: int,
    request: Request,
    doctor: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    if await session.scalar(select(DocumentDrug.id).where(DocumentDrug.id == drug_id)) is None:
        raise HTTPException(status_code=404, detail="Medication not found")
    await _claim(session, DocumentDrug, drug_id, ("failed", "unresolved"), "Only a failed lookup can be retried.")
    await resolve_document_drug(request.app.state.session_factory, request.app.state.label_client, drug_id)
    # The row may have been renamed into several ingredient rows or deleted; answer with the list.
    return await list_labels(doctor, session)
