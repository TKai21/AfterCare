"""Doctor guidance memory: the applicability gate, embedding, and retrieval.

The gate is the safety mechanism: a "general" entry may only reach a patient
whose discharge document or confirmed instructions mention its drugs and
conditions as whole words. Anything unclear fails closed.
"""
import re

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from extraction import normalize
from llm import LLMError
from models import DischargeDocument, DoctorGuidance, ExtractedItem, User


def tag_names(tag: str) -> list[str]:
    """Split a tag such as "colchicine / Colcrys" into its names."""
    return [name.strip() for name in tag.split("/") if name.strip()]


def _mentions(names: list[str], text: str) -> bool:
    for name in names:
        needle = normalize(name)
        if needle and re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", text):
            return True
    return False


def eligible(entry: DoctorGuidance, patient_text: str) -> bool:
    """True only for a tagged general entry whose every drug tag and at least
    one condition tag (when it has any) appear in the patient's text."""
    if entry.scope != "general":
        return False
    drugs = [tag_names(tag) for tag in (entry.drug_tags or [])]
    conditions = [tag_names(tag) for tag in (entry.condition_tags or [])]
    if not any(drugs) and not any(conditions):
        return False
    text = normalize(patient_text)
    # A tag with no usable name can never match, so it fails closed.
    if not all(_mentions(names, text) for names in drugs):
        return False
    if conditions and not any(_mentions(names, text) for names in conditions):
        return False
    return True


def patient_text(document: DischargeDocument, items: list[ExtractedItem]) -> str:
    """Document text plus the text of non-rejected items. Rejected items' names
    may still appear in the document text itself, so rejecting an item does not
    keep its terms out of the gate."""
    parts = [document.text or ""]
    parts.extend(item.text for item in items if item.status != "rejected")
    return "\n".join(parts)


async def embed_entry(llm, entry: DoctorGuidance) -> bool:
    try:
        vectors = await llm.embed([entry.question + "\n" + entry.answer])
    except LLMError:
        entry.embedding = None
        entry.embedding_model = None
        return False
    entry.embedding = [float(v) for v in vectors[0]]
    entry.embedding_model = llm.embedding_model
    return True


async def retrieve(
    session: AsyncSession,
    llm,
    patient: User,
    document: DischargeDocument,
    items: list[ExtractedItem],
    query: str,
    min_similarity: float,
    *,
    doctor_id: int,
    limit: int = 5,
) -> list[tuple[DoctorGuidance, float]]:
    """The best matching guidance entries of one doctor that may be used for
    this patient and document. Only `doctor_id`'s entries are ever candidates:
    a clone never sees another doctor's advice."""
    rows = (
        await session.scalars(
            select(DoctorGuidance).where(
                DoctorGuidance.doctor_id == doctor_id,
                DoctorGuidance.active.is_(True),
                DoctorGuidance.embedding.is_not(None),
                DoctorGuidance.embedding_model == llm.embedding_model,
            )
        )
    ).all()
    text = patient_text(document, items)
    candidates = [
        row for row in rows
        if (
            row.scope == "patient"
            and row.patient_id == patient.id
            and row.document_id == document.id
        )
        or eligible(row, text)
    ]
    if not candidates:
        return []
    query_vec = np.asarray((await llm.embed([query]))[0], dtype=float)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []
    scored: list[tuple[DoctorGuidance, float]] = []
    for row in candidates:
        vec = np.asarray(row.embedding, dtype=float)
        if vec.shape != query_vec.shape:
            continue
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        score = float(vec @ query_vec / (norm * query_norm))
        if score >= min_similarity:
            scored.append((row, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit]
