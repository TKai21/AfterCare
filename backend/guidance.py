"""Doctor guidance memory: the applicability gate, embedding, and retrieval.

The gate is the safety mechanism: a "general" entry may only reach a patient
whose discharge document or confirmed instructions mention its drugs and
conditions as whole words. Anything unclear fails closed.
"""
import re
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from extraction import normalize
from llm import LLMError
from models import DischargeDocument, DoctorGuidance, ExtractedItem, GuidanceRule, User

# At most this many rules of one entry among the hits, so one long answer
# cannot crowd out the doctor's other advice.
RULES_PER_ENTRY = 3


@dataclass(frozen=True)
class RuleHit:
    """One rule of an entry, found by retrieval. The entry carries the gate,
    the title and the scope; the rule carries the evidence."""
    entry: DoctorGuidance
    rule: GuidanceRule


# What retrieval returns: a rule of an entry that has rules, else the whole entry.
Unit = Union[DoctorGuidance, RuleHit]


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


def _cosine(vector, query_vec: np.ndarray, query_norm: float) -> float | None:
    vec = np.asarray(vector, dtype=float)
    if vec.shape != query_vec.shape:
        return None
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return float(vec @ query_vec / (norm * query_norm))


def rules_usable(entry: DoctorGuidance, rules: list[GuidanceRule], model: str) -> bool:
    """True when the entry is scored by its rules: they were read and every live
    advice rule has an embedding of the current model. Habits and suppressed rows
    are never scored, so their embeddings never decide this; an entry left with
    no live advice rule adds nothing. Otherwise the entry is scored whole."""
    return (
        entry.rules_status == "ready"
        and bool(rules)
        and all(
            r.embedding is not None and r.embedding_model == model
            for r in rules if r.kind == "advice" and not r.suppressed
        )
    )


def select_hits(scored: list[tuple[Unit, float]], floor: float, margin: float,
                limit: int) -> list[tuple[Unit, float]]:
    """Best first, at most `limit`, keeping a hit only when its score is at
    least `floor` and within `margin` of the best score."""
    ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
    if not ranked:
        return []
    cutoff = max(floor, ranked[0][1] - margin)
    return [pair for pair in ranked if pair[1] >= cutoff][:limit]


async def retrieve(
    session: AsyncSession,
    llm,
    patient: User,
    document: DischargeDocument,
    items: list[ExtractedItem],
    query: str,
    *,
    doctor_id: int,
    floor: float,
    margin: float,
    limit: int = 6,
) -> list[tuple[Unit, float]]:
    """The best matching guidance of one doctor that may be used for this
    patient and document. Only `doctor_id`'s entries are ever candidates: a
    clone never sees another doctor's advice.

    The gate runs on entries first; only then are an eligible entry's rules
    scored (RuleHit), or the entry itself when it has no usable rules. A hit
    must score at least `floor` and no more than `margin` below the best score;
    at most `limit` are returned, best first."""
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
    rules_by_entry: dict[int, list[GuidanceRule]] = {row.id: [] for row in candidates}
    for rule in await session.scalars(
        select(GuidanceRule)
        .where(GuidanceRule.guidance_id.in_(list(rules_by_entry)))
        .order_by(GuidanceRule.guidance_id, GuidanceRule.position)
    ):
        rules_by_entry[rule.guidance_id].append(rule)
    query_vec = np.asarray((await llm.embed([query]))[0], dtype=float)
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm == 0:
        return []
    scored: list[tuple[Unit, float]] = []
    for row in candidates:
        rules = rules_by_entry[row.id]
        if rules_usable(row, rules, llm.embedding_model):
            hits = []
            # rules_usable looked only at the live advice rules' embeddings;
            # only that advice is scored, so an entry whose rules are all habits
            # or suppressed adds nothing and its whole answer never becomes
            # evidence.
            for rule in rules:
                if rule.kind != "advice" or rule.suppressed:
                    continue
                score = _cosine(rule.embedding, query_vec, query_norm)
                if score is not None:
                    hits.append((RuleHit(row, rule), score))
            hits.sort(key=lambda pair: pair[1], reverse=True)
            scored.extend(hits[:RULES_PER_ENTRY])
        else:
            score = _cosine(row.embedding, query_vec, query_norm)
            if score is not None:
                scored.append((row, score))
    return select_hits(scored, floor, margin, limit)


def unit_entry(unit: Unit) -> DoctorGuidance:
    return unit.entry if isinstance(unit, RuleHit) else unit


HABITS_IN_CONTEXT = 30


async def doctor_habits(session: AsyncSession, doctor_id: int,
                        limit: Optional[int] = HABITS_IN_CONTEXT) -> list[GuidanceRule]:
    """The doctor's live habits, given to the clone on every turn (a habit is
    never retrieved by similarity). Only habits of this doctor's active entries
    that the doctor has not suppressed; a habit written more than once (same
    normalised statement) is kept once, from the most recently updated entry,
    and the list runs newest entry first. limit=None returns every habit."""
    rows = await session.execute(
        select(GuidanceRule)
        .join(DoctorGuidance, DoctorGuidance.id == GuidanceRule.guidance_id)
        .where(
            DoctorGuidance.doctor_id == doctor_id,
            DoctorGuidance.active.is_(True),
            GuidanceRule.doctor_id == doctor_id,
            GuidanceRule.kind == "habit",
            GuidanceRule.suppressed.is_(False),
        )
        .order_by(DoctorGuidance.updated_at.desc(), DoctorGuidance.id.desc(),
                  GuidanceRule.position, GuidanceRule.id)
    )
    seen: set[str] = set()
    habits: list[GuidanceRule] = []
    for rule in rows.scalars():
        key = normalize(rule.statement)
        if key in seen:
            continue
        seen.add(key)
        habits.append(rule)
    return habits[:limit]
