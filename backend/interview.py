"""The doctor clone's interview: one question per round, items built from the
doctor's habits on similar past cases. This module holds the item state (pure
functions) and the small model calls that build items, read a reply and write a
question. Nothing here talks to the database except similar_case_habits."""
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Collection, Optional, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent import _CLONE_BANNED, _loose, check_clone_question
from drugnames import get_lexicon
from guidance import _cosine
from llm import LLMError
from models import DoctorGuidance, GuidanceRule

logger = logging.getLogger(__name__)

ITEM_STATUSES = ("unknown", "answered", "unsure", "skipped")
QUOTE_SHOWN_MAX = 60


@dataclass
class Item:
    label: str
    habits: list[int]
    safety: bool
    status: str = "unknown"
    quote: str = ""
    attempts: int = 0


@dataclass
class Interview:
    items: list[Item]
    rounds: list[dict] = field(default_factory=list)


def is_interview(intake) -> bool:
    return isinstance(intake, dict) and isinstance(intake.get("items"), list)


def load_interview(intake) -> Optional[Interview]:
    if not is_interview(intake):
        return None
    items = []
    for raw in intake["items"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("label"), str):
            return None
        status = raw.get("status", "unknown")
        items.append(Item(
            label=raw["label"],
            habits=[h for h in raw.get("habits") or [] if isinstance(h, int)],
            safety=bool(raw.get("safety")),
            status=status if status in ITEM_STATUSES else "unknown",
            quote=str(raw.get("quote") or ""),
            attempts=int(raw.get("attempts") or 0),
        ))
    rounds = [dict(r) for r in intake.get("rounds") or [] if isinstance(r, dict)]
    return Interview(items=items, rounds=rounds)


def dump_interview(iv: Interview) -> dict:
    return {"items": [asdict(i) for i in iv.items], "rounds": [dict(r) for r in iv.rounds]}


def next_item(iv: Interview) -> Optional[int]:
    unknown = [n for n, i in enumerate(iv.items) if i.status == "unknown"]
    for n in unknown:
        if iv.items[n].safety:
            return n
    return unknown[0] if unknown else None


def done(iv: Interview, max_rounds: int) -> bool:
    return next_item(iv) is None or len(iv.rounds) >= max_rounds


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


def fill(iv: Interview, index: int, quote: str, sources: list[str]) -> bool:
    """Mark the item answered with `quote`, only when the quote is in one of the
    patient's own texts (case and whitespace aside)."""
    needle = _squash(quote)
    if not needle or not 0 <= index < len(iv.items):
        return False
    if not any(needle in _squash(s) for s in sources):
        return False
    item = iv.items[index]
    item.status, item.quote = "answered", " ".join(quote.split())
    return True


def mark(iv: Interview, index: int, status: str) -> None:
    if status not in ITEM_STATUSES:
        raise ValueError(f"Unknown item status: {status}")
    iv.items[index].status = status


def _shorten(quote: str) -> str:
    if len(quote) <= QUOTE_SHOWN_MAX:
        return quote
    cut = quote[:QUOTE_SHOWN_MAX]
    if not quote[QUOTE_SHOWN_MAX].isspace():
        cut = cut.rsplit(None, 1)[0] if " " in cut else cut
    return cut.rstrip() + "..."


def so_far(iv: Interview) -> list[dict]:
    return [{"label": i.label, "status": i.status,
             "quote": _shorten(i.quote) if i.status == "answered" else ""}
            for i in iv.items if i.status in ("answered", "unsure")]


def summary_text(iv: Interview) -> str:
    known = [f'{i.label} ("{i.quote}")' if i.quote else i.label
             for i in iv.items if i.status == "answered"]
    unsure = [i.label for i in iv.items if i.status == "unsure"]
    open_ = [i.label for i in iv.items if i.status in ("skipped", "unknown")]
    lines = [f"{name}: {', '.join(vals)}"
             for name, vals in (("Known", known), ("Not sure", unsure), ("Not answered", open_))
             if vals]
    return "\n".join(lines)


# ---------- model calls ----------

LABEL_MAX_WORDS = 6
ITEM_ATTEMPTS_MAX = 2

BUILD_PROMPT = (
    "You help a doctor's AI clone plan questions for a patient who reported a problem. "
    "You get the patient's message and numbered habits: questions the doctor usually asks "
    "in similar cases. Group the habits into at most {max_items} items, each one thing to find "
    "out from the patient. Reply with JSON only: {{\"reports_symptom\": true when the patient "
    "reports a symptom or a problem they have now, else false, \"items\": [{{\"label\": a "
    "short noun phrase of at most 6 words with no numbers, no drug names and no advice, "
    "\"habits\": [habit ids from the list]}}]}}. Use only ids from the list.")
EXTRACT_PROMPT = (
    "You read a patient's reply for a doctor's AI clone. Items are things the clone wants to "
    "know. For each item the patient's words already answer (in the original question or the "
    "reply), return its index, status \"answered\" and a quote copied exactly from the "
    "patient's words. Do not guess; leave unanswered items out. Reply with JSON only: "
    "{\"items\": [{\"index\": number, \"status\": \"answered\", \"quote\": text}]}.")
QUESTION_PROMPT = (
    "You write one short question in the voice of {doctor}, a doctor, to learn about one item "
    "from a patient. One sentence ending with \"?\", at most 25 words, plain history question "
    "(for example \"Do you ...\", \"Have you ...\", \"When did you ...\"), no advice, no "
    "numbers, no drug names, no commas. Reply with JSON only: {{\"question\": text}}.")


async def similar_case_habits(session: AsyncSession, llm, doctor_id: int, question: str,
                              limit: int) -> tuple[list[GuidanceRule], list[GuidanceRule]]:
    """The live ask and warning habits of the `limit` guidance entries of this
    doctor nearest to `question` (the same embedding and cosine as advice
    retrieval), ordered by the entry's similarity. Raises LLMError when the
    question cannot be embedded."""
    rows = (await session.scalars(select(DoctorGuidance).where(
        DoctorGuidance.doctor_id == doctor_id,
        DoctorGuidance.active.is_(True),
        DoctorGuidance.embedding.is_not(None),
        DoctorGuidance.embedding_model == llm.embedding_model))).all()
    query = np.asarray((await llm.embed([question]))[0], dtype=float)
    norm = float(np.linalg.norm(query))
    if not rows or norm == 0 or limit <= 0:
        return [], []
    scored = [(row, _cosine(row.embedding, query, norm)) for row in rows]
    scored = [(row, score) for row, score in scored if score is not None]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    rank = {row.id: n for n, (row, _) in enumerate(scored[:limit])}
    habits = (await session.scalars(select(GuidanceRule).where(
        GuidanceRule.guidance_id.in_(list(rank)), GuidanceRule.kind == "habit",
        GuidanceRule.suppressed.is_(False),
        GuidanceRule.habit_type.in_(("ask", "warning"))))).all()
    habits.sort(key=lambda r: (rank[r.guidance_id], r.position, r.id))
    return ([r for r in habits if r.habit_type == "ask"],
            [r for r in habits if r.habit_type == "warning"])


async def _json(llm, system: str, user: str) -> Optional[dict]:
    """One JSON-mode call; None on a model error or anything but a JSON object."""
    try:
        message = await llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            json_mode=True)
        data = json.loads(message.get("content") or "")
    except (LLMError, ValueError, TypeError, AttributeError) as e:
        logger.warning("Interview model call failed: %s", e)
        return None
    return data if isinstance(data, dict) else None


def _label_ok(label, allowed_drugs: Collection[str]) -> bool:
    if not isinstance(label, str):
        return False
    label = " ".join(label.split())
    if not label or len(label.split()) > LABEL_MAX_WORDS or re.search(r"\d", label):
        return False
    if _CLONE_BANNED.search(label):
        return False
    lexicon = get_lexicon()
    allowed = set(allowed_drugs).union(*(lexicon.equivalents(n) for n in allowed_drugs))
    return not lexicon.find(label) - allowed


def _safety_terms(warning_habits, document_terms: Sequence[str], taken: list[str],
                  max_safety: int) -> list[str]:
    """Warning terms of the habits, then of the document, deduped and not named
    by an item label already (`taken`), at most `max_safety`."""
    out: list[str] = []
    candidates = [t for h in warning_habits for t in (h.warning_terms or [])] + list(document_terms)
    for term in candidates:
        if len(out) >= max_safety:
            break
        if not isinstance(term, str) or not _loose(term):
            continue
        key = _loose(term)
        if any(key in _loose(t) or _loose(t) in key for t in [*out, *taken]):
            continue
        out.append(" ".join(term.split()))
    return out


async def build_items(llm, question: str, ask_habits, warning_habits, document_terms: list[str],
                      allowed_drugs, *, max_items: int,
                      max_safety: int) -> Optional[tuple[bool, list[Item]]]:
    """(reports_symptom, items) for a first-turn message, or None when the model
    call or its output is unusable. Ask habits are grouped by the model; every
    id must be one of `ask_habits` and every label plain (a bad item is dropped).
    Safety items (one per warning term, asked first) are added by the program."""
    listing = "\n".join(f"{h.id}: {h.quote}" for h in ask_habits) or "(none)"
    data = await _json(llm, BUILD_PROMPT.format(max_items=max_items),
                       f"Patient message:\n{question}\n\nHabits:\n{listing}")
    if data is None or not isinstance(data.get("reports_symptom"), bool) \
            or not isinstance(data.get("items"), list):
        return None
    known = {h.id for h in ask_habits}
    items: list[Item] = []
    for raw in data["items"]:
        if len(items) >= max_items:
            break
        if not isinstance(raw, dict) or not _label_ok(raw.get("label"), allowed_drugs):
            continue
        ids = raw.get("habits")
        if (not isinstance(ids, list) or not ids
                or not all(isinstance(i, int) and not isinstance(i, bool) and i in known
                           for i in ids)):
            continue
        items.append(Item(" ".join(raw["label"].split()), list(dict.fromkeys(ids)), False))
    labels = [i.label for i in items]
    items += [Item(t, [], True) for t in
              _safety_terms(warning_habits, document_terms, labels, max_safety)]
    return data["reports_symptom"], items


async def extract(llm, iv: Interview, original: str, reply: str) -> dict[int, str]:
    """Fill every non-safety unknown item whose answer the patient's own words
    contain; returns item index -> quote of the items filled. A quote that is
    not in `original` or `reply` fills nothing (fill checks it)."""
    open_ = [n for n, i in enumerate(iv.items) if i.status == "unknown" and not i.safety]
    if not open_:
        return {}
    listing = "\n".join(f"{n}: {iv.items[n].label}" for n in open_)
    data = await _json(llm, EXTRACT_PROMPT,
                       f"Items:\n{listing}\n\nOriginal question:\n{original}\n\nReply:\n{reply}")
    if data is None or not isinstance(data.get("items"), list):
        return {}
    got: dict[int, str] = {}
    for raw in data["items"]:
        if not isinstance(raw, dict) or raw.get("status") != "answered":
            continue
        index, quote = raw.get("index"), raw.get("quote")
        if (isinstance(index, int) and not isinstance(index, bool) and index in open_
                and isinstance(quote, str) and fill(iv, index, quote, [original, reply])):
            got[index] = iv.items[index].quote
    return got


def _habit_questions(item: Item, habit_texts: dict[int, str],
                     asked_before: Sequence[str]) -> list[str]:
    seen = {_loose(q) for q in asked_before}
    texts = (" ".join(habit_texts[h].split()) for h in item.habits if h in habit_texts)
    return [t for t in texts if t.endswith("?") and _loose(t) not in seen]


async def write_question(llm, item: Item, habit_texts: dict[int, str], asked_before: list[str],
                         allowed_drugs, doctor_name: str,
                         rephrase: bool) -> Optional[tuple[str, bool]]:
    """(question, is_doctor_words) for `item`, or None when no checked question
    exists (the caller marks the item skipped, or unsure when `rephrase`).
    First attempt: a habit question word for word. Otherwise the model writes
    one, checked by check_clone_question, with one repair call; then a habit
    question word for word. `habit_texts` holds live habits only."""
    habit_questions = _habit_questions(item, habit_texts, asked_before)
    if habit_questions and not rephrase:
        return habit_questions[0], True
    system = QUESTION_PROMPT.format(doctor=doctor_name)
    user = f"Item: {item.label}\nAlready asked: {json.dumps(list(asked_before))}"
    if rephrase:
        user += "\nThe patient did not understand the last question; ask it differently."
    problem = None
    for _ in range(2):
        data = await _json(llm, system,
                           user if problem is None else f"{user}\nYour last question was "
                           f"rejected: {problem} Write another.")
        text = data.get("question") if data else None
        if not isinstance(text, str):
            problem = "Reply with JSON {\"question\": text}."
            continue
        try:
            check_clone_question(text, asked_before, frozenset(allowed_drugs))
        except ValueError as e:
            problem = str(e)
            continue
        return " ".join(text.split()), False
    return (habit_questions[0], True) if habit_questions else None
