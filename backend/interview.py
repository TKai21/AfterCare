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

from agent import _CLONE_BANNED, PendingClarify, _loose, check_clone_question
from drugnames import get_lexicon
from guidance import _cosine
from llm import LLMError
from safety import _content, negated_terms
from models import DoctorGuidance, GuidanceRule

logger = logging.getLogger(__name__)

ITEM_STATUSES = ("unknown", "answered", "unsure", "skipped")
QUOTE_SHOWN_MAX = 60


# The procedure's stages, always asked in this order (spec 2026-11-19, 4.1).
SAFETY, PROBLEM, FOLLOW_UP, MEDICATION = 1, 2, 3, 4
STAGE_NAMES = {SAFETY: "Safety check", PROBLEM: "Your symptoms", FOLLOW_UP: "A few more questions",
               MEDICATION: "Your medicines"}
# Core items: key -> (stage, label, fixed question). The three about the problem
# are in every interview of the procedure: they are the facts every case is
# compared on. "medicines" is asked whenever the document lists a medication,
# "other_medicines" only when the build call says medicines matter here. There is no closing "anything else?"
# question: it invited patients to retell everything (trial of 2026-11-19); the
# summary they confirm is where they add or correct.
CORE = {
    "duration": (PROBLEM, "how long it has lasted", "How long has this been going on?"),
    "severity": (PROBLEM, "how bad it is",
                 "How bad is it right now: mild, moderate or severe?"),
    "course": (PROBLEM, "getting better or worse",
               "Is it getting better, getting worse, or staying the same?"),
    "medicines": (MEDICATION, "taking prescribed medicines", ""),
    "other_medicines": (MEDICATION, "anything else taken for it",
                        "Have you taken anything else for this, such as an over-the-counter "
                        "medicine?"),
}
MAX_SIDE_QUESTIONS = 3
CONFIRM_TEXT = "Here is what I understood. Is this right?"
# A typed "I don't know": the item is unsure at once, not asked again.
NOT_SURE_REPLY = re.compile(
    r"^\W*(?:i\W?m|i am)?\s*(?:not (?:really )?sure|unsure|no idea)\b[^?]*$|"
    r"^\W*i (?:do not|don\W?t|dont) (?:really )?(?:know|remember)\b[^?]*$", re.IGNORECASE)


def steps(iv: "Interview") -> list[int]:
    """The stages this interview has, in order."""
    return sorted({stage_of(i) for i in iv.items})
CONFIRMED_TEXT = "That's right."
END_TEXT = "That's everything I needed. You can ask another question any time."


@dataclass
class Item:
    label: str
    habits: list[int]
    safety: bool
    status: str = "unknown"
    quote: str = ""
    attempts: int = 0
    stage: int = FOLLOW_UP
    core: str = ""  # a key of CORE, or "" for a habit or safety item
    fixed: str = ""  # a core item's fixed question, used when the doctor has none


@dataclass
class Interview:
    items: list[Item]
    rounds: list[dict] = field(default_factory=list)
    side: int = 0  # side questions answered so far (procedure only)
    confirmed: Optional[bool] = None  # the patient's answer to the summary, once given


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
        try:
            attempts = int(raw.get("attempts") or 0)
        except (TypeError, ValueError):
            return None
        safety = bool(raw.get("safety"))
        stage = raw.get("stage")
        core = raw.get("core")
        items.append(Item(
            label=raw["label"],
            habits=[h for h in raw.get("habits") or [] if isinstance(h, int)],
            safety=safety,
            status=status if status in ITEM_STATUSES else "unknown",
            quote=str(raw.get("quote") or ""),
            attempts=attempts,
            # A row saved before the stages: safety first, then the rest.
            stage=(stage if isinstance(stage, int) and not isinstance(stage, bool)
                   and stage in STAGE_NAMES else SAFETY if safety else FOLLOW_UP),
            core=core if isinstance(core, str) and core in CORE else "",
            fixed=str(raw.get("fixed") or ""),
        ))
    rounds = [dict(r) for r in intake.get("rounds") or [] if isinstance(r, dict)]
    side = intake.get("side")
    confirmed = intake.get("confirmed")
    return Interview(items=items, rounds=rounds,
                     side=side if isinstance(side, int) and not isinstance(side, bool) else 0,
                     confirmed=confirmed if isinstance(confirmed, bool) else None)


def dump_interview(iv: Interview) -> dict:
    return {"items": [asdict(i) for i in iv.items], "rounds": [dict(r) for r in iv.rounds],
            "side": iv.side, "confirmed": iv.confirmed}


def is_procedure(iv: Interview) -> bool:
    """An interview built as the staged procedure (it has core items)."""
    return any(i.core for i in iv.items)


def next_item(iv: Interview) -> Optional[int]:
    """The next unknown item: by stage, then in list order (safety is stage 1)."""
    unknown = [n for n, i in enumerate(iv.items) if i.status == "unknown"]
    return min(unknown, key=lambda n: (stage_of(iv.items[n]), n)) if unknown else None


def stage_of(item: Item) -> int:
    """A safety item is always the first stage."""
    return SAFETY if item.safety else item.stage


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


def _mostly_original(quote: str, original: str) -> bool:
    """The quote is the patient's whole first message, or a long stretch of it
    (one the shown quote would have to cut): it says nothing about one item."""
    q, o = _squash(quote), _squash(original)
    return q == o or (len(quote) > QUOTE_SHOWN_MAX and q in o)


def so_far(iv: Interview, original: str = "") -> list[dict]:
    """What the patient has told so far. An answered safety item is "absent"
    (the patient denied it); an answered item whose quote is the patient's whole
    first message is "mentioned", with no quote (as is a quote longer than the shown cap taken from it); else a quote shortened."""
    out = []
    staged = is_procedure(iv)
    # In the procedure the list is by stage (stable: list order inside a stage).
    for i in (sorted(iv.items, key=stage_of) if staged else iv.items):
        if i.status == "unsure":
            entry = {"label": i.label, "status": "unsure", "quote": ""}
        elif i.status == "answered" and i.safety:
            entry = {"label": i.label, "status": "absent", "quote": ""}
        elif i.status == "answered" and original and _mostly_original(i.quote, original):
            entry = {"label": i.label, "status": "mentioned", "quote": ""}
        elif i.status == "answered":
            entry = {"label": i.label, "status": "answered", "quote": _shorten(i.quote)}
        else:
            continue
        out.append({**entry, "stage": stage_of(i)} if staged else entry)
    if staged:
        # A sign the patient denied with the danger check's No, next to the safety items.
        listed = {_squash(e["label"]) for e in out}
        denied = [{"label": name, "status": "absent", "quote": "", "stage": SAFETY}
                  for name in check_names(confirmed_absent(iv)) if _squash(name) not in listed]
        first = sum(1 for e in out if e["stage"] == SAFETY)
        out[first:first] = denied
    return out


def summary_text(iv: Interview) -> str:
    known = [f"no {i.label}" if i.safety else f'{i.label} ("{i.quote}")' if i.quote else i.label
             for i in iv.items if i.status == "answered"]
    unsure = [i.label for i in iv.items if i.status == "unsure"]
    open_ = [i.label for i in iv.items if i.status in ("skipped", "unknown")]
    lines = [f"{name}: {', '.join(vals)}"
             for name, vals in (("Known", known), ("Not sure", unsure), ("Not answered", open_))
             if vals]
    denied = confirmed_absent(iv)
    if denied:
        lines.append(f"Asked \"are you having this right now?\", the patient chose No: "
                     f"{', '.join(check_names(denied))}")
    check = danger_check(iv)
    if check is not None:
        lines.append(f"Asked \"are you having this right now?\", no answer yet: "
                     f"{', '.join(check_names(check['terms']))}")
    if iv.confirmed is True:
        lines.append("The patient confirmed this summary.")
    elif iv.confirmed is False:
        lines.append("The patient added to or corrected this summary: see their last reply.")
    return "\n".join(lines)


# ---------- model calls ----------

LABEL_MAX_WORDS = 6
ITEM_ATTEMPTS_MAX = 2

BUILD_PROMPT = (
    "You help a doctor's AI clone plan questions for a patient who reported a problem. "
    "You get the patient's message and numbered habits: questions the doctor usually asks "
    "in similar cases. Group the habits into at most as many items as the message allows, each "
    "one thing to find out from the patient. Reply with JSON only: {\"reports_symptom\": true "
    "when the patient reports a symptom or a problem they have now, else false, \"items\": "
    "[{\"label\": a short noun phrase of at most 6 words with no numbers, no drug names "
    "and no advice, \"habits\": [habit ids from the list]}], \"relevant_document_terms\": "
    "[the discharge-document warning terms listed that a doctor would ask this patient about "
    "given this message, copied exactly; [] when none]}. Use only ids from the list.")
# Added to BUILD_PROMPT's user message in the procedure: which habits ask a core question.
CORE_REQUEST = (
    "Also add to the JSON \"core\": {\"duration\": [ids of habits that ask how long the "
    "problem has lasted], \"severity\": [ids that ask how bad it is], \"course\": [ids that "
    "ask whether it is getting better or worse], \"medicines\": [ids that ask whether the "
    "patient is taking their prescribed medicines], \"other_medicines\": [ids that ask what "
    "else the patient has taken or tried for it]}. Use [] when no habit asks it. A habit listed "
    "under \"core\" must not also be in \"items\". Also add \"ask_medicines\": true when a "
    "medicine the patient takes could cause or change this problem, or the patient's message "
    "mentions a medicine; else false.")
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
    if not rows or limit <= 0:
        return [], []
    query = np.asarray((await llm.embed([question]))[0], dtype=float)
    norm = float(np.linalg.norm(query))
    if norm == 0:
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


# Words that do not tell two danger signs apart ("chest pain" and "abdominal
# pain" are different signs).
_SAME_SIGN_IGNORE = _content("pain pains symptom symptoms")
# A sign made only of these says "it is getting worse": the procedure asks that
# as its own question (CORE "course"), and such a term makes no yes/no question.
_ONLY_WORSENING = _content("worse worsen worsens worsening worsened acute acutely become "
                           "becomes becoming get gets getting persist persists persisting "
                           "persistent symptom symptoms")


def _same_sign(a: str, b: str) -> bool:
    """Two wordings of one danger sign ("symptoms worsen acutely", "acute
    worsening of symptoms"): half of the shorter one's telling words are shared."""
    x, y = _content(a) - _SAME_SIGN_IGNORE, _content(b) - _SAME_SIGN_IGNORE
    return bool(x and y) and len(x & y) * 2 >= min(len(x), len(y))


def _safety_terms(warning_habits, document_terms: Sequence[str], taken: list[str],
                  max_safety: int, allowed_drugs: Collection[str],
                  skip_worsening: bool = False) -> list[str]:
    """Warning terms of the habits, then of the document, deduped and not named
    by an item label already (`taken`), at most `max_safety`. A term that fails
    the label check (digit, drug name outside the allowed ones, advice word, too
    long) is left out: it is never asked as a question, and dropping it is safe
    because the close gate still checks every doctor and document warning term
    and the static backstop still fires on any message."""
    out: list[str] = []
    candidates = [t for h in warning_habits for t in (h.warning_terms or [])] + list(document_terms)
    for term in candidates:
        if len(out) >= max_safety:
            break
        if not _label_ok(term, allowed_drugs) or not _loose(term):
            continue
        key = _loose(term)
        if any(key in _loose(t) or _loose(t) in key for t in [*out, *taken]):
            continue
        if any(_same_sign(term, t) for t in out):
            continue
        if skip_worsening and not _content(term) - _ONLY_WORSENING:
            continue
        out.append(" ".join(term.split()))
    return out


def _medicines_question(medications: Sequence[str]) -> str:
    names = list(dict.fromkeys(" ".join(m.split()) for m in medications if m and m.strip()))
    if not names:
        return ""
    if len(names) > 3:
        listed = "your medicines"
    else:
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f"Have you been taking {listed} as your discharge instructions say?"


def core_items(core_habits: dict[str, list[int]], medications: Sequence[str],
               ask_medicines: bool = True) -> list[Item]:
    """The procedure's core items, each with its fixed question and the
    doctor's habits that ask the same thing. Whether the patient takes their
    medicines as directed is asked whenever the document lists a medication
    (a patient who stopped one says so only when asked: trial of 2026-11-20);
    what else they took only with `ask_medicines`."""
    fixed = {key: question for key, (_, _, question) in CORE.items()}
    fixed["medicines"] = _medicines_question(medications)
    return [Item(label, list(core_habits.get(key, [])), False, stage=stage, core=key,
                 fixed=fixed[key])
            for key, (stage, label, _) in CORE.items()
            if fixed[key] and (key != "other_medicines" or ask_medicines)]


async def build_items(llm, question: str, ask_habits, warning_habits, document_terms: list[str],
                      allowed_drugs, *, max_items: int, max_safety: int,
                      procedure: Optional[Sequence[str]] = None,
                      ) -> Optional[tuple[bool, list[Item]]]:
    """(reports_symptom, items) for a first-turn message, or None when the model
    call or its output is unusable. Ask habits are grouped by the model; every
    id must be one of `ask_habits` and every label plain (a bad item is dropped).
    Safety items (one per warning term, asked first) are added by the program.

    With `procedure` (the document's medication names) the items are the staged
    procedure: the core items are added, a habit the model maps to a core item
    is that item's question, not a follow-up, and a danger sign that only says
    "getting worse" is left to the core question about that."""
    listing = "\n".join(f"{h.id}: {h.quote}" for h in ask_habits) or "(none)"
    doc_listing = "\n".join(f"- {t}" for t in document_terms) or "(none)"
    data = await _json(llm, BUILD_PROMPT, f"At most {max_items} items.\n\n"
                       f"Patient message:\n{question}\n\nHabits:\n{listing}\n\n"
                       f"Document warning terms:\n{doc_listing}"
                       + (f"\n\n{CORE_REQUEST}" if procedure is not None else ""))
    if data is None or not isinstance(data.get("reports_symptom"), bool) \
            or not isinstance(data.get("items"), list):
        return None
    known = {h.id for h in ask_habits}
    core_habits: dict[str, list[int]] = {}
    if procedure is not None and isinstance(data.get("core"), dict):
        used: set[int] = set()
        for key in CORE:
            ids = data["core"].get(key)
            if not isinstance(ids, list):
                continue
            core_habits[key] = [i for i in dict.fromkeys(
                i for i in ids if isinstance(i, int) and not isinstance(i, bool))
                if i in known and i not in used]
            used.update(core_habits[key])
    in_core = {i for ids in core_habits.values() for i in ids}
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
        ids = [i for i in dict.fromkeys(ids) if i not in in_core]
        if not ids:
            continue  # every habit of it asks a core question
        items.append(Item(" ".join(raw["label"].split()), ids, False))
    labels = [i.label for i in items]
    # Document warning terms become questions only when the model marked them
    # relevant to this complaint (and they are in the list); the static backstop
    # still fires on any message, so an unasked term is not lost.
    asked_doc = data.get("relevant_document_terms")
    wanted = {_loose(t) for t in asked_doc if isinstance(t, str)} if isinstance(asked_doc, list) else set()
    relevant = [t for t in document_terms if _loose(t) in wanted]
    items += [Item(t, [], True, stage=SAFETY) for t in
              _safety_terms(warning_habits, relevant, labels, max_safety, allowed_drugs,
                            skip_worsening=procedure is not None)]
    if procedure is not None:
        items += core_items(core_habits, procedure, data.get("ask_medicines") is True)
    return data["reports_symptom"], items


_NEGATION_WORDS = frozenset({"not", "no", "never", "none", "without", "dont", "don", "didnt",
                             "didn", "havent", "haven", "nothing", "neither", "nor"})


def _only_a_denial(quote: str, terms: Sequence[str]) -> bool:
    """The quote is nothing but a denial of a danger sign: a warning term is
    plainly negated in it and no other content word is left."""
    denied = negated_terms(quote, list(terms)) if terms else set()
    if not denied:
        return False
    named = _content(" ".join(terms)) | _content(" ".join(denied))
    return not (_content(quote) - named - _NEGATION_WORDS)


async def extract(llm, iv: Interview, original: str, reply: str,
                  warning_terms: Sequence[str] = ()) -> dict[int, str]:
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
        terms = [*warning_terms, *(i.label for i in iv.items if i.safety)]
        if (isinstance(index, int) and not isinstance(index, bool) and index in open_
                and isinstance(quote, str) and not _only_a_denial(quote, terms)
                and fill(iv, index, quote, [original, reply])):
            got[index] = iv.items[index].quote
    return got


def _habit_questions(item: Item, habit_texts: dict[int, str],
                     asked_before: Sequence[str]) -> list[str]:
    seen = {_loose(q) for q in asked_before}
    texts = (" ".join(habit_texts[h].split()) for h in item.habits if h in habit_texts)
    return [t for t in texts if t.endswith("?") and _loose(t) not in seen]


_QUESTION_WORDS = frozenset({
    "do", "does", "did", "have", "has", "had", "is", "are", "was", "were", "will", "would",
    "can", "could", "tell", "describe", "say", "how", "when", "where", "what", "which", "who",
    "any", "you", "your", "not", "feel", "start", "started"})


def _on_item(question: str, item: Item, habit_texts: dict[int, str]) -> bool:
    """A clone-written question must share a content word (stemmed, no stop or
    question words) with the item's label or one of its habit texts."""
    ref = _content(item.label).union(*(_content(habit_texts[h]) for h in item.habits
                                       if h in habit_texts)) - _content(" ".join(_QUESTION_WORDS))
    return bool(_content(question) & ref)


_VERBISH = re.compile(r"^(you|your|i|my|unable|cannot|can't|worsen\w*)\b|"
                      r"\b(is|are|was|were|has|have|had|feel|feels|feeling|becomes?|persists?|"
                      r"getting|gets|hurts|hurting|keeps?|started|worse|worsening|unable)\b", re.I)


def safety_question(term: str, rephrase: bool) -> str:
    """The plain yes/no question for a danger-sign item, built by code from a
    fixed template: the model never words a question that names a danger sign."""
    term = " ".join(term.split())
    if rephrase:
        return f"Just to be sure, do you have any {term} right now?"
    if _VERBISH.search(term):
        return f"Is this happening now: {term}?"
    return f"Have you had any {term}?"


async def write_question(llm, item: Item, habit_texts: dict[int, str], asked_before: list[str],
                         allowed_drugs, doctor_name: str,
                         rephrase: bool) -> Optional[tuple[str, bool]]:
    """(question, is_doctor_words) for `item`, or None when no checked question
    exists (the caller marks the item skipped, or unsure when `rephrase`).
    First attempt: a habit question word for word. Otherwise the model writes
    one, checked by check_clone_question, with one repair call; then a habit
    question word for word. `habit_texts` holds live habits only."""
    if item.safety:
        return safety_question(item.label, rephrase), False
    habit_questions = _habit_questions(item, habit_texts, asked_before)
    if habit_questions and not rephrase:
        return habit_questions[0], True
    if item.fixed and not rephrase and _loose(item.fixed) not in {_loose(q) for q in asked_before}:
        return item.fixed, False  # a core item's fixed question, written by us
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
            if not _on_item(text, item, habit_texts):
                raise ValueError("The question must be about the item: use its words.")
        except ValueError as e:
            problem = str(e)
            continue
        text = " ".join(text.split())
        # A model question that is word for word one of the doctor's habits is the doctor's.
        return text, any(text == " ".join(t.split()) for t in habit_texts.values())
    return (habit_questions[0], True) if habit_questions else None


# ---------- one interview round (used by the ask route) ----------

NOT_SURE_TEXT = "I'm not sure."


async def live_habit_texts(session: AsyncSession, doctor_id: int,
                           ids: Collection[int]) -> dict[int, str]:
    """id -> quote of the habits among `ids` still live for this doctor (not
    suppressed, entry active): a habit suppressed after the items froze is
    never asked word for word."""
    if not ids:
        return {}
    rows = (await session.execute(
        select(GuidanceRule.id, GuidanceRule.quote)
        .join(DoctorGuidance, DoctorGuidance.id == GuidanceRule.guidance_id)
        .where(GuidanceRule.id.in_(list(ids)), GuidanceRule.doctor_id == doctor_id,
               GuidanceRule.kind == "habit", GuidanceRule.suppressed.is_(False),
               DoctorGuidance.active.is_(True)))).all()
    return {rule_id: quote for rule_id, quote in rows if isinstance(quote, str)}


def habit_ids(iv: Interview) -> list[int]:
    return list(dict.fromkeys(h for i in iv.items for h in i.habits))


def as_pending(iv: Interview, original: str) -> PendingClarify:
    """The interview as the agent's PendingClarify: the last round is the one
    the patient's newest message replies to."""
    done_rounds = [{"asked": r.get("asked") or "", "reply": r.get("reply") or ""}
                   for r in iv.rounds[:-1]]
    last = iv.rounds[-1].get("asked") or "" if iv.rounds else ""
    return PendingClarify(question=original, asked=last, rounds=done_rounds)


def asked_item(iv: Interview) -> Optional[int]:
    """The item the last round asked about, or None."""
    n = iv.rounds[-1].get("item") if iv.rounds else None
    return n if isinstance(n, int) and not isinstance(n, bool) and 0 <= n < len(iv.items) else None


async def ask_next(llm, iv: Interview, habit_texts: dict[int, str], allowed_drugs,
                   doctor_name: str, max_rounds: int) -> Optional[tuple[int, str, bool]]:
    """Ask the next item: (item index, question, is_doctor_words), with the new
    round appended (its reply None) and the item's attempt counted. An item
    with no checked question becomes skipped (unsure on its second attempt,
    which is always a rephrase), and the next one is tried. None when the
    interview is done (no unknown item, or the round cap)."""
    while not done(iv, max_rounds):
        n = next_item(iv)
        item = iv.items[n]
        if item.attempts >= ITEM_ATTEMPTS_MAX:
            mark(iv, n, "unsure")
            continue
        rephrase = item.attempts >= 1
        asked_before = [r.get("asked") or "" for r in iv.rounds]
        got = await write_question(llm, item, habit_texts, asked_before, allowed_drugs,
                                   doctor_name, rephrase)
        if got is None:
            mark(iv, n, "unsure" if rephrase else "skipped")
            continue
        item.attempts += 1
        asked = {"asked": got[0], "item": n, "reply": None}
        if is_procedure(iv):
            asked["doctor"] = got[1]  # to show the question again after a side question
        iv.rounds.append(asked)
        return n, got[0], got[1]
    return None


def open_safety_item(iv: Interview) -> Optional[str]:
    """The label of a safety item the patient did not deny (unknown, unsure or
    skipped): the case cannot close."""
    return next((i.label for i in iv.items if i.safety and i.status != "answered"), None)


# ---------- the procedure's summary and side questions ----------

def awaiting_confirmation(iv: Interview) -> bool:
    """The last round showed the summary and the patient has not answered it."""
    return bool(iv.rounds) and iv.rounds[-1].get("confirm") is True \
        and iv.rounds[-1].get("reply") is None


def ask_confirmation(iv: Interview) -> None:
    iv.rounds.append({"asked": CONFIRM_TEXT, "item": None, "reply": None, "confirm": True})


def open_question(iv: Interview) -> Optional[tuple[int, str, bool]]:
    """The question still waiting for a reply (item index, text, is the doctor's
    words), so it can be shown again after a side question."""
    n = asked_item(iv)
    if n is None or iv.rounds[-1].get("reply") is not None:
        return None
    return n, iv.rounds[-1].get("asked") or "", bool(iv.rounds[-1].get("doctor"))


PROBLEM_CORE = ("duration", "severity", "course")


def unsettled_problem_item(iv: Interview) -> Optional[str]:
    """The label of a core question about the problem (how long, how bad, better
    or worse) the patient could not answer: the clone does not close a case it
    cannot describe."""
    return next((i.label for i in iv.items
                 if i.core in PROBLEM_CORE and i.status != "answered"), None)


# ---------- the danger check: ask, do not guess (2026-11-21) ----------

CHECK_ONE = "Are you having this right now?"
CHECK_MANY = "Are you having any of these right now?"
CHECK_ADVICE = ("If you are, call 911 or follow the emergency instructions in your discharge "
                "document.")
CHECK_YES_TEXT = "Yes, I am."
CHECK_NO_TEXT = "No, I am not."
# Backstop labels in the past tense, as the question names them.
_CHECK_NAMES = {"fainted": "fainting", "passed out": "passing out", "collapsed": "collapsing"}


def check_names(terms: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(_CHECK_NAMES.get(t, t) for t in terms))


def check_text(terms: Sequence[str]) -> str:
    """The danger check as plain text (the page shows the same words as a card)."""
    names = check_names(terms)
    title = CHECK_ONE if len(names) == 1 else CHECK_MANY
    return f"{title[:-1]}: {', '.join(names)}? {CHECK_ADVICE}"


def ask_danger_check(iv: Interview, terms: Sequence[str], reply: str) -> None:
    """Hold `reply` (it names `terms`, and the program cannot tell whether it
    denies them) on the open round until the patient answers the check."""
    iv.rounds[-1]["check"] = {"terms": list(terms), "held": reply}


def danger_check(iv: Interview) -> Optional[dict]:
    """The check waiting for the patient's Yes or No: {terms, held}, or None."""
    check = iv.rounds[-1].get("check") if iv.rounds else None
    if (not isinstance(check, dict) or iv.rounds[-1].get("reply") is not None
            or not isinstance(check.get("held"), str) or not isinstance(check.get("terms"), list)
            or not check["terms"] or not all(isinstance(t, str) for t in check["terms"])):
        return None
    return check


def answer_danger_check(iv: Interview) -> tuple[list[str], str]:
    """The patient chose No: (the terms, the held reply). The terms stay on the
    round as denied by the patient's own button, for the doctor and the close."""
    check = iv.rounds[-1].pop("check")
    iv.rounds[-1]["denied"] = list(dict.fromkeys(
        [*(iv.rounds[-1].get("denied") or []), *check["terms"]]))
    return list(check["terms"]), check["held"]


def confirmed_absent(iv: Interview) -> list[str]:
    """Every danger sign the patient denied with the check's No button."""
    return list(dict.fromkeys(
        t for r in iv.rounds for t in (r.get("denied") or []) if isinstance(t, str)))
