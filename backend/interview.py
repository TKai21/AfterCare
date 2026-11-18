"""The doctor clone's interview: one question per round, items built from the
doctor's habits on similar past cases. This module holds the item state (pure
functions) and the small model calls that build items, read a reply and write a
question. Nothing here talks to the database except similar_case_habits."""
from dataclasses import asdict, dataclass, field
from typing import Optional

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
