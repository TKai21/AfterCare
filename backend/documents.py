"""Uploaded file -> text, and text -> the numbered paragraphs the app cites.

Paragraphs are the evidence units: each should be about one instruction (a
sentence, a bullet, a heading, a "Label: value" line), so a citation grounds only
the numbers of that unit. TXT blocks and PDF pages are read the same way: blank
lines always separate paragraphs, and within a block visual lines are merged
back into units (PDF text wraps mid-sentence and almost never has blank lines).
"""
import io
import re
import unicodedata
from pathlib import PurePosixPath

from pypdf import PdfReader

MAX_PARAGRAPH = 600
NO_TEXT_ERROR = "No readable text found. Scanned PDFs need OCR first."


def normalize_text(text: str) -> str:
    """NFKC folds PDF ligatures ("\ufb01" -> "fi") and other compatibility forms,
    so the stored text, the paragraphs and the model's quotes all agree."""
    return unicodedata.normalize("NFKC", text)


def file_suffix(filename: str) -> str:
    return PurePosixPath(filename.replace("\\", "/")).suffix.lower()


def read_upload(data: bytes, filename: str) -> str:
    suffix = file_suffix(filename)
    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:  # malformed PDFs raise all kinds of errors in pypdf
            raise ValueError("The PDF could not be read.") from exc
    elif suffix in {".txt", ".md"}:
        try:
            pages = [data.decode("utf-8-sig")]
        except UnicodeDecodeError as exc:
            raise ValueError("The text file must be UTF-8.") from exc
    else:
        raise ValueError("Use a PDF, TXT, or Markdown file.")
    pages = [normalize_text(page) for page in pages]
    if not any(page.strip() for page in pages):
        raise ValueError(NO_TEXT_ERROR)
    return "\n\n".join(page.strip("\n") for page in pages if page.strip())


# A line that always starts a new unit: a bullet or numbered item, or a heading /
# "Label: value" line such as "Diet:" or "Discharge date: May 1, 2026".
_BULLET = re.compile(r"^(?:[-*\u2022\u25aa\u25e6\u2023\u2043]|\d{1,2}[.)])\s")
_LABEL = re.compile(r"^[A-Z][A-Za-z0-9 ,()'/&-]{0,60}:(?:\s|$)")
# A list item under a heading: a bullet, or a number or letter with "." or ")".
_LIST_ITEM = re.compile(r"^(?:[-*\u2022\u25aa\u25e6\u2023\u2043]|\d{1,2}[.)]|[A-Za-z][.)])\s")
_ENDS_UNIT = re.compile(r"[.!?:;][\"')\]]*$")
_STARTS_CAPITAL = re.compile(r"^[\"'(]?[A-Z0-9]")
# A period after these does not end a sentence.
_ABBREVIATIONS = frozenset("dr mr mrs ms st jr sr vs no approx e.g i.e fig".split())
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(]?[A-Z0-9])")
# A line much shorter than the block's longest line was not wrapped by the PDF
# writer (a heading or a title), so a capitalized next line starts a new unit.
_SHORT_LINE = 0.6


def _ends_with_abbreviation(text: str) -> bool:
    words = text.split()
    return bool(words) and words[-1].rstrip(".").lower() in _ABBREVIATIONS and words[-1][-1] == "."


def _new_unit(prev: str, line: str, width: int) -> bool:
    if _BULLET.match(line) or _LABEL.match(line):
        return True
    if not _STARTS_CAPITAL.match(line) or _ends_with_abbreviation(prev):
        return False
    return bool(_ENDS_UNIT.search(prev)) or len(prev) < _SHORT_LINE * width


def _merge_lines(block: str, lists: bool = False) -> list[str]:
    """Visual lines -> units. A line continues the previous one unless it is a
    bullet or a label, or it starts with a capital after a line that ended a
    sentence (or after a short line). With `lists`, a list item (_LIST_ITEM, so
    also "a)" or "B.") also starts a unit right after a unit ending with ":" or
    after another list item."""
    lines = [" ".join(line.split()) for line in block.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []
    width = max(len(line) for line in lines)
    units = [lines[0]]
    for prev, line in zip(lines, lines[1:]):
        in_list = lists and _LIST_ITEM.match(line) and (
            units[-1].endswith(":") or _LIST_ITEM.match(units[-1]))
        if in_list or _new_unit(prev, line, width):
            units.append(line)
        else:
            units[-1] += " " + line
    return units


def _sentences(unit: str) -> list[str]:
    out: list[str] = []
    start = 0
    for match in _SENTENCE_BREAK.finditer(unit):
        if _ends_with_abbreviation(unit[start:match.end()].rstrip()):
            continue
        out.append(unit[start:match.end()].strip())
        start = match.end()
    out.append(unit[start:].strip())
    return [s for s in out if s]


def _by_words(sentence: str) -> list[str]:
    """Last resort for one sentence over the cap: split between words."""
    pieces: list[str] = []
    piece = ""
    for word in sentence.split():
        if piece and len(piece) + 1 + len(word) > MAX_PARAGRAPH:
            pieces.append(piece)
            piece = word
        else:
            piece = f"{piece} {word}" if piece else word
    return pieces + [piece] if piece else pieces


def _cap(unit: str) -> list[str]:
    """Split a unit over MAX_PARAGRAPH at sentence boundaries, packing whole
    sentences into each part."""
    if len(unit) <= MAX_PARAGRAPH:
        return [unit]
    parts: list[str] = []
    part = ""
    for sentence in _sentences(unit):
        for piece in _by_words(sentence) if len(sentence) > MAX_PARAGRAPH else [sentence]:
            if part and len(part) + 1 + len(piece) > MAX_PARAGRAPH:
                parts.append(part)
                part = piece
            else:
                part = f"{part} {piece}" if part else piece
    return parts + [part] if part else parts


def paragraphs(text: str) -> list[str]:
    result: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        for unit in _merge_lines(block):
            result.extend(_cap(unit))
    return result


MAX_LIST_ITEMS = 15
# The longest merged evidence paragraph (heading plus items, with line breaks).
MAX_MERGED = 900
# A dose unit: a list whose items carry one (or name a drug) is never merged, so
# a citation of one medication item never grounds another item's dose.
_DOSE_UNIT = re.compile(
    r"(?<![A-Za-z])(?:mg|mcg|ml|units?|tablets?|capsules?|puffs?|drops?|tsp|tbsp)(?![A-Za-z])"
    r"|(?<=[0-9])\s*g(?![A-Za-z])",
    re.IGNORECASE)


def _names_a_dose(item: str) -> bool:
    from drugnames import get_lexicon  # the lexicon loads a data file: only when needed
    return bool(_DOSE_UNIT.search(item) or get_lexicon().find(item))


def evidence_paragraphs(text: str) -> list[str]:
    """The clone's evidence units: paragraphs(), except that a unit ending with ":"
    and the list items directly under it (in the same block, so until a blank
    line) are one paragraph with the heading on its first line and each item on
    its own line. Citing "Call 911 if:" then covers its items; the agent grounds
    each statement in the heading and one item (agent.validate). A merged
    paragraph has at most MAX_LIST_ITEMS items and MAX_MERGED characters, and
    merging stops at an item the paragraph cap (_cap) split; later items stay
    paragraphs of their own. A list with any item that names a drug or carries a
    dose unit (_names_a_dose) is never merged. The review page, PDF
    highlighting and extraction keep paragraphs()."""
    result: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        # (part, whether its unit was split by the cap)
        units = [(part, len(parts) > 1) for unit in _merge_lines(block, lists=True)
                 for parts in [_cap(unit)] for part in parts]
        i = 0
        while i < len(units):
            heading, split = units[i]
            end = i + 1
            if heading.endswith(":") and not split:
                last = end
                while last < len(units) and _LIST_ITEM.match(units[last][0]):
                    last += 1
                items = [part for part, _ in units[end:last]]
                if not any(_names_a_dose(item) for item in items):
                    size = len(heading)
                    while (end < last and end - i <= MAX_LIST_ITEMS and not units[end][1]
                           and size + 1 + len(units[end][0]) <= MAX_MERGED):
                        size += 1 + len(units[end][0])
                        end += 1
            result.append("\n".join(part for part, _ in units[i:end]))
            i = end
    return result
