"""De-identification at the model boundary (design:
docs/superpowers/specs/2026-11-18-deidentification-design.md).

PrivateLLM wraps the model client. On the way out every text has the names of
patients, clinicians and facilities, and phone numbers, e-mail addresses,
record numbers and addresses replaced by placeholders such as
"[PATIENT_1]"; on the way back the placeholders become the original words
again. Callers never see a placeholder, and the database keeps the original
text. A payload that still holds a known identifier is not sent.
"""
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Optional

from llm import LLMError

log = logging.getLogger(__name__)

PATIENT, DOCTOR, FACILITY = "PATIENT", "DOCTOR", "FACILITY"
PHONE, EMAIL, ID, ADDRESS, DOB = "PHONE", "EMAIL", "ID", "ADDRESS", "DOB"
KINDS = (PATIENT, DOCTOR, FACILITY, PHONE, EMAIL, ID, ADDRESS, DOB)
PLACEHOLDER = re.compile(rf"\[(?:{'|'.join(KINDS)})_\d+\]")

# A part of a name ("Rose" of "Rose Miller") is masked by itself only when it
# is not also an everyday or medical word; the full name is always masked.
_NOT_A_NAME_ALONE = frozenset("""
april may june august summer autumn winter dawn eve
rose grace hope faith joy love will mark bill pat sue art rich frank miles chase
young old white black brown green gray grey long short small little best good
day night rice fish bean salt pepper ginger olive basil sage honey berry cherry
pain rash burns head foot hand hands back legs arms heart chest hip
doctor nurse patient clinic hospital medical center health care family medicine
primary urology cardiology surgery general default
""".split())
_MIN_PART = 3

_LINE_START = r"^[ \t]*(?:\[\d+\][ \t]*)?(?:[-*•][ \t]*)?"
_LABELLED = [
    (PATIENT, r"(?:patient(?:'s)? name|patient|name)"),
    (DOCTOR, r"(?:attending(?: physician)?|physician|provider|surgeon|doctor|pcp|"
             r"primary care (?:provider|physician))"),
    (ID, r"(?:mrn|medical record(?: number| no\.?)?|patient id|account(?: number| no\.?)?)"),
    (DOB, r"(?:dob|date of birth|birth ?date)"),
    (ADDRESS, r"(?:home address|address)"),
]
_LABELLED_LINES = [
    (kind, re.compile(rf"{_LINE_START}{label}[ \t]*[:#][ \t]*(.+)$", re.IGNORECASE | re.MULTILINE))
    for kind, label in _LABELLED
]
_TITLE = re.compile(r"^(?:dr|mr|mrs|ms|miss|prof)\.?\s+", re.IGNORECASE)
# A word of a person's name: a capital, then lower case ("Okafor", "O'Brien",
# "Smith-Jones"); not "AI" or "ER", and never the possessive ("Kai's" is "Kai").
_NAME_WORD = r"[A-Z][a-z\u00C0-\u024F]*(?:[-'\u2019][A-Z][a-z\u00C0-\u024F]+)*(?![A-Za-z])"
# After a label ("Patient:") a name may also be all capitals ("JOHN DOE").
_LABELLED_NAME_WORD = r"[A-Z][A-Za-z\u00C0-\u024F'\u2019-]*"
_INITIAL = r"[A-Z]\."
_FACILITY_WORD = r"[A-Z][A-Za-z\u00C0-\u024F&'.\u2019-]*"
_DOCTOR_IN_TEXT = re.compile(
    rf"\bDr\.?[ \t]+((?:{_INITIAL}[ \t]+)?{_NAME_WORD}(?:[ \t]+(?:{_INITIAL}[ \t]+)?{_NAME_WORD})?)")
_FACILITY_LINE = re.compile(
    rf"{_LINE_START}((?:{_FACILITY_WORD}|of|the)(?:[ \t]+(?:{_FACILITY_WORD}|of|the)){{0,6}}[ \t]+"
    r"(?:Hospital|Medical Center|Medical Centre|Health System|Health Center|Medical Group|Clinic))"
    r"(?=[ \t]*(?:$|[-–|,(]))", re.MULTILINE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_LONG = re.compile(r"(?<![\w-])(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?![\w-])")
# "555-0177", but not a dose range such as "500-1000 mg".
_PHONE_SHORT = re.compile(
    r"(?<![\w.-])\d{3}-\d{4}(?![\w-])(?![ \t]*(?:mg|mcg|g|kg|ml|l|units?|iu|kcal|calories|steps|%)\b)",
    re.IGNORECASE)
_STREET = re.compile(
    r"\b\d{1,6}[ \t]+(?:[NSEW]\.?[ \t]+)?(?:[A-Z][\w'.-]*[ \t]+){1,4}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl)\b\.?"
    r"(?:,?[ \t]+(?:Apt|Suite|Ste|Unit|#)\.?[ \t]*\w+)?")


def clean_name(raw: str) -> str:
    """A labelled line's value as a bare name: no title, nothing after the first
    comma or bracket ("Dr. Helen Okafor, Family Medicine" -> "Helen Okafor")."""
    value = re.split(r"[,(;|]|\s-\s", raw.strip(), maxsplit=1)[0]
    value = re.sub(r"['’]s$", "", _TITLE.sub("", value.strip()).strip())
    return " ".join(value.split())


def _looks_like_a_name(value: str, labelled: bool = False) -> bool:
    words = value.split()
    word = _LABELLED_NAME_WORD if labelled else _NAME_WORD
    return (0 < len(words) <= 4
            and all(re.fullmatch(rf"(?:{word}|{_INITIAL})", w) for w in words)
            and any(len(w) > 1 and re.fullmatch(word, w) for w in words)
            and not all(w.lower() in _NOT_A_NAME_ALONE for w in words))


def identifiers_in(text: str) -> list[tuple[str, str]]:
    """The (kind, value) identifiers that rules read from `text` itself: labelled
    header lines ("Patient: ..."), "Dr. Firstname Lastname", and a facility line."""
    found: list[tuple[str, str]] = []
    for kind, pattern in _LABELLED_LINES:
        for match in pattern.finditer(text):
            raw = match.group(1).strip()
            if kind in (PATIENT, DOCTOR):
                name = clean_name(raw)
                if _looks_like_a_name(name, labelled=True):
                    found.append((kind, name))
            elif kind == DOB:
                found.append((kind, re.split(r"[(;|]", raw, maxsplit=1)[0].strip(" .")))
            else:
                found.append((kind, raw.rstrip(" .")))
    for match in _DOCTOR_IN_TEXT.finditer(text):
        name = clean_name(match.group(1))
        if _looks_like_a_name(name):
            found.append((DOCTOR, name))
    for match in _FACILITY_LINE.finditer(text):
        found.append((FACILITY, " ".join(match.group(1).split())))
    return [(kind, value) for kind, value in found if len(value) >= _MIN_PART]


@dataclass
class Vault:
    """The identifiers known before a call: account names and what the stored
    documents' header lines name. Values are kept as written."""
    known: dict[str, str] = field(default_factory=dict)  # casefolded value -> kind
    spelled: dict[str, str] = field(default_factory=dict)  # casefolded value -> as written

    def add(self, kind: str, value: str) -> None:
        value = " ".join((value or "").split())
        if len(value) >= _MIN_PART:
            self.known.setdefault(value.casefold(), kind)
            self.spelled.setdefault(value.casefold(), value)

    def add_text(self, text: str) -> None:
        for kind, value in identifiers_in(text or ""):
            self.add(kind, value)

    def items(self) -> list[tuple[str, str]]:
        return [(kind, self.spelled[key]) for key, kind in self.known.items()]


def _is_plain_word(part: str, is_medical: Callable[[str], bool]) -> bool:
    return part.casefold() in _NOT_A_NAME_ALONE or is_medical(part)


class Masker:
    """One outbound call's masking: the same value gets the same placeholder in
    every text of the call, numbered per kind as values are first masked, and
    unmasking a masked text gives back the original exactly."""

    def __init__(self, known: Iterable[tuple[str, str]] = (),
                 is_medical: Callable[[str], bool] = lambda word: False):
        self._known = list(known)
        self._is_medical = is_medical
        self._placeholder: dict[str, str] = {}  # casefolded value -> placeholder
        self.original: dict[str, str] = {}  # placeholder -> value as first seen
        self._counts: dict[str, int] = {}
        self.values: set[str] = set()  # casefolded full values the guard looks for

    def _token(self, kind: str, value: str) -> str:
        key = " ".join(value.split()).casefold()
        if key not in self._placeholder:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            token = f"[{kind}_{self._counts[kind]}]"
            self._placeholder[key] = token
            self.original[token] = value
        return self._placeholder[key]

    def _replace_value(self, text: str, kind: str, value: str) -> str:
        """Every occurrence of `value` (any case, flexible spaces), and for a
        person's name each part that is a name by itself (capitalised only).
        The whole value is restored as the vault or the text spells it."""
        words = value.split()
        whole = re.compile(r"(?<![\w])" + r"\s+".join(map(re.escape, words)) + r"(?![\w])",
                           re.IGNORECASE)
        if whole.search(text):
            token = self._token(kind, value)
            text = whole.sub(token, text)
        if kind in (PATIENT, DOCTOR) and len(words) > 1:
            for part in words:
                bare = part.strip(".'’-")
                if len(bare) < _MIN_PART or _is_plain_word(bare, self._is_medical):
                    continue
                alone = re.compile(r"(?<![\w\[])" + re.escape(bare) + r"(?![\w\]])")
                if alone.search(text):
                    # Its own placeholder, so unmasking gives back exactly "Sam".
                    text = alone.sub(self._token(kind, bare), text)
        return text

    def mask(self, text: str) -> str:
        if not text:
            return text
        candidates = [*self._known, *identifiers_in(text)]
        for kind, value in candidates:
            if kind in (PATIENT, DOCTOR, FACILITY):
                self.values.add(" ".join(value.split()).casefold())
        # Longest first, so "Helen Okafor" is one name before "Okafor" is tried.
        for kind, value in sorted(dict.fromkeys(candidates), key=lambda kv: -len(kv[1])):
            text = self._replace_value(text, kind, value)
        # URLs are left alone: the ones a call carries are the library's public
        # sources, and a name inside one is masked like any other.
        for kind, pattern in ((EMAIL, _EMAIL), (ID, _SSN), (PHONE, _PHONE_LONG),
                              (PHONE, _PHONE_SHORT), (ADDRESS, _STREET)):
            text = pattern.sub(lambda m, k=kind: self._token(k, m.group()), text)
        return text

    def unmask(self, text: str) -> str:
        if not text or not self.original:
            return text
        return PLACEHOLDER.sub(lambda m: self.original.get(m.group(), m.group()), text)

    def leaked(self, text: str) -> Optional[str]:
        """A known name or facility still in masked `text`, or None."""
        folded = " ".join((text or "").split()).casefold()
        for value in self.values:
            if re.search(r"(?<![\w])" + re.escape(value) + r"(?![\w])", folded):
                return value
        return None


VaultLoader = Callable[[], Awaitable[Vault]]


class PrivateLLM:
    """The model client with de-identification: the same chat, embed and
    transcribe interface as LLMClient. `load_vault` reads the known identifiers
    from the database; it is called again once the vault is older than
    `max_age` seconds. A failed load keeps the previous vault, and with none at
    all the rules that read each text still apply."""

    def __init__(self, inner, load_vault: Optional[VaultLoader] = None, *,
                 is_medical: Callable[[str], bool] = lambda word: False, max_age: float = 30.0):
        self.inner = inner
        self._load_vault = load_vault
        self._is_medical = is_medical
        self._max_age = max_age
        self._vault = Vault()
        self._loaded_at: Optional[float] = None

    @property
    def embedding_model(self) -> str:
        return self.inner.embedding_model

    def __getattr__(self, name):  # settings and anything else the client offers
        return getattr(self.inner, name)

    def forget(self) -> None:
        """The next call reloads the vault (after a new document or account)."""
        self._loaded_at = None

    async def _masker(self) -> Masker:
        stale = self._loaded_at is None or time.monotonic() - self._loaded_at > self._max_age
        if self._load_vault is not None and stale:
            try:
                self._vault = await self._load_vault()
                self._loaded_at = time.monotonic()
            except Exception:
                log.exception("Loading the identifiers to mask failed; keeping the last ones")
        return Masker(self._vault.items(), self._is_medical)

    @staticmethod
    def _check(masker: Masker, texts: Iterable[str]) -> None:
        for text in texts:
            if masker.leaked(text) is not None:
                # The value is not logged: the log must not hold it either.
                log.error("A model call still held a known identifier and was not sent")
                raise LLMError("The request could not be de-identified, so it was not sent.")

    async def chat(self, messages, tools=None, json_mode=False):
        masker = await self._masker()
        outbound = []
        for message in messages:
            sent = dict(message)
            if isinstance(sent.get("content"), str):
                sent["content"] = masker.mask(sent["content"])
            if sent.get("tool_calls"):
                sent["tool_calls"] = [_with_arguments(call, masker.mask)
                                      for call in sent["tool_calls"]]
            outbound.append(sent)
        self._check(masker, (
            text for m in outbound
            for text in [m.get("content") if isinstance(m.get("content"), str) else "",
                         *(_arguments(c) for c in m.get("tool_calls") or [])]))
        reply = await self.inner.chat(outbound, tools=tools, json_mode=json_mode)
        back = dict(reply)
        if isinstance(back.get("content"), str):
            back["content"] = masker.unmask(back["content"])
        if back.get("tool_calls"):
            back["tool_calls"] = [_with_arguments(call, masker.unmask)
                                  for call in back["tool_calls"]]
        texts = [back.get("content") if isinstance(back.get("content"), str) else "",
                 *(_arguments(c) for c in back.get("tool_calls") or [])]
        if any(PLACEHOLDER.search(text) for text in texts):
            raise LLMError("The model's reply used a placeholder that stands for nothing.")
        return back

    async def embed(self, texts):
        masker = await self._masker()
        masked = [masker.mask(text) for text in texts]
        self._check(masker, masked)
        return await self.inner.embed(masked)

    async def transcribe(self, data: bytes, filename: str, content_type: str) -> str:
        # Audio cannot be de-identified; it is sent as recorded and not stored.
        return await self.inner.transcribe(data, filename, content_type)


def _arguments(call: dict) -> str:
    arguments = (call.get("function") or {}).get("arguments")
    return arguments if isinstance(arguments, str) else ""


def _with_arguments(call: dict, change: Callable[[str], str]) -> dict:
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
        return call
    return {**call, "function": {**function, "arguments": change(function["arguments"])}}


def vault_loader(session_factory) -> VaultLoader:
    """Reads every account name and every stored document's header identifiers."""
    from sqlalchemy import select

    from models import DischargeDocument, User

    async def load() -> Vault:
        vault = Vault()
        async with session_factory() as session:
            for role, name in (await session.execute(select(User.role, User.display_name))).all():
                name = clean_name(name or "")
                if _looks_like_a_name(name):
                    vault.add(DOCTOR if role == "doctor" else PATIENT, name)
            for (text,) in (await session.execute(select(DischargeDocument.text))).all():
                vault.add_text(text)
        return vault

    return load
