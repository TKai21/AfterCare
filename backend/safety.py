"""Server-side emergency term detection for patient chat messages.

False positives are acceptable; missed emergencies are not. No question-form
parsing is attempted. Document warning signs match by content words (stop
words dropped, crude stemming, negations folded to "not"), so the document's
"you cannot keep fluids down" matches "I can't keep fluids down".
"""

import re

from extraction import normalize
from models import ExtractedItem

_BREATH_TAIL = (
    r"(?:\w+\s+)?(?:breath\w*|catch\s+(?:my|a|your)\s+breath|get\s+(?:any\s+)?air)"
)
_CHEST = r"(?:my\s+|the\s+)?chest"
_CANT = r"(?:can'?t|can\s?not|couldn'?t|unable\s+to|not\s+able\s+to)"

# label -> regex, matched against the canonical (lowercase, punctuation-free)
# message. Patterns use a leading word boundary and stems (seiz\w*, faint\w*)
# so inflected forms ("seizures", "overdosed") still hit. Labels are the
# human-readable terms returned to the caller. Err toward detection.
_PATTERNS: dict[str, str] = {
    "chest pain": (
        rf"\bchest\s+(?:(?:is|feels|has|keeps)\s+)?(?:\w+\s+)?"
        rf"(?:pain\w*|ach\w*|hurt\w*|sore)|\bpain\w*\s+(?:in|on)\s+{_CHEST}"
    ),
    "chest pressure": (
        rf"\bchest\s+(?:(?:is|feels|has|keeps)\s+)?(?:\w+\s+)?"
        rf"(?:pressure|tight\w*|heavy|squeez\w*|crush\w*)"
        rf"|\b(?:pressure|tight\w*|squeez\w*|heaviness)\s+(?:in|on)\s+{_CHEST}"
    ),
    "trouble breathing": (
        r"\b(?:trouble|difficult\w*|hard|struggl\w*|labou?red)\s+(?:to\s+|with\s+)?"
        r"(?:\w+\s+)?(?:breath\w*)|\b(?:trouble|difficult\w*|hard|struggl\w*)\s+"
        r"(?:to\s+)?(?:catch\w*|get\w*)\s+(?:my|a|your)?\s*(?:breath|air)"
        r"|\b(?:stopped|not|isn'?t)\s+breathing"
        r"|\bhardly\s+(?:\w+\s+)?breath\w*"
        r"|\bbreathing\s+(?:\w+\s+){0,2}(?:hard|difficult\w*|labou?red|shallow|painful)"
    ),
    "gasping": r"\bgasp\w*",
    "choking": r"\bchok\w*",
    "collapsed": r"\bcollaps\w*",
    "trouble swallowing": (
        r"\b(?:trouble|difficult\w*|hard|struggl\w*|unable|can'?t|can\s?not)\s+"
        r"(?:to\s+)?(?:\w+\s+)?swallow\w*"
    ),
    "sudden vision loss": (
        r"\b(?:sudden\w*|abrupt\w*)\s+(?:\w+\s+){0,2}(?:vision|blind\w*)"
        r"|\b(?:lost|losing|loss\s+of)\s+(?:my\s+|the\s+)?(?:vision|sight)"
        r"|\bgoing\s+blind|\bdouble\s+vision"
    ),
    "shortness of breath": r"\bshort(?:ness)?\s+of\s+breath|\bout\s+of\s+breath",
    "short of breath": r"\bshort(?:ness)?\s+of\s+breath|\bout\s+of\s+breath",
    "can't breathe": (
        rf"\b(?:can'?t|can\s?not|couldn'?t|unable\s+to|not\s+able\s+to)\s+{_BREATH_TAIL}"
    ),
    "cannot breathe": "",  # alias of "can't breathe", set below
    "face drooping": (
        r"\bface\s+(?:is\s+)?(?:droop\w*|sagg\w*)|\bdroop\w*\s+face|\bfacial\s+droop\w*"
    ),
    "stroke": r"\bstroke\b",
    "slurred speech": r"\bslurr\w*\s+(?:speech|words)|\bspeech\s+(?:is\s+)?slurr\w*|\bslurring",
    "weakness on one side": (
        r"\bweak\w*\s+(?:on|in)\s+(?:one|the\s+left|the\s+right|my\s+left|my\s+right)"
    ),
    "numbness on one side": (
        r"\b(?:face|arm|leg|hand)\s+(?:is\s+|feels\s+|went\s+|has\s+gone\s+)?numb\w*|"
        r"\bnumb\w*\s+(?:on|in)\s+(?:one|the\s+left|the\s+right|my\s+left|my\s+right)"
    ),
    "worst headache": r"\bworst\s+(?:\w+\s+)?headache|\bthunderclap",
    "fainted": r"\bfaint\w*",
    "passed out": r"\bpass\w*\s+out|\bblack\w*\s+out",
    "unconscious": r"\bunconscious\w*|\bunresponsive|\bnot\s+responding",
    "coughing up blood": r"\bcough\w*\s+(?:up\s+)?blood",
    "vomiting blood": (
        r"\b(?:vomit\w*|throw\w*\s+up|throw\s+up)\s+(?:up\s+)?blood|\bblood\w*\s+vomit"
        r"|\bblood\s+in\s+(?:\w+\s+){0,2}vomit"
    ),
    "heavy bleeding": (
        r"\bheav\w*\s+bleed\w*|\bbleed\w*\s+(?:heav\w*|a\s+lot|badly|profusely)"
        r"|\bhemorrhag\w*|\bhaemorrhag\w*"
        r"|\bbleed\w*\s+through|\bsoak\w*\s+(?:through\s+)?(?:\w+\s+){0,2}(?:bandage|dressing)"
        r"|\bblood\w*\s+(?:is\s+)?soak\w*"
        r"|\bblood\s+in\s+(?:\w+\s+){0,2}(?:stool|urine|poop|pee)"
    ),
    "bleeding that won't stop": (
        r"\b(?:bleed\w*|blood)\s+(?:\w+\s+){0,3}(?:won'?t|will\s+not|doesn'?t|does\s+not|"
        r"not|isn'?t|cannot|can'?t)\s+(?:\w+\s+)?stop\w*"
        r"|\b(?:won'?t|can'?t|cannot|not)\s+(?:\w+\s+)?stop\w*\s+(?:the\s+)?bleed\w*"
    ),
    "seizure": r"\bseiz\w*|\bconvuls\w*",
    "suicidal": r"\bsuicid\w*",
    "want to die": r"\bwant(?:s|ed)?\s+to\s+die|\bdon'?t\s+(?:\w+\s+)?want\s+to\s+(?:\w+\s+)?(?:live|be\s+alive|be\s+here|go\s+on|exist)",
    "kill myself": r"\bkill\w*\s+(?:my\s?self|me)\b|\bend\w*\s+(?:my\s+life|it\s+all)|\bwant\w*\s+to\s+end\s+(?:it|everything)",
    "self harm": r"\bself\s?harm\w*|\b(?:hurt|harm|cut)\w*\s+my\s?self",
    "overdose": r"\boverdos\w*|\b(?:took|taken|swallow\w*|ate|drank)\s+(?:\w+\s+){0,2}too\s+(?:many|much)",
    "severe allergic reaction": r"\ballergic\s+reaction|\bgoing\s+into\s+anaphyla\w*",
    "anaphylaxis": r"\banaphyla\w*",
    "swelling of the throat": (
        r"\bthroat\s+(?:is\s+|feels\s+)?(?:clos\w*|swell\w*|swollen|tight\w*)"
        r"|\bswell\w*\s+(?:of|in)\s+(?:the\s+|my\s+)?throat|\bswollen\s+throat"
    ),
    "throat closing": "",  # alias of "swelling of the throat", set below
    "face swelling": (
        r"\b(?:face|lips?|tongue)\s+(?:is\s+|are\s+|feels?\s+)?(?:swell\w*|swollen)"
        r"|\bswell\w*\s+(?:of|in)\s+(?:the\s+|my\s+)?(?:face|lips?|tongue)"
        r"|\bswollen\s+(?:face|lips?|tongue)"
    ),
    "blue lips": (
        r"\b(?:blue|bluish|purple|grey|gray)\s+(?:lips?|face)|"
        r"\b(?:lips?|face|skin|fingers?)\s+(?:\w+\s+){0,2}(?:blue|bluish|purple|grey|gray)"
    ),
    "heart attack": r"\bheart\s+attack|\bcardiac\s+arrest",
    "fast or irregular heartbeat": (
        r"\b(?:fast|rapid|racing|pounding|irregular|skipping)\s+(?:heart\s*beats?|heart\s*rate|pulse)"
        r"|\bheart\s+(?:is\s+)?(?:racing|pounding|skipping)"
    ),
    "vomiting that won't stop": (
        r"\bkeep\w*\s+(?:on\s+)?(?:throw\w*\s+up|vomit\w*|puk\w*)"
        rf"|\b{_CANT}\s+(?:\w+\s+)?stop\s+(?:throw\w*\s+up|vomit\w*|puk\w*)"
        r"|\b(?:vomit\w*|throw\w*\s+up|puk\w*)\s+(?:\w+\s+){0,3}(?:won'?t|will\s+not|"
        r"doesn'?t|does\s+not|not|isn'?t)\s+(?:\w+\s+)?stop\w*"
    ),
    "can't keep fluids down": rf"\b{_CANT}\s+keep\s+(?:\w+\s+){{0,3}}down\b",
    "can't urinate": rf"\b{_CANT}\s+(?:\w+\s+)?(?:urinat\w*|pee|void|pass\s+urine)\b",
}
_PATTERNS["cannot breathe"] = _PATTERNS["can't breathe"]
_PATTERNS["throat closing"] = _PATTERNS["swelling of the throat"]
_PATTERNS["short of breath"] = _PATTERNS["shortness of breath"]
_COMPILED = {label: re.compile(rx) for label, rx in _PATTERNS.items()}
EMERGENCY_TERMS: tuple[str, ...] = tuple(_PATTERNS)

# Generic words that appear in nearly every discharge sheet. A document phrase
# made of only one of them would banner almost every message, so it is dropped
# (the static patterns above still catch real emergencies). A phrase needs a
# specific word ("clots", "urinate") or at least two generic ones ("fever ...
# chills") to count.
GENERIC_SINGLE_WORDS = frozenset(
    {
        "pain", "fever", "call", "doctor", "symptoms", "symptom", "swelling",
        "redness", "nausea", "your", "home", "worse", "bleeding", "headache",
        "dizziness", "vomiting", "rash", "cough", "chills", "soreness",
        "discomfort", "infection", "problems", "problem", "signs", "sign",
    }
)

# Words that carry no symptom of their own in a warning-sign line: function
# words, the "call 911 / go to the ER / call the clinic" frame, and intensity
# words a patient may leave out ("severe pain that your medicine does not
# control" must match "the pain my medicine does not control").
STOP_WORDS = frozenset(
    """a an the and or of to for if in on at with by from as into about over under than then
    so is are was were be been being am i you your my me we our us he she they it its this
    that these those there here which who what when where while do does did doing have has
    had having can could will would should may might must shall get gets got very really too
    also just still now any some all more most much many such severe severely high higher
    low lower heavy heavily new sudden suddenly mild bad call go return er emergency room
    clinic doctor provider office hospital right away immediately please see notice
    experience develop start starts started feel feels felt""".split()
)
# Words that must match but do not make a phrase specific on their own.
# Modifiers that carry no symptom on their own: a fragment such as "a very fast"
# (from "a very fast or irregular heartbeat") must not fire on "How fast should I walk?".
WEAK_WORDS = frozenset({"not", "keep", "down", "up", "stop", "come", "make", "take",
                        "very", "fast", "slow", "rapid", "irregular", "pressure", "belly"})
NEGATIONS = frozenset({"no", "not", "never", "cannot", "unable", "cant", "wont", "dont",
                       "doesnt", "didnt", "isnt", "couldnt", "wouldnt", "arent"})

_MIN_CHARS = 4
_SPLIT_CLAUSES = re.compile(r"[,;]")
_SPLIT_WORDS = re.compile(r"[,;]|\bor\b|\band\b", re.IGNORECASE)
_LEADING = re.compile(r"^(?:or|and|if|for)\s+", re.IGNORECASE)


def _canon(s: str) -> str:
    """Normalize, then turn punctuation (except apostrophes) into spaces."""
    s = normalize(s).replace("'", "\0")
    s = re.sub(r"[^\w\0]+", " ", s)
    return s.replace("\0", "'").strip()


def _stem(word: str) -> str:
    """A crude suffix strip, applied the same way to document and message words, so
    "fluids" == "fluid", "urinating" == "urinate", "chills" == "chill"."""
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2]:
        word = word[:-1]  # "controlled" -> "control", like "control"
    return word


def _words(text: str) -> list[str]:
    """Stemmed words with every negation folded to "not"."""
    out = []
    for word in _canon(text).split():
        if word.endswith("n't") or word in NEGATIONS:
            out.append("not")
        else:
            out.append(_stem(word.replace("'", "")))
    return out


_GENERIC_STEMS = frozenset(_stem(w) for w in GENERIC_SINGLE_WORDS)
_STOP_STEMS = frozenset(STOP_WORDS | {_stem(w) for w in STOP_WORDS})
_WEAK_STEMS = frozenset(_stem(w) for w in WEAK_WORDS)


def _content(phrase: str) -> frozenset[str]:
    """The words a message must contain to match `phrase`: no stop words, no
    numbers or one-letter units (a patient's "102" need not equal the
    document's "101 F")."""
    return frozenset(
        w for w in _words(phrase)
        if len(w) > 1 and not any(c.isdigit() for c in w) and w not in _STOP_STEMS
    )


def _specific_enough(content: frozenset[str]) -> bool:
    strong = content - _GENERIC_STEMS - _WEAK_STEMS
    generic = content & _GENERIC_STEMS
    return any(len(w) >= _MIN_CHARS for w in strong) or len(strong) >= 2 or len(generic) >= 2


def _phrases(text: str):
    """The whole line, its clauses (split on , ;) and its parts (also split on or/and)."""
    yield text
    yield from _SPLIT_CLAUSES.split(text)
    yield from _SPLIT_WORDS.split(text)


def warning_terms(items: list[ExtractedItem]) -> list[str]:
    """Key phrases from active warning_signs items, from the document's own bullet
    (source_quote) first and then the plain-language text. One phrase per distinct
    set of content words, in order."""
    out: list[str] = []
    seen: set[frozenset[str]] = set()
    for it in items:
        if it.category != "warning_signs" or it.status == "rejected":
            continue
        for line in (it.source_quote or "", it.text or ""):
            for part in _phrases(line):
                phrase = _LEADING.sub("", part.strip(" \t\n.:")).strip()
                content = _content(phrase)
                if not content or content in seen or not _specific_enough(content):
                    continue
                seen.add(content)
                out.append(phrase)
    return out


def detect_emergency(message: str, extra_terms: list[str]) -> list[str]:
    """Return matched human-readable terms (deduplicated, static terms first).

    A document term matches when every one of its content words is in the
    message, in any order. A term whose words are all covered by a longer term
    that already matched is not reported again."""
    text = _canon(message)
    hits: list[str] = []
    seen_patterns: set[str] = set()
    for label, rx in _COMPILED.items():
        # Alias labels share one pattern: one phrase yields one label.
        if rx.pattern not in seen_patterns and rx.search(text):
            seen_patterns.add(rx.pattern)
            hits.append(label)
    for label in match_terms(message, extra_terms):
        if label not in hits:
            hits.append(label)
    return hits


def match_terms(message: str, terms: list[str]) -> list[str]:
    """The document terms whose content words are all in the message (any order),
    as readable labels, longest term first. A term whose words are all covered by
    a longer term that already matched is not reported again."""
    words = set(_words(message))
    matched: list[frozenset[str]] = []
    out: list[str] = []
    entries = [(term, _content(term)) for term in terms]
    for term, content in sorted(entries, key=lambda t: -len(t[1])):
        if not content or not content <= words or any(content <= m for m in matched):
            continue
        matched.append(content)
        label = _canon(term)
        if label not in out:
            out.append(label)
    return out


_CLAUSE_SPLIT = re.compile(r"[.;,!?:\n]|\bbut\b|\band\b|\bor\b|\bnor\b|\bhowever\b|\balthough\b", re.IGNORECASE)
_NEGATOR_STEMS = frozenset({"not", "deny", "without", "non"})  # no/n't/never fold to "not"
_NEGATION_PREFIX = frozenset(_stem(w) for w in (
    "i", "we", "he", "she", "they", "patient", "have", "has", "had", "do", "did", "am",
    "is", "was"))
_NEGATION_FILLERS = frozenset(_stem(w) for w in (
    "any", "a", "an", "the", "have", "has", "had", "having", "got", "get", "getting",
    "experienced", "experiencing", "noticed", "seen", "felt", "feel", "feeling",
    "sign", "signs", "of"))
_MAX_FILLERS = 2
_NEGATION_TAILS = ((), ("at", "all"), ("though",), ("either",), ("today",))


def _negated_in_clause(words: list[str], content: frozenset[str],
                       term_words: frozenset[str]) -> bool:
    """A clause that mentions the term denies it only when it is exactly
    [I/have/do...] + negator + up to 2 filler words + the term's own words +
    optionally "at all"/"though"/"either"/"today". Any other word ("until",
    "just", "more", "since", "sure", "think", ...) means not negated."""
    if not any(w in content for w in words):
        return True
    n = next((i for i, w in enumerate(words) if w in _NEGATOR_STEMS), None)
    if n is None or not all(w in _NEGATION_PREFIX for w in words[:n]):
        return False
    rest = words[n + 1:]
    for k in range(_MAX_FILLERS + 1):
        if not all(w in _NEGATION_FILLERS for w in rest[:k]):
            break
        for tail in _NEGATION_TAILS:
            end = len(rest) - len(tail)
            if tuple(rest[end:]) != tail or end <= k:
                continue
            if all(w in term_words for w in rest[k:end]):
                return True
    return False


def negated_terms(message: str, terms: list[str]) -> set[str]:
    """The matched terms (labels from match_terms) the patient plainly denies.

    A term is negated only when every clause (split on . ; , ! ? : and the words
    but/and/or/nor/however/although) that mentions its words is a plain denial,
    see _negated_in_clause: a negator, at most 2 filler words, the term, nothing
    else. Hedges, double negatives and qualifiers ("until", "just", "more") fail.
    It errs toward not negated: a missed negation only sends a case to the
    doctor, a false one could let a warning sign through."""
    matched = set(match_terms(message, terms))
    clauses = [_words(c) for c in _CLAUSE_SPLIT.split(message)]
    out: set[str] = set()
    for term in terms:
        label = _canon(term)
        if label not in matched or label in out:
            continue
        content = _content(term) - {"not"}
        term_words = frozenset(_words(term)) - {"not"}
        if content and all(_negated_in_clause(c, content, term_words) for c in clauses):
            out.add(label)
    return out


# ---------- denied danger signs on interview replies (2026-11-17 spec 4.4) ----------

# Labels a denial never clears: a patient who writes about suicide or self-harm
# at all is shown the emergency answer, whatever the wording.
SELF_HARM_LABELS = frozenset({"suicidal", "want to die", "kill myself", "self harm", "overdose"})
# A clause of only these words is a bare "no" (raw words, before negations fold
# to "not": "can't" or "unable" alone is not a "no", it may mean "I can't breathe").
_BARE_NO = frozenset({"no", "nope", "nah"})
# Words that, in any other clause of a reply that denies a danger sign, may take
# the denial back, affirm it, say it comes and goes or point to it again:
# "Well I did", "Yup", "Not true", "Never mind", "it hurts", "until now". Stemmed
# like _words. The reviewers' M4 probes are the test of this list.
_TAKE_BACK = frozenset(_stem(w) for w in """
    yes yeah yep yup ya yah yea yess sure ok okay uh huh mhm hmm actually wait lied lie
    lying liar kidding kid jk joke joking true truth wrong false mind back again anymore
    until till except only unless sometimes whenever other comes come came goes going gone
    happened happen happens happening once twice before later earlier still worse worst
    oops misread meant mean sorry correct correction changed change hold thought do does did
    done there here now got had hurt hurts hurting ache aches aching feel feels felt
    feeling hard difficult struggle struggling bad badly terrible awful horrible severe
    intense unbearable""".split())
# A start verb ("It started an hour ago") takes a denial back unless the clause
# names its own subject first ("the redness started two days ago").
_START = frozenset(_stem(w) for w in "start started starts starting began begin begun".split())
_REFERS_BACK = frozenset(_stem(w) for w in "it its it's that this these those they them one some".split())
_CLAUSE_PARTS = re.compile(r"[.;,!?:\n]|\bbut\b|\band\b|\bor\b|\bnor\b|\bhowever\b|\balthough\b",
                           re.IGNORECASE)


def _term_words(term: str) -> frozenset[str]:
    return frozenset(_words(term)) - {"not"}


def _denies_exactly(words: list[str], term: frozenset[str]) -> bool:
    """[I/have/do...] + negator + up to 2 filler words + exactly `term`'s words,
    each once + optionally "at all"/"though"/"either"/"today"."""
    n = next((i for i, w in enumerate(words) if w in _NEGATOR_STEMS), None)
    if n is None or not all(w in _NEGATION_PREFIX for w in words[:n]):
        return False
    rest = words[n + 1:]
    for k in range(_MAX_FILLERS + 1):
        if not all(w in _NEGATION_FILLERS for w in rest[:k]):
            break
        for tail in _NEGATION_TAILS:
            end = len(rest) - len(tail)
            if end <= k or tuple(rest[end:]) != tail:
                continue
            portion = rest[k:end]
            if len(set(portion)) == len(portion) and frozenset(portion) == term:
                return True
    return False


def _takes_back(words: list[str]) -> bool:
    """A clause that is not a denial of a matched term but may undo one."""
    if any(w in _TAKE_BACK for w in words):
        return True
    # An elliptical inability: "Can't.", "I can't", "unable", "he is not".
    if all(w == "not" or w in _NEGATION_PREFIX for w in words):
        return True
    start = next((i for i, w in enumerate(words) if w in _START), None)
    return start is not None and all(
        w in _REFERS_BACK or w in _STOP_STEMS or w in _NEGATION_PREFIX for w in words[:start])


def denial_possible(message: str, matched: list[str], extra_terms: list[str]) -> bool:
    """The program half of the spec 4.4 rule, checked before any model call: a
    plain emergency ("I can't breathe", self-harm, an affirmed term) never
    waits on the reply judge. True when all hold, else False:

    - no label is a self-harm or suicide label, and none denies itself
      ("can't breathe" has no plain denial);
    - negated_terms finds every label denied in its own clause, and every
      clause that names a label's words denies exactly that label, each word
      once ("No chest pain chest pain" and "he is not breathing" do not);
    - no other clause may take the denial back (_TAKE_BACK, an elliptical
      "can't", "It started ...") and the reply is plain ASCII (look-alike
      letters such as a Cyrillic "I" hide words);
    - detect_emergency, rerun on the reply without the denied clauses, finds
      nothing. Other clauses are allowed ("It is clear mucus").

    `matched` is joined with the backstop's own reading of `message`, so a
    term left out by the caller is still checked. It errs toward the
    emergency answer: a missed denial only shows that answer once more."""
    labels = list(dict.fromkeys([*matched, *detect_emergency(message, extra_terms)]))
    if not labels or not normalize(message).isascii():
        return False
    for label in labels:
        key = _canon(label).casefold()
        if key in SELF_HARM_LABELS or "not" in _words(label):
            return False
    if not {_canon(label) for label in labels} <= negated_terms(message, labels):
        return False
    label_words = [_term_words(label) for label in labels]
    kept: list[str] = []
    for clause in _CLAUSE_PARTS.split(message):
        words = _words(clause)
        if not words or set(_canon(clause).split()) <= _BARE_NO:
            continue
        mentioned = [t for t in label_words if set(words) & _content_words(t)]
        if mentioned:
            if not any(_denies_exactly(words, t) for t in mentioned):
                return False
            continue  # a denied clause: left out of the rerun
        if _takes_back(words):
            return False
        kept.append(clause)
    return not detect_emergency(". ".join(kept), extra_terms)


def _content_words(term_words: frozenset[str]) -> frozenset[str]:
    return frozenset(w for w in term_words if w not in _STOP_STEMS)


def denial_clears(message: str, matched: list[str], extra_terms: list[str],
                  judged_absent: set[str]) -> bool:
    """True when an interview reply's emergency match is a plain denial the
    emergency answer may skip (spec 4.4): denial_possible holds and the reply
    judge listed every matched term as "absent" (`judged_absent`)."""
    if not denial_possible(message, matched, extra_terms):
        return False
    absent = {_canon(t).casefold() for t in judged_absent}
    labels = dict.fromkeys([*matched, *detect_emergency(message, extra_terms)])
    return all(_canon(label).casefold() in absent for label in labels)
