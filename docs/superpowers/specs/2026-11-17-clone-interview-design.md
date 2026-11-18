# Doctor clone interview - design

Date: 2026-11-17. Amends section 4.7 (multi-round intake) of
`2026-11-15-clone-habits-design.md`; everything there not named here stays.

## 1. Intent

When a patient reports a problem, the clone should find out the patient's state
the way this doctor would: ask, listen, ask the next thing that matters, and
stop when it knows enough. The team's words: interactive, "not fixed
questions", questions chosen from the doctor's own habits to collect what the
doctor needs.

Today (4.7) the clone sends a list of up to 4 questions per round, the patient
answers them all in one text box, and a patient who denies a danger sign ("no
trouble breathing") gets the 911 card, which ends the intake. Measured on
2026-11-17 on a copy of the demo database: runny nose, reply "No fever and no
trouble breathing" -> emergency card -> handoff.

## 2. Goals, non-goals, anti-goals

Goals
- One question per round, chosen from what is still unknown.
- What the doctor needs to know is derived from this doctor's habits on similar
  past cases, not from a fixed template.
- The patient sees what the clone has understood so far.
- A denied danger sign no longer ends the interview; an affirmed one still
  shows the emergency answer at once.
- The doctor gets a structured summary on handoff.

Non-goals
- No precomputed per-topic checklists or doctor-side editing of items (later).
- No change to answer grounding, the drug checks, or the close gate (4.4).

Anti-goals (a failure even if it works)
- The clone states something the patient did not say as a known fact.
- A danger sign the patient affirmed, or left unclear, is missed.
- Questions that give advice, or that ask again what the patient already said.

## 3. Approaches considered

- **A, chosen: items from the doctor's habits.** Each case gets a list of
  things to learn, built from the ask and warning habits of the doctor's most
  similar past cases. Each round fills items from the reply and asks about one
  missing item. Adaptive, in the doctor's voice, and the program can check
  coverage and when to stop.
- B: free model interviewer. Most natural, but no program check of coverage,
  stopping or safety.
- C: today's verbatim habits, one per round. Smallest change, still fixed
  questions.

## 4. Design

### 4.1 Interview items (first turn)

On a first-turn message that would today start an intake (ask-first picker
says `reports_symptom`), the server builds the case's items:

1. Take the doctor's past cases (guidance entries) most similar to the
   patient's question by embedding, using the same ranking as advice
   retrieval; up to `INTERVIEW_CASES` (default 4).
2. Collect their non-suppressed `ask` habits and `warning` habits.
3. One small model call (JSON) groups the ask habits into at most
   `INTERVIEW_MAX_ITEMS` (default 5) items:
   `{"label": short noun phrase, "habits": [rule ids]}`. The program checks
   every id is in the input set and every label is at most 6 words with no
   numbers, drug names outside the allowed names (4.5), or advice words.
   Invalid output -> fall back to today's picker (4.7).
4. Every warning term of the collected warning habits whose phrase is not
   already in an item becomes a **safety item** (label = the term, `safety:
   true`), up to `INTERVIEW_MAX_SAFETY` (default 3), ordered by the habit's
   case similarity. Safety items are asked first.

Items already answered in the first message are filled by the extractor
(4.2) before the first question.

Stored on the waiting care-team row in the existing `intake` JSON as
`{"items": [{"label", "habits", "safety", "status", "quote"}], "rounds":
[{"asked", "item", "reply"}]}`; `status` is `unknown | answered | unsure`.
Older `intake` lists (4.7 shape) still read.

### 4.2 Each round

1. **Extract.** The existing reply judge call gains an `items` field: for each
   unknown item, `{"label", "status": "answered" | "unknown", "quote"}`. An
   `answered` item must quote the patient's words; the program accepts it only
   if the quote is a substring of this reply (whitespace and case normalised)
   or of the original question. Otherwise it stays `unknown`. A safety item is
   never filled by the extractor alone: its status follows the judge's
   `warning_signs` for that term (present -> handoff, absent -> answered,
   unclear -> stays unknown and is asked again once, then handoff).
2. **Pick.** The program picks the next item: first unknown safety item, else
   the first unknown item in list order.
3. **Ask.** If one of the item's habits is a question, the clone may use it
   word for word (doctor's question). Otherwise one small call writes one
   question about that item in the doctor's voice; it must pass the existing
   clone-question checks (4.7: one sentence, at most 25 words, no numbers,
   links, advice or suggestion words, allowed drug names only, no repeat) and
   name the item's label words or a habit's words (loose overlap check).
   Failing twice -> the item's first habit question verbatim; no habit
   question -> skip the item.
4. **"I'm not sure" button.** Sends a fixed reply; the asked item becomes
   `unsure` and is not asked again. On a safety item `unsure` means the case
   cannot close (handoff at the end).

### 4.3 Stopping

The interview ends when no item is `unknown`, after `INTERVIEW_MAX_ROUNDS`
(default 6) rounds, or when the patient asks a new question (as 4.3 today).
Then, unchanged from 4.4 and 4.7: answer the original question or close in
the doctor's words if the close gate passes; otherwise hand off. The handoff
body gains the item list ("Known: ...", "Not sure: ...", "Not answered: ...")
above the rounds.

### 4.4 Denied danger signs (backstop on interview replies)

On an interview reply turn only, an emergency match no longer shows the
emergency answer when **all** hold:
- the match is not a self-harm or suicide label (never relaxed);
- a program check finds every matched term inside an explicit negation in its
  own clause ("no", "not", "never", "without", "denies", "don't have");
- the reply judge lists every matched term as `absent`;
- the full backstop, rerun on the reply with the denied clauses removed,
  matches nothing;
- every clause of the reply is an explicit denial naming what is denied
  ("No fever and no trouble breathing."). A bare "No." or "Not really.", a
  hedge ("Well, a little.", "I think") or any other clause, even a harmless
  one ("It is clear mucus."), shows the emergency answer (team decision
  2026-11-17, after review showed hedged retractions cannot be listed).
Anything else (an affirmed term, a disagreement, `unclear`, a retraction such
as "I did not faint. Well I did") shows the emergency answer as today. The
earlier attempt (M4) was reverted after review; this rule set is the
structural form that review asked for, and every probe string kept from M4
stays a regression test.

### 4.5 Patient page

The clarify card shows one question, with the doctor's or the clone's label
as today, and above it a "So far" line listing answered items with the
patient's quoted words shortened (`label: quote`), and "Not sure" items. A
"Not sure" button sits next to Send while an interview question is pending.
The card says "Question 2 of up to 6"-style progress.

### 4.6 Doctor inbox

The waiting and handoff rows show the item list (known, not sure, not
answered) above all rounds.

### 4.7 Resolved details (from the builder check; calls by Claude)

1. **The program drives the interview.** While an item is `unknown` and rounds
   remain, the agent model is not called: the program extracts, picks and asks
   (judge call + question call). The agent runs once at the end with all
   rounds in context and `clarify` not allowed (answer, close or not_covered).
   `INTERVIEW_MAX_ROUNDS` replaces `CLARIFY_MAX_ROUNDS` for item-shape intakes.
2. **Judge input.** On interview turns the judge's term list is the safety
   items' terms plus every term (doctor, document or static backstop label
   text that matched) found in this reply, so "no trouble breathing" can be
   judged.
3. **Close gate.** A matched warning term counts as absent at the close gate
   exactly when it meets the 4.4 rule (program negation in its clause and the
   judge says `absent`); the earlier "omitted term blocks" rule stays. So a
   patient who denied every danger sign can be closed; a sign that is present,
   unclear or `unsure` hands off.
4. **An item with no usable question** becomes `skipped`; it is listed as
   "Not answered" and does not stop the interview from ending.
5. **Re-ask.** Each item has an attempt count (max 2). A second attempt is a
   clone-written rephrase; the no-repeat check compares against the exact
   earlier wording, so a rephrase passes. If the rephrase fails its checks or
   is still unclear, the item becomes `unsure`. Every attempt is a round.
6. **No items.** When the similar cases give no ask habits and there are no
   safety items (for example Dr. Ting Kai), the case uses today's 4.7 flow
   unchanged (doctor habits plus clone-written questions).
7. **Safety item sources.** Warning terms of the similar cases' warning habits
   always become safety items. The patient's document warning terms
   (`warning_terms(items)`) become safety items only when the build call marks
   them relevant to this complaint (`relevant_document_terms` in its JSON; the
   program keeps only terms that are in the list it sent). At most
   `INTERVIEW_MAX_SAFETY` in all. Static backstop patterns are not turned into
   questions, and an unasked document term is not lost: the backstop still
   fires on any message and the close gate still checks every term.
8. **Interview reply turn** means: this patient-doctor pair has a waiting row
   whose `intake` has the item shape, not expired. The backstop still runs
   first; when it matches, the route looks up that row before deciding whether
   4.4 relaxes it. A first message, an old-shape intake, or an expired row is
   never relaxed.
9. **"Not sure"** is a request field `not_sure: true` on `/api/ask` with the
   current `round` number. The server stores the patient message "I'm not
   sure.", marks the asked item `unsure`, makes no judge call, and picks the
   next item. A `round` that does not match the row (double tap) returns the
   current state without a new round.
10. **Items freeze** at the first turn. A habit suppressed later is not used
    as a verbatim question. The "So far" line shows the stored quote verbatim;
    a quote over 60 characters is cut at a word boundary with a visible "...".

## 5. Evaluation

The PortalPal test split (55 conversations) is rerun with the interview on, a
single run with the gpt-4o-mini judge, comparing to the 2026-11-16 numbers:
warning handoff, paraphrase, benign closed, asks per conversation, and drug
violations. Budget about $1. Simulated patient replies answer only what is
asked (replies.json gives the facts).

## 6. Constraints

Same as the 2026-11-15 spec: OpenAI gpt-4o-mini for the clone and the small
calls, fail closed, repo docs in English, no em dash, never write to
`aftercare-new.sqlite3` from tests or evals.

## 7. Left to the builder

The exact embedding query for similar cases, item JSON prompt wording, the
loose overlap rule in 4.2 step 3, and CSS of the "So far" line (must match the
existing clarify card).

## 8. Decided by the team

Approach A; one question per round; derive items from the doctor's habits on
similar cases; show what the clone understood; fix the denied danger sign.
On an interview reply, every clause must be an explicit denial naming what
is denied before a matched danger sign is relaxed (2026-11-17, after review
showed hedged retractions cannot be listed).

## 9. Calls made by Claude

Six rounds at most; five items plus three safety items at most; the "Not
sure" button; one more small call per round (about $0.002 per conversation).
