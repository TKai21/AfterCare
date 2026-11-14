# AfterCare - Clone Drafts (Draft and Approve) Design

Date: 2026-11-14
Status: approved by Kai (go-ahead without a question round)
Builds on: `2026-11-14-guidance-rules-design.md` (part 1 of 3). Part 2 of 3 of the clone upgrade.

## 1. Intent

A question reaches the doctor's inbox because the clone could not answer it for the patient on its own. Often it could get most of the way: the patient's document, the doctor's earlier rules and the drug labels cover part of it. Today the doctor still writes every answer from scratch.

With this change the clone **drafts** the answer in the inbox. Each sentence of the draft shows where it came from, and a sentence the sources do not support is marked for the doctor to check. The doctor edits and sends. What the doctor changes is recorded, so the app can (1) show how much of the clone's drafts the doctor keeps, as a number that should rise over time, and (2) feed the doctor's recent corrections back into the next drafts.

Success: the doctor sends a drafted answer with few edits; the inbox shows "drafts sent, sent unchanged, average kept"; a correction the doctor made shows up as a style or content change in later drafts.

## 2. Goals

1. "Draft with my clone" on each open question; the draft fills an empty answer box.
2. Each draft sentence lists its sources (document paragraph, the doctor's rule, a label or library passage) or is flagged "check this".
3. Numbers in a sentence must appear in its cited sources, or the sentence is flagged.
4. Sending an answer records the draft it started from and how much of it was kept.
5. The next drafts see the doctor's three most recent corrected drafts as examples.
6. Draft statistics on the doctor page.

## 3. Non-goals and anti-goals

Non-goals: drafting automatically for every new question (a model call per question nobody may open); fine-tuning on edits; patient-visible drafts.

Anti-goals:

- **A draft reaches a patient unreviewed.** Drafts exist only on the doctor page. Only the doctor's send creates an answer.
- **A draft leaks another patient's advice.** Guidance for drafts goes through the same retrieval gate as the patient's own clone: general entries the patient's document matches, and patient entries for this patient and document only.
- **Flags that cry wolf or stay silent.** A sentence is "supported" only when it cites evidence and every number it states is in that evidence; otherwise it is flagged. The doctor sees the flag, never a hidden judgment.

## 4. Approaches

- **A. A drafting call with numbered evidence and per-sentence citations (chosen).** Reuses the agent's numbering and number grounding, shows support sentence by sentence, and needs one model call.
- **B. Run the patient-facing agent and show its answer.** Already exists, but the agent refuses (`not_covered`) exactly on the questions in the inbox, and its rules forbid the things a doctor's draft must contain (a judgment from the doctor). Rejected.
- **C. Fine-tune on (draft, edit) pairs.** Too little data, cannot cite. Rejected, as in the clone design; corrections are used as in-context examples instead.

## 5. Data

New table `answer_drafts`: `id`, `question_id` (FK, cascade), `doctor_id` (FK users, cascade), `sentences` (JSON: `[{text, sources: [number], supported, problem}]`), `evidence` (JSON: `[{number, kind, title, quote}]`), `text` (the sentences joined), `created_at`, and, filled on send, `sent_answer`, `kept` (0 to 1), `sent_unchanged`, `sent_at`.

A question has at most one unsent draft per doctor: a new draft replaces it.

## 6. Drafting

`POST /api/doctor/questions/{id}/draft`:

- 404 like answering (another doctor's question), 409 if answered, 503 without a model, 502 on a model failure or an unreadable reply.
- Evidence, numbered from 1: the patient's document paragraphs; the doctor's guidance from `retrieve` for the question (rules give their quote, whole entries their answer); library passages for the question, with the patient's approved label RxCUIs.
- Corrections: the doctor's three most recent sent drafts with `kept < 0.97`, as (draft, what I sent) pairs.
- One JSON-mode call. The model writes as the doctor to the patient, short and plain, and returns `{"sentences": [{"text": "...", "sources": [n]}]}`.
- Validation: drop empty sentences; unknown source numbers are removed; `supported` = has sources and every number token of the text is in the cited evidence (the agent's `number_tokens`); `problem` is `no_source` or `number` when not supported. No sentences at all is a 502.

## 7. Sending and statistics

`AnswerBody.draft_id` (optional). On a successful answer, a draft of this question and doctor gets `sent_answer`, `sent_at`, `kept` = word-level `difflib.SequenceMatcher` ratio between draft text and answer, and `sent_unchanged` = equal after collapsing whitespace. A draft id of another question is ignored.

`GET /api/doctor/draft-stats` returns `{sent, unchanged, mean_kept, weeks: [{week_start, sent, mean_kept}]}` over the doctor's sent drafts, last 8 weeks.

`GET /api/doctor/questions` adds `draft` (the unsent draft or null) to each open question.

## 8. UI

On an open question card: a "Draft with my clone" button (with a spinner while drafting, "Redraft" once a draft exists). The draft panel lists the sentences; a supported sentence carries small numbered source markers whose titles and quotes open in a popover-free list below; a flagged sentence is tinted with the apricot warning color and says "Check this: not in your sources" or "Check this: a number is not in the cited source". "Use draft" copies the text into the answer box (it fills an empty box automatically).

The Questions tab aside shows the stats as three numbers once at least one draft was sent.

## 9. Testing

Unit: validation (unknown sources removed, number flag, no-source flag), evidence assembly respects the gate, corrections pick only edited drafts. Routes: draft lifecycle (create, replace, 404/409/503/502), send records kept and unchanged, stats, questions list includes the draft. Frontend via the syntax tests and a browser check.

## 10. Calls made by the builder

- Three correction examples, 0.97 threshold, 8 weeks of stats.
- Draft on request only.
- The doctor's guidance limit for drafting is the clone's (6 hits).
