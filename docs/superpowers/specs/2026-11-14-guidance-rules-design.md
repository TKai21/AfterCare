# AfterCare - Guidance Rules and Conflict Detection Design

Date: 2026-11-14
Status: approved by Kai (go-ahead without a question round)
Builds on: `2026-11-13-doctor-clone-design.md`, `2026-11-13-multi-doctor-clones-design.md`, `2026-11-13-drug-label-library-design.md`.
Part 1 of 3 of the clone upgrade. Part 2 is draft-and-approve (the clone drafts answers in the doctor's inbox), part 3 is calibrated abstention (conformal risk control on when the clone answers).

## 1. Intent

Today the clone's memory is a list of whole question-and-answer entries, matched to a patient question by one cosine similarity per entry. That works when the patient asks almost the same question again. It fails when the answer needs two things the doctor said in two different answers, and a long answer that covers several topics is matched as one blurred vector.

This change turns each answer into the doctor's **rules**: short, standalone statements, each tied to the exact words of the doctor that support it. The clone retrieves rules, not entries, so it can combine rules from several answers, and every rule still points at the doctor's own words. Once advice is rules, the app can also **find conflicts**: a new rule that disagrees with one of the doctor's earlier rules, or with an approved drug label, is shown to the doctor before it misleads a patient.

Success: (1) a question whose answer spans two entries is answered, citing both; (2) a long multi-topic entry is found for a question about only one of its topics; (3) when the doctor saves advice that contradicts earlier advice or a label, the guidance page says so, naming both sides.

## 2. Goals

1. Each guidance entry is split into rules by the model when it is saved or edited.
2. Retrieval scores rules. An entry without usable rules is scored as a whole, as today.
3. The agent sees each rule as its own piece of evidence and cites rules.
4. Conflicts between a new rule and the same doctor's other rules, and between a new rule and approved drug label passages for the entry's drug tags, are detected and listed on the guidance page.
5. The doctor sees the rules the clone read from each entry, sees open conflicts, and can dismiss a conflict.

## 3. Non-goals and anti-goals

Non-goals: the doctor editing rules one by one (they edit the answer, rules are rebuilt); conflicts with a patient's discharge document (the agent already handles "Sources disagree" at answer time, per patient); a knowledge graph or rule engine; fine-tuning.

Anti-goals, failure even if it works:

- **A rule says something the doctor did not.** Every rule carries a quote that must be an exact span of the doctor's answer, and every number in the rule must appear in that quote. The quote, not the rule, is what grounds numbers in the clone's answer and what the patient sees.
- **Rules widen who advice applies to.** The applicability gate (scope, drug and condition tags) stays at entry level and runs before rules are scored. A rule never reaches a patient its entry could not reach.
- **The memory breaks when rule extraction fails.** No model, a model error or an unreadable reply keeps the entry searchable as a whole, as today. The emergency path is untouched.
- **Conflict detection blocks the doctor.** Saving never waits on a human step and never fails because the conflict check failed; conflicts are advisory.

## 4. Approaches

- **A. Rules as quoted spans plus a restatement (chosen).** The model returns `{statement, quote}` pairs; the server checks the quote is in the answer and the statement's numbers are in the quote. Retrieval and citations work per rule. Keeps the existing safety model (numbers grounded in the doctor's words) exactly.
- **B. Sentence chunking, no model.** Split answers into sentences and embed each. Cheap and deterministic, but sentences are not standalone ("Take it with food." loses which drug), so retrieval quality is worse and conflicts cannot be judged without the model anyway.
- **C. Structured rule records (condition, drug, action, threshold) in a rule engine.** Most precise for composition, but brittle extraction, a schema that does not fit free-text advice ("call the clinic if you see clots"), and nothing to ground against. Rejected.

A keeps B's fallback (an entry with no rules is scored whole) and gets most of C's benefit from the model composing rules at answer time.

## 5. Data

New table `guidance_rules`:

| column | type | notes |
|---|---|---|
| id | int PK | |
| guidance_id | FK doctor_guidance, cascade | |
| doctor_id | FK users, cascade | denormalized for the per-doctor query |
| position | int | order in the answer |
| statement | text | standalone restatement, at most 400 characters |
| quote | text | exact span of the answer that supports it |
| embedding, embedding_model | JSON, varchar | embedding of the statement |

New table `guidance_conflicts`:

| column | type | notes |
|---|---|---|
| id | int PK | |
| doctor_id | FK users, cascade | |
| rule_id | FK guidance_rules, cascade | the newer rule |
| other_rule_id | FK guidance_rules, cascade, nullable | the doctor's earlier rule |
| label_rxcui, label_title, label_url, label_quote | nullable | the drug label passage, when the other side is a label |
| explanation | text | one sentence from the model |
| status | varchar | `open` or `dismissed` |
| created_at | datetime | |

New column `doctor_guidance.rules_status`: NULL (not tried, no model), `ready`, `failed`. Added to `ADDED_COLUMNS` for existing databases and to `sql/schema.sql`.

Rebuilding an entry's rules deletes its old rules, which cascades to their conflicts on both sides. Conflicts are recomputed for the rebuilt entry only.

## 6. Indexing pipeline

`index_entry(session, llm, store, entry)` replaces `embed_entry` at every call site (answer, patch with changed text, the retry endpoint, the demo seed):

1. Embed the whole entry, as today (the fallback vector).
2. Extract rules: one JSON-mode chat call with the question, the answer and the tags. The prompt asks for 1 to 8 rules, each standalone (names the drug or condition, says when it applies) and each with the shortest exact quote that supports it.
3. Validate each rule: statement non-empty and at most 400 characters; quote found in the answer after collapsing whitespace (the stored quote is the answer's own text); every number token of the statement (the agent's `number_tokens`) in the quote's tokens. Invalid rules are dropped. Zero valid rules, an unreadable reply or a model error sets `rules_status = failed` and the entry stays searchable as a whole.
4. Embed the statements in one batch. A failed batch is the same as a failed extraction.
5. Detect conflicts (section 8). Its failure is logged and leaves no conflicts; it never changes `rules_status`.

An inactive entry is still indexed (it may be re-activated) but its rules are never retrieved and never used as the other side of a conflict.

## 7. Retrieval and the agent

`retrieve` keeps its gate and returns hits `(entry, rule or None, score)`:

- For each eligible entry: if `rules_status == ready` and its rules have embeddings of the current model, score each rule's statement; else score the entry's own embedding.
- Keep hits at or above `GUIDANCE_MIN_SIMILARITY`, best first, at most 3 rules per entry, at most 6 hits in total.

The agent numbers each hit as its own evidence. For a rule: key `("rule", rule.id)`, kind `doctor`, the same patient-facing title as today, `content` = the quote (the patient's source shows the doctor's own words), `check_text` = the quote. The model sees `rule`, `my_words` (the quote) and `question_answered` (context only). The system prompt says `my_words` wins when it and `rule` differ, and that two pieces of my earlier advice that disagree must be reported as differing with `add_care_team_question` (reason "Sources disagree"). A whole-entry hit is numbered and shown exactly as today. `guidance_ids` still lists entry ids, so cited counts do not change.

## 8. Conflict detection

For a freshly indexed entry E with rules R:

- Own-rule candidates: active, ready rules of the same doctor from other active entries, with cosine to some rule in R at or above `CONFLICT_MIN_SIMILARITY` (0.5), top 3 per rule of R. Pairs whose entries can never apply to the same patient are skipped: both patient-scoped for different patients.
- Label candidates: for each drug tag of E whose name matches an approved `DrugLabel.ingredient` (whole word, case-insensitive), the top 2 label passages of that RxCUI from the library store for each rule of R (only passages from that label's document).
- One JSON-mode chat call lists R and the numbered candidates, and asks for `{"conflicts": [{"rule": i, "candidate": j, "explanation": "..."}]}` where a conflict means a patient could not follow both. Out-of-range indices are dropped. Each conflict becomes a `guidance_conflicts` row with status `open`.

No candidates means no model call.

## 9. API and UI

- `guidance_views` adds `rules_status`, `rules: [{id, statement, quote}]` and `conflicts: [{id, explanation, other: {kind: "advice", guidance_id, question, quote} | {kind: "label", title, url, quote}}]`, listing open conflicts on either side of the entry.
- `POST /api/doctor/guidance/{id}/embed` (the retry) runs the whole pipeline. A conflict is dismissed with `POST /api/doctor/conflicts/{id}/dismiss` (404 for another doctor's).
- The Guidance tab badge shows the number of open conflicts. Each entry shows a "How your clone reads this" disclosure listing its rules, a "Rules could not be read" note with Retry when `failed`, and, for each open conflict, a warning block naming the other side, the explanation, and Dismiss.

## 10. Testing

`FakeLLM` gains prompt routing: a test (or conftest default) registers a responder for the rule extraction and conflict prompts, so the existing scripted tests are not consumed by indexing calls. The default responder returns one rule per sentence with the sentence as the quote, and no conflicts. New tests cover validation (quote not in the answer, invented number), fallback to whole-entry scoring, the gate running before rules, the 3-per-entry cap, rule evidence and citations, conflicts with own rules and with labels, dismissal and cross-doctor 404s, rebuild cascades, and the frontend rendering via the existing syntax tests.

## 11. Calls made by the builder

- Rule limit 8, statement limit 400 characters, 3 rules per entry, 6 hits, conflict similarity 0.5: starting values, to be revisited by part 3's calibration.
- Indexing runs inline in the request, like embedding today; no background job.
- Conflicts with the discharge document stay with the agent's existing per-patient rule.
