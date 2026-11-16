# Doctor clone habits, ask-first turns and safety hardening - design

Status: draft for team review. Base branch: `clone-rules` (team `main`).

## 1. Intent

The doctor's goal: the clone should fully remember **how this doctor practices** (which drug they reach for first, and which questions they always ask before advising) so that the doctor stops answering the same kinds of message again and again.

Today the clone only stores *medical advice* (rules read from the doctor's answers, gated by drug and condition tags). Two kinds of evidence say that is not enough:

- The clone preference evaluation (`eval/clone/rules-baseline/`): the "ask first" dimensions D3 and D4 score 0 for every doctor, with or without memory. The clone never asks the patient anything back.
- A real physician's replies (PortalPal, section 7): 87 of 100 replies ask the patient questions first, 84 send them to schedule, 79 say when to go to the emergency department. Almost none give medical advice. This doctor's habits are general ("when a patient reports a new symptom, ask about recent medication changes"); the current gate refuses any general rule without a drug or condition tag, so these habits cannot be stored at all.

The same evaluation found two production blockers that this work also fixes:

- **Unprescribed drugs.** All 3 flagged answers in the rules baseline come from memory: the clone tells a patient without naproxen that "I prefer naproxen" or "you can use it as the first-line treatment". The confidence gate did not stop them. Nothing in the server checks drug names.
- **Retrieval floor.** A spike over the 63 case-doctor pairs (scratch script, results in section 5.3) shows embedding retrieval already ranks a rule of the right dimension first in 43 of 63 pairs and in the top 6 in 60 of 63. Retrieval misses (30 of 63) come from the fixed cosine floor 0.45, not from ranking.

## 2. Goals, non-goals, anti-goals

Goals:

1. **Habit rules.** A doctor's answers are read into two kinds of rule: *advice* (as today) and *habits* (what to ask, how to close, which symptoms need the doctor). Habits apply to all of that doctor's patients, need no tags, and may carry no medical content.
2. **Ask-first turns.** The clone can reply with a new kind `clarify`: the habit questions, before it advises. On the patient's reply it either **closes** the case itself, the way the doctor closes (schedule, when to seek urgent care), or **hands it off** to the doctor with the questions and answers summarised. It closes only when every safety condition holds (section 4.4).
3. **Strict drug names.** No delivered text may name a drug unless the patient is prescribed it or a cited paragraph of their discharge document names it.
4. **Retrieval floor redesign**, chosen on tuning data and checked on held-out data.
5. **Measured, not assumed.** The preference evaluation, the calibration question set and a new PortalPal evaluation (real physician) all run before and after, with the numbers in the README.

Non-goals:

- A reranker for doctor memory (the spike showed it does not fix the misses; section 5.3).
- More than one ask-first round per patient question.
- The doctor filling in a habit form (rejected approach, section 3).
- Draft UI changes beyond showing the handoff summary in the inbox.
- Voice messages, non-English, real doctor data.

Anti-goals (failure even if it runs):

- **The clone closes a case whose reply holds a warning sign.** Closing must fail closed: any doubt hands off.
- **A habit rule carries medical content** (a dose, a drug name, a medication change, a diagnosis) and reaches patients without the tag gate.
- **A patient sees a drug they are not prescribed** that their own document does not name.
- **Thresholds tuned on test data**: floor and margins are chosen on the tuning split only.
- **PortalPal content reaches a dev or production database** or is presented as a real AfterCare doctor. It is evaluation data under CC BY-NC 4.0, run in a temporary database, attributed.

## 3. Approaches considered

How habits are stored (the team chose **1**):

1. **Habits are a second kind of rule, read from the doctor's own answers** by the existing rules reader. No tags; a program check limits their content; shown on My guidance where the doctor can edit or delete them. Reuses quotes, citations and conflict checks; the doctor does no extra work and the clone learns more with every answer. Risk: the reader's accuracy, so the doctor reviews and the evaluation measures it.
2. A habit form the doctor fills in. Clear and auditable, but it is the extra doctor work this project exists to remove, and doctors are often unaware of their own habits.
3. The doctor's recent answers as few-shot examples. Cheapest, but uncitable, unreviewable, and it carries other patients' conversations into this patient's answer (privacy). Excluded.

What happens after the patient answers the clone's questions (the team chose **c**):

- a. The clone always closes. Least doctor time; misses warning signs.
- b. The clone always hands off with a summary. Safest; the doctor still reads every case.
- c. **Close when every safety condition holds, otherwise hand off with a summary.**

Retrieval (decided from the spike): keep embedding ranking, replace the fixed floor (section 5.3). A cross-encoder reranker was measured and dropped: own-dimension rule at rank 1 in 41 of 63 pairs against 43 for embedding, top 6 in 62 against 60, with scores no easier to threshold.

## 4. Design

### 4.1 Habit rules

`guidance_rules` gains `kind`: `advice` (default, every existing row) or `habit`, and for habits `habit_type`: `ask` (a question the doctor asks in a situation), `close` (how the doctor ends a case: schedule, self-care that names no drug, when to call) or `warning` (a symptom or change for which the doctor wants to be told or wants urgent care), plus `warning_terms` (JSON list, `warning` only: the symptom phrases, each an exact span of the quote).

The rules reader (`RULES_PROMPT`, `validate_rules`) returns both kinds from one call. A habit must still quote the doctor's exact words. On top of the existing checks, a habit rule is **dropped** (not stored, not shown) if its statement or quote:

- contains any number other than 911;
- names any drug in the drug lexicon (section 4.5);
- matches the medication-change pattern (`MEDICATION_CHANGE` in `agent.py`), except the question form "recent changes in your medications" and close equivalents, which `ask` habits may carry (builder: a narrow allow pattern with tests).

A habit has no tags and passes the gate for every patient of that doctor, whichever entry it was read from (a patient-scope entry included: habits are learned from answering individual patients). **A habit's evidence is its quote only**: the clone sees the quote and the statement, the patient's source card shows the quote, and the whole answer it came from is never shown or given to the clone as habit evidence (unlike advice rules, which see the full advice). The habit validator also drops a habit whose quote contains a person's name (the patient's or doctor's display name, or a capitalised name after a greeting such as "Hi"/"Dear"; builder may extend) so nothing about the original patient travels with it.

Saving a general answer without tags is now allowed (the 422 `GENERAL_TAG_ERROR` in `routes_doctor.py` is lifted). Its advice rules still never pass the gate (an untagged general entry is refused, as today); only its habits are used. The answer form says so when General is ticked and no tag is set. Habit rules are **not** searched by similarity: all of the doctor's habit rules are given to the clone on every turn, deduplicated, up to `HABITS_IN_CONTEXT` (default 30, most recently updated first). Rationale: a doctor has few habits, and the floor problem in section 5.3 would hide them.

My guidance shows habits in their own section ("How I respond"), grouped by type, each with its quote; the doctor can delete a habit (stored as a suppressed rule so a re-read of the same answer does not bring it back) or edit the answer it came from. The conflict check runs on habits against the doctor's other habits only (never against drug labels).

Existing entries are re-read the next time they are edited or retried, as today; a one-off command re-reads all entries of one doctor (builder: `manage_knowledge.py` or a script; it costs one model call per entry).

### 4.2 The `clarify` kind

New answer kind `clarify`. Output contract addition: `{"kind": "clarify", "answer": "...", "citations": [...], "care_team_question": null}`. Validation (program, in `validate`):

- cites at least one habit rule of type `ask`, and only habit rules;
- every sentence ends with "?" except at most one short lead-in sentence (for example the doctor's own opener);
- no numbers, no drug names (section 4.5), no medication-change wording;
- at most 4 questions (builder may tune with the evaluation);
- not allowed when the previous assistant message in this conversation was a `clarify` (one round only), and never when the backstop matched emergency terms.

The system prompt gets one rule: when the patient reports a symptom or problem and the doctor's `ask` habits apply, ask those questions first with kind `clarify`, citing them, unless the document or the doctor's advice already answers the question.

The confidence gate does not score `clarify` (it delivers no medical fact; the program checks above stand in for it). A `clarify` message is stored with `payload.kind = "clarify"` and `payload.pending = true`; the Ask page shows it as the clone's questions with the doctor's habit as the source, and the input box keeps working as usual.

### 4.3 The reply turn

When the latest assistant message in this patient-doctor conversation is a pending `clarify`, the next patient message is its reply. The turn runs as usual (backstop first, prefetch, agent) with the original question, the clarify questions and the reply in context, and `clarify` not allowed. The agent may reply:

- `answer`, citing as usual (for example the document now answers it);
- `close` (new kind): the doctor's close, citing at least one `close` habit, with the same program checks as an answer plus the drug check;
- `not_covered`, which hands off.

The pending flag is cleared on the clarify message after any reply turn.

If the reply is not an answer to the questions but a new question, the agent answers the new question as usual and the original symptom is handed off ("Patient did not answer the clone's questions").

A pending `clarify` also creates a care-team question with status `waiting_on_patient` for this doctor, holding the original question and the clone's questions. The doctor's inbox shows it in a separate "Clone is asking" list that is not counted as open work. It becomes `open` (a normal handoff, reason "Patient did not answer the clone's questions") when it is older than `CLARIFY_TIMEOUT_HOURS` (default 24), computed when the inbox is read, so no scheduler is needed. A reply turn turns it into the handoff (`open`) or, when the clone closes, `resolved` with the close text as its answer. So a reported symptom always reaches the doctor unless the clone closed it.

### 4.4 When the clone may close (fail closed)

The server, not the model, decides whether a `close` is delivered. It is delivered only if **all** hold; otherwise the turn becomes a handoff:

1. the emergency backstop matched nothing in the reply (`detect_emergency` with the document's warning terms, as today);
2. **no warning sign is present**, in the reply or in the original question. Warning terms are the doctor's `warning` habit terms and the document's warning terms (`warning_terms(items)`), matched with the backstop's own phrase matcher (`safety.py`). A matched term counts as absent only if **both** agree: (a) a program check finds it inside an explicit negation in the same clause ("no", "not", "denies", "without", "never had" directly governing it; builder writes the pattern with tests, and does not change the backstop's own matching), and (b) the model, in the same reply turn, lists that term as `absent` in a structured field `warning_signs: [{"term", "status": "present" | "absent" | "unclear"}]` covering every matched term. Any disagreement, `unclear`, or missing term means present. The first-message case: a warning sign in the original question still allows `clarify`, but the case can never close (it is handed off after the answers, so the doctor gets the answers too);
3. the reply does not report a medication change, a new medication, or a dose taken differently: the medication-change pattern on the patient's text, plus the drug lexicon on the reply, where a drug the patient is **prescribed** (the allowed names of section 4.5 without the patient-message part) does not block a close and any other drug name does;
4. the agent's output passes validation and the drug check;
5. the confidence gate passes when it is on (a `close` is scored like an answer).

A **handoff** adds a care-team question for this doctor with reason "Needs doctor after questions" and a structured body: the original question, the clone's questions, the patient's reply, and which condition stopped the close. The patient sees the fixed handoff sentence. The doctor's inbox shows the three parts separately, and "Draft with my clone" gets them as context.

Emergency terms in the reply still produce the fixed emergency answer first, as today.

**Negation on reply turns (team decision, review item M4).** Only on a reply turn (a pending clarify exists in this patient-doctor conversation), the backstop drops a matched term, static pattern or document warning term, that `safety.negated_terms` finds plainly denied (every clause naming the term's words names all of them once as a negator, at most 2 allowed fillers, the term and an optional allowed tail; and with those clauses removed, the unchanged backstop no longer matches the term, so a synonym such as "soaking through the bandage" or the term said again keeps it; and the reply holds no retraction word such as "yes", "actually", "wait", "I did" or "I lied"), so "No fever and no chest pain." to "Do you have a fever?" is not an emergency. A term that is not negated, or any doubt, still gives the emergency answer; every other message uses the backstop unchanged, and so does a reply whose waiting question is no longer `waiting_on_patient`. A term whose own words hold a negation ("can't breathe", "bleeding that won't stop", a document's "cannot keep fluids down") is never treated as denied, and neither is a self-harm or suicide label (`safety.NEVER_RELAXED`: suicidal, want to die, kill myself, self harm, overdose), so "I don't want to die" is still an emergency. A denied term still counts for condition 2: it joins the warning terms the model is given, and the model must list it as "absent" in every entry.

### 4.5 Strict drug names

A **drug lexicon** file `data/drug_names.json`, committed, built once by a script from RxNav (`/REST/Prescribe/allconcepts.json?tty=IN+BN`: about 5,900 ingredients and 4,100 brands, public, no key), with each name's RxCUI and, for brands, ingredient RxCUIs (builder: one related-concepts call per brand, cached; or the Prescribe RXNCONSO release if simpler). Names are normalised like guidance tags. **Excluded** from the lexicon: every nutrient name in the food table, and a reviewed list of common words and diet terms that RxNorm also lists (measured: water, calcium, potassium, sodium, caffeine, oxygen, vitamin D, iron, magnesium, glucose, zinc, plus brand collisions such as "Today" and "Alli"). The exclusion list is a committed file with a test that every food-table nutrient is in it.

**Allowed names for a turn** (team decision): names in the patient's non-rejected medication items and the ingredient and brand names of their resolved `document_drugs` RxCUIs; names that appear in a document paragraph the answer cites; and **names the patient used in this turn's message** (with the lexicon's brand and ingredient equivalents, so "Tylenol" allows "acetaminophen"). So "Can I take aspirin?" may be answered with "no aspirin while taking naproxen" from the doctor's advice, while the clone volunteering a drug the patient neither takes nor asked about (rules baseline c02, c04) is blocked. Names inside cited doctor advice are not allowed by that alone. The calibration questions keep their labels; any that change outcome are listed in the results.

**Check**: every delivered text (`answer`, `clarify`, `close`, rule 7 general information) is scanned for lexicon names as whole words; any name not allowed fails validation with an error naming it, which uses the existing repair round; a second failure falls back to the doctor. The fixed emergency and handoff sentences are not checked (they are fixed text).

The lexicon is also used by the habit validator (4.1) and the close conditions (4.4). The preference evaluation's `KNOWN_DRUGS` list stays as the evaluation's own independent check.

### 4.6 Retrieval floor

Advice-rule retrieval keeps embedding ranking after the gate. The fixed floor `GUIDANCE_MIN_SIMILARITY` is replaced by: the top `GUIDANCE_TOP_K` rules (default 6, as today's limit) whose score is at least `GUIDANCE_FLOOR` (low absolute floor) and within `GUIDANCE_MARGIN` of the best score. Defaults are chosen by a sweep on the **tuning** split only (section 5.3). Both the prefetch and the tool use it.

## 5. Evaluation

### 5.1 Data and splits

| Set | Tuning | Test |
|---|---|---|
| Calibration questions (`eval/clone_questions.json`, 93: 62 answer, 31 escalate) | its `calibration` split | its `test` split |
| Preference evaluation (25 cases, 3 synthetic doctors) | - | all |
| PortalPal (100, one physician; 60 rows are the doctor's memory) | 15 rows | 25 rows, see 5.4 |

All runs use the real library index (the 49 manifest documents). The calibration runs on `clone-rules` used an empty index (README); they are redone.

### 5.2 Order of runs

1. Baselines exist: preference (`eval/clone/rules-baseline/`), calibration v2 (redo with the real index first, as the true baseline).
2. After each part (drug check, floor, habits and clarify), rerun the preference evaluation and the calibration set; recalibrate the confidence gate last, on the calibration split.

### 5.3 Retrieval floor sweep

Retrieval-only (embeddings, no chat model), on the calibration split: for each question that relies on a doctor's advice (its `facts` come from guidance), whether a rule of that entry is returned; for each escalate question, how many advice rules are returned. Pick the floor and margin that keep every answerable question's rule and return the fewest rules on escalate questions. Report the same numbers on the test split and on the preference cases (where the spike measured, with the old floor: own rule ranked first 43 of 63, top 3 51, top 6 60).

### 5.4 PortalPal evaluation (new)

Source: `PortalPal-AI/Patient-Message-Response-Drafting` (Seegmiller et al., 2026, arXiv 2601.11344), CC BY-NC 4.0, 100 rows of a semi-synthetic patient message, a de-identified EHR summary, and one physician's reply. Committed under `eval/portalpal/` with the citation and licence.

- **World**: one evaluation doctor ("Portal physician", not a real name). Each row's EHR summary becomes that patient's confirmed document (problem list and active medications as items), in a temporary database like the preference evaluation.
- **Memory**: the physician's replies to the memory rows are saved as that doctor's answers (general scope with the row's main condition as tag, builder may choose) and read into rules, as the product would. The evaluation reports which habits were read, for review.
- **Test turn 1**: each test message to the clone. Measured: kind is `clarify`; recall of the physician's own question categories for that message (duration, associated symptoms, medication changes, triggers, history; a judge maps both sides to categories); no medical advice delivered.
- **Test turn 2**: two scripted replies per test message, written once and committed, never generated during a run: a **benign** reply (answers the questions, nothing the physician's reply names as a reason to escalate) and a **warning** reply (mentions one sign the physician's own reply names, such as worsening pain, fever or vomiting). Expected: benign closes with the physician's close (schedule; when to seek urgent care); warning hands off. Expected labels come from how the reply was written, not from a model.
- **Metrics**: ask-first rate and question recall; close rate on benign replies (doctor time saved); **handoff rate on warning replies (target 100%; any miss is a failure to report and fix)**; close fidelity (judge: does the close match the physician's close for that message); drug check violations (target 0). Each with memory against without.
- **Rule 9 scope**: a discharged patient reporting a new symptom is in scope (the clone asks and routes it); only questions unrelated to the patient's health stay out of scope. The system prompt's rule 9 is reworded accordingly and covered by tests.

### 5.5 Preference evaluation updates

- D3 and D4 judge items accept a `clarify` turn's questions as the first ask (today they read the answer only).
- Report `clarify`, `close` and handoff counts per doctor and dimension.
- Synthetic memory may now produce habit rules; the retrieval check reports habits separately.

## 6. Constraints

- Everything new keeps the existing order: backstop first, emergency never waits on a model call or on memory.
- New columns are added on startup for SQLite and MySQL like `rules_status`; `sql/schema.sql` updated.
- Every new model call (rules reading already exists) is covered by FakeLLM tests; the full suite runs with `pytest -q -W error`.
- The house style: no em dash in code, docs or UI text.

## 7. Left to the builder

- Exact prompt wording, the allow pattern for medication-change questions, the lexicon build method, the warning-term matcher reuse, UI layout of "How I respond" and the handoff card, branch name, order of implementation inside each part.
- `HABITS_IN_CONTEXT`, the clarify question cap, and the PortalPal memory/test split seed.

## 8. Decided by the team

- Habits are a second kind of rule read from the doctor's answers (approach 1).
- After the clone's questions: close when every safety condition holds, otherwise hand off with a summary (c).
- Drug names: prescribed, cited document paragraphs, and drugs the patient named in this turn.
- Negated warning terms: a term counts as absent only when the negation check and the model both say absent.
- The emergency backstop reads negation on reply turns only: a plainly denied term in a reply to the clone's questions is not an emergency; everywhere else the backstop is unchanged (4.4).

## 9. Calls made by Claude (not explicitly decided by the team)

- A habit's evidence and source card are its quote only; habits learned from one patient's entry apply to all of that doctor's patients; quotes with a person's name are dropped.
- Untagged general answers can be saved; only their habits are used.
- A pending `clarify` is a `waiting_on_patient` care-team question that becomes open after 24 hours, computed at read time; a new unrelated question hands the symptom off.
- A warning sign in the first message still allows questions but forbids a close.
- A prescribed drug named in the reply does not block a close.

- Warning signs come from three sources (doctor's `warning` habits, document warning terms, emergency backstop); any match forbids a close.
- Habits are given in full, not retrieved; habits are never checked against drug labels.
- The confidence gate skips `clarify` but scores `close`.
- One ask-first round only.
- Drug lexicon from RxNorm's prescribable subset with a food and common-word exclusion list.
- Retrieval floor: top K with a low absolute floor and a margin from the best score, tuned on the calibration split.
- PortalPal split with a fixed seed: 60 memory rows, 15 tuning rows (prompt wording, question cap), 25 test rows (small; numbers are reported with counts, not as rates alone).
