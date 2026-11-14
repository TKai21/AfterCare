# AfterCare - Doctor Clone Design

Date: 2026-11-13
Status: draft, awaiting team approval
Builds on: `2026-11-07-aftercare-milestone1-design.md`. This reverses that spec's non-goal "a clinician-facing interface".

## 1. Intent

Patients ask many follow-up questions after discharge. Today the AfterCare agent answers only what the discharge document, the education library and the food table support, and sends everything else to a "Care team" list that nobody answers inside the app. The doctor still answers those questions by hand, often the same ones again and again.

The doctor clone lets a (fictional) doctor, **Dr. Ting Kai**, answer those questions once in a doctor interface. Every answer becomes part of the clone's memory: Dr. Ting's usual medications, dosing habits and advice to patients. The next time a patient asks something the memory covers, the clone answers directly, citing Dr. Ting's earlier advice. Anything the memory does not cover still goes to Dr. Ting's queue and waits for an answer.

Success: over time, fewer questions reach Dr. Ting, and the answers patients get match what Dr. Ting would have said, with a citation showing which earlier advice it came from.

## 2. Decisions made with the team

- **Approach A, retrieval memory (chosen).** The doctor's answers are stored as entries, embedded, and retrieved for each question. Approach B, fine-tuning a model on the doctor's answers, was rejected for now: it cannot cite which advice it used, a new answer only takes effect after retraining, a wrong answer cannot be removed, the server's number-grounding check has nothing to check against, and it needs hundreds of examples and a GPU. B may come later, only for tone, on the self-hosted vLLM server.
- **Doctor in the loop.** The doctor answers questions through an interface; the clone learns from those answers and answers later questions with them; questions it cannot answer are stored until the doctor answers.
- **No face, no voice.** Real-time avatar, speech-to-text, text-to-speech and voice cloning are out of scope, and so is a real-time pipeline (Pipecat, LiveKit). This is text Q&A on the existing `/api/ask`.

## 3. Goals

1. Doctor accounts and a doctor interface, separate from the patient app.
2. A queue of open questions from all patients, with enough patient context to answer: the question, the patient's discharge document and confirmed instructions, and the recent conversation.
3. The doctor answers a question; the patient sees the answer in their Care team tab, marked as from Dr. Ting Kai.
4. Each answer is saved as a guidance entry in the clone's memory, with a scope (section 5).
5. The clone (the existing agent) searches the memory and answers from it, citing the entry.
6. The doctor can view, edit, retire and re-activate guidance entries.

## 4. Non-goals and anti-goals

Non-goals for now: fine-tuning; face and voice; multiple doctors or clinics (one doctor, every patient is that doctor's patient); notifications (email, push); the doctor writing guidance that is not an answer to a patient question (see section 12, open to a later milestone).

Anti-goals, failure even if it works:

- **The patient thinks they talked to the real doctor.** The clone never claims to be Dr. Ting and never writes in Dr. Ting's first person. AI answers and the doctor's own answers look different on screen.
- **Advice for one patient reaches another patient it does not fit.** See scope, section 5.
- **The clone invents medical advice** that no source says, including doses, "general" medication information from its own knowledge, or Dr. Ting's habits it was never told.
- **Memory silently overrides the discharge document** without saying so.
- **The emergency path depends on the memory or the model.** The R19 emergency backstop stays first and unchanged.

## 5. Scope of a guidance entry

When the doctor answers, the answer form has a checkbox: **"General advice: the assistant may use this for other patients."** Unchecked by default.

- **Patient-specific** (unchecked): the clone may use the entry only for the patient who asked.
- **General** (checked): the clone may use it for any patient.

Why: the same medication advice can harm a different patient (for example extra colchicine for a patient with kidney disease). This default was proposed by the designer, not decided by the team; the team may change it.

### Applicability gate for general entries (decided by the team: option B)

Similarity alone would let a general entry reach a patient it does not fit (a colchicine answer surfacing for a heart-failure patient who is not on colchicine). So a general entry carries **tags**, and code, not the model, decides whether it may be used for a patient.

- Each general entry has `drug_tags` and `condition_tags` (lists). A tag may list alternative names separated by "/", for example `colchicine / Colcrys`; any one name matching counts as that tag matching.
- When the doctor ticks "General advice", the app suggests tags from the question and answer with one model call (JSON list of drug names and condition names, each required to appear in the doctor's answer or question text). They show as editable chips the doctor can remove or add to before sending. If the suggestion call fails, the doctor types them.
- A general entry must have at least one tag; the form will not send a general answer with no tags, since an untagged entry would bypass the gate.
- **The gate:** a general entry is eligible for a patient only if every drug tag, and at least one condition tag when there are any, appears in that patient's active discharge document text or confirmed instructions. Matching is case-insensitive on whole words after the same normalization extraction uses.
- Ineligible entries are filtered out before the model sees anything; `search_doctor_guidance` never returns them.
- Patient-specific entries have no tags and no gate; their scope already limits them to one patient and one document.
- Tags are editable in "My guidance"; editing them changes eligibility immediately.

## 6. Authority order

When sources disagree, the clone follows, highest first:

1. The patient's own discharge document. It stays the top authority, as in milestone 1.
2. Dr. Ting's patient-specific guidance for this patient.
3. Dr. Ting's general guidance.
4. The education library.
5. The food table.

If a lower source conflicts with a higher one, the clone follows the higher one and says that the sources differ, then adds the question to the queue so the doctor can settle it. That turn returns kind `answer` (the patient gets the higher source's answer now) plus a queue entry with reason "Sources disagree", which the doctor's queue card shows. If the same patient already has an open question with that reason and the same text, no second one is added. (Designer's call.)

Designer's call, flagged for the team: a patient-specific answer the doctor gave after discharge is arguably newer than the document and could rank above it. This design keeps the document first because the patient confirmed it item by item and it is what the checklist is built from; a doctor who wants to change an instruction does so by telling the patient to upload an updated document.

## 7. How it works

### Accounts and roles

- `users` gains `role`: `patient` (default) or `doctor`.
- Sign-up creates patients only. Doctor accounts are created by a script (`scripts/create_doctor.py`, username and display name as arguments, password prompted). The demo seed creates `dr.ting` / display name "Dr. Ting Kai".
- Every route checks the role. Patient routes reject doctors and doctor routes reject patients (403). A login token carries the role, but the server reads the role from the database on every request.
- The doctor's display name comes from the account, not from configuration.

### Question lifecycle

`care_team_questions` gains:

- `status`: `open`, `answered` or `resolved`. This replaces the `resolved` boolean; existing rows map `resolved=true` to `resolved`.
- `answer`, `answered_by` (user id) and `answered_at`.

The flow:

1. A question becomes `open` when the agent adds it, exactly as today.
2. The doctor answers it, and it becomes `answered`. The patient sees the answer.
3. The patient may mark it `resolved`, as today. The patient may still resolve an open question themselves, for example after a phone call.

Only open questions appear in the doctor's queue. If the patient resolved the question while the doctor was writing, the doctor's answer is still accepted: it is saved, shown to the patient, and creates the guidance entry; the status stays `resolved`. A second answer to an already answered question is rejected (409). (Designer's call.) An answer cannot be edited after it is sent; the doctor corrects advice by editing the guidance entry (section 7, memory) and, if needed, answering again on a new question.

### Memory: guidance entries

New table `doctor_guidance`:

- `id`, `doctor_id`, `question` (the patient question it answered), `answer` (the doctor's text)
- `scope`: `general` or `patient`; `patient_id` and `document_id` (the patient's discharge document at answer time) when the scope is `patient`. A patient-specific entry is retrieved only while that document is the patient's active one; after the patient uploads a new document, it stays in the doctor's guidance list marked "for an earlier document" and is not used. (Designer's call.)
- `source_question_id`, the care-team question it came from
- `active` (boolean), `created_at`, `updated_at`
- `embedding` (JSON float list) and `embedding_model`

It lives in the main application database, not the education-library SQLite index, so it works on MySQL and SQLite and is deleted with its doctor.

- One entry is created per answer, from the question and the answer text. The embedding is computed from "question + answer".
- Editing an entry re-embeds it. Retiring sets `active=false`; retired entries are never retrieved. There is no hard delete in the interface, so a past citation still resolves.
- If embedding fails when the doctor answers, the answer is still saved and sent to the patient, and the entry is saved without a vector and marked "not searchable yet" in the doctor's guidance list, with a retry button. The doctor's answer to the patient never depends on the embedding service.
- Retrieval: cosine similarity over active entries the asking patient may see (general entries that pass the applicability gate in section 5, plus patient-specific entries for that patient and document), with entries from a different embedding model skipped. Top 5 at or above a similarity threshold (setting `GUIDANCE_MIN_SIMILARITY`; the builder picks the default by testing on the demo cases). NumPy over all rows is fine at this scale; the number of entries is in the hundreds.

### The clone: changes to the agent

- New tool `search_doctor_guidance(query)`. It returns matching entries, each with an evidence number like every other source. Evidence kind `doctor`, title "Dr. Ting Kai's advice, <date>", content = the doctor's answer, check text = the answer. The question the entry answered is shown to the model as context but is not cited evidence, so numbers in it cannot ground an answer.
- The system prompt adds: search the doctor's guidance for any question the document does not answer, before the library; the authority order in section 6; a guidance entry about a different situation does not apply (for example, advice about a different medication or condition); never present guidance as the doctor speaking to the patient now.
- Milestone 1 rule 4 ("never suggest changing, stopping, adding, or substituting a medication or dose") changes to: only when a cited document paragraph or doctor guidance entry says so. Library passages and the model's own knowledge still never justify a medication change. Server-side check: an answer with kind `answer` whose text matches medication-change wording must cite at least one `document` or `doctor` source, or it is rejected (repair round, then fallback). The builder defines the wording list and tests it on the demo cases.
- Number grounding is unchanged: every number in an answer must appear in the cited evidence, now including doctor entries.
- When nothing covers the question, it goes to the queue as today, with the patient told it was sent to Dr. Ting Kai.

### Patient experience

- **Ask tab:** an answer citing doctor guidance shows a source chip "Dr. Ting Kai's advice (Nov 12, 2026)". The answer header says it is the AfterCare assistant using Dr. Ting's earlier advice. Not-covered answers say the question was sent to Dr. Ting Kai.
- **Care team tab:** each question shows its status. Answered questions show the doctor's answer in a visually distinct "Dr. Ting Kai replied" block with the date. The tab title shows a count of new answers since the patient last opened it (per-browser, local storage is fine).

```
 Care team
 ┌───────────────────────────────────────────────────┐
 │ Nov 12 · From the assistant              ANSWERED │
 │ Can I take another colchicine if it still hurts?  │
 │ ┌ Dr. Ting Kai replied · Nov 12 ───────────────┐  │
 │ │ Yes, one extra 0.6 mg tablet after 1 hour,   │  │
 │ │ then no more that day.                       │  │
 │ └──────────────────────────────────────────────┘  │
 │                               [ Mark resolved ]   │
 └───────────────────────────────────────────────────┘
```

### Doctor interface

A separate page, `/doctor`, reached by logging in with a doctor account (the login page sends doctors there and patients to `/app`). Same visual style as the patient app. Two tabs:

**Questions** (default): open questions, oldest first, then a collapsed list of answered ones.

```
 Dr. Ting Kai · AfterCare                         Log out
 [ Questions (3) ]  [ My guidance ]

 ┌ sam.delgado · asked Nov 12 ────────────────────────────┐
 │ Can I take another colchicine if it still hurts?       │
 │ Why it came here: not in the discharge document        │
 │ ▸ Discharge document   ▸ Confirmed instructions        │
 │ ▸ Recent conversation (last 6 messages)                │
 │ ┌────────────────────────────────────────────────────┐ │
 │ │ Your answer...                                     │ │
 │ └────────────────────────────────────────────────────┘ │
 │ The assistant follows the discharge document over      │
 │ this answer if they conflict.                          │
 │ [x] General advice: the assistant may use this         │
 │     for other patients                                 │
 │   Question as saved: [ Can I take another colchicine   │
 │                        if a gout flare still hurts? ]  │
 │   Drugs:      (colchicine / Colcrys ×)  [+ add]        │
 │   Conditions: (gout ×)                  [+ add]        │
 │                                       [ Send answer ]  │
 └────────────────────────────────────────────────────────┘
```

The tag rows and "Question as saved" appear only when "General advice" is ticked.

The patient is shown by username (the app has no patient names). The document and conversation panels are collapsed by default.

When the doctor ticks "General advice", an editable "Question as saved" field appears, prefilled with the patient's question, so the doctor can generalize it and remove anything specific to that patient before it is shared with the clone for everyone. (Designer's call.) Under the answer box, a one-line note reminds the doctor that the assistant follows the patient's discharge document over this answer if the two conflict (section 6).

**My guidance:** every entry, newest first, with filters for scope and active/retired. Each row shows the question, the answer, the scope (and which patient for patient-specific entries), the date, and how many times the clone has cited it. Actions: edit answer, change scope, retire or re-activate, and retry embedding when "not searchable yet". Editing changes future clone answers only; past chat messages keep the text they showed.

Citation counts come from recording each cited guidance id with the chat message payload, which the app already stores.

## 8. Constraints

- Everything from milestone 1 still holds: FastAPI, async SQLAlchemy on MySQL or SQLite, JWT, plain HTML/JS frontend, OpenAI-compatible chat and embeddings (vLLM later), no em dash in any user-facing text or prompt.
- New tables and columns must be created by `create_all` on a fresh database and added to `sql/schema.sql`. The `care_team_questions.resolved` to `status` change needs a migration note in the README for existing databases (dev databases may be dropped and recreated, as milestone 1 already documents).
- The R19 emergency backstop runs first on every patient message and never depends on the model or the memory.
- All doctor and patient data is fictional.

## 9. Demo data

The seed script gains a doctor account and a fictional history so the clone has memory at demo time: a few answered questions across the demo cases, at least one general entry (for example colchicine timing for a gout flare) and one patient-specific entry, plus one open question waiting in the queue. Demo answers must be clinically plausible and clearly fictional. A demo walkthrough:

1. Patient asks a question nobody has answered; it goes to Dr. Ting.
2. Log in as `dr.ting`, answer it, tick "General advice".
3. A second patient asks a similar question; the clone answers, citing Dr. Ting's advice.

## 10. Testing

Unit and route tests with the fake LLM, in the style of milestone 1: role checks on every route, the question lifecycle, scope filtering (a patient never retrieves another patient's patient-specific entry), the applicability gate (a general colchicine entry never reaches a patient whose document does not mention colchicine; alternative names match; an untagged general answer is refused), retired entries never retrieved, embedding failure on answer, the medication-change citation check, evidence numbering and number grounding for doctor entries, and the authority order in the prompt. A real-model check of the demo walkthrough in section 9 before the team demo.

## 11. Left to the builder

File and module layout; the guidance similarity threshold default (tuned on the demo cases); the medication-change wording list; exact UI copy outside what this document quotes; how the doctor page shares code with the patient app; pagination (not needed at demo scale).

## 12. Possible later work

- Doctor-written guidance without a patient question ("my standard gout flare advice").
- Fine-tuning for tone on the self-hosted model (approach B).
- Multiple doctors, each patient linked to one doctor.
- Letting the doctor review a sample of clone answers after the fact.
