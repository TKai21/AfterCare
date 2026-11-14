# Doctor Clone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a fictional doctor, Dr. Ting Kai, answer queued patient questions in a doctor interface, store each answer as retrievable guidance, and let the existing AfterCare agent answer later questions from that guidance, gated so general advice only reaches patients whose documents mention its drugs and conditions.

**Architecture:** A `role` on users splits patient and doctor routes. Care-team questions gain a status and an answer. A new `doctor_guidance` table in the main database holds answers with embeddings and tags; `guidance.py` owns the applicability gate and retrieval. The agent gets a `search_doctor_guidance` tool whose results are evidence kind `doctor`, plus a server-side medication-change citation check. A new `/doctor` page is the doctor interface.

**Tech Stack:** FastAPI, async SQLAlchemy (SQLite in tests, MySQL supported), python-jose JWT, NumPy cosine search, OpenAI-compatible chat and embeddings via `LLMClient`, plain HTML/JS frontend, pytest with `FakeLLM`.

**Spec:** `docs/superpowers/specs/2026-11-13-doctor-clone-design.md` (builds on `docs/superpowers/specs/2026-11-07-aftercare-milestone1-design.md`).

## Prerequisite

The working tree had uncommitted changes from another session (document viewer, PDF storage, extraction and agent fixes). Execution starts only after those are committed on the branch, so each task's diff contains only its own work. Run `.venv/bin/pytest -q -W error` first; it must pass.

## Global Constraints

- No em dash (U+2014) in any code string, prompt, UI copy, or doc. Use "-".
- The R19 emergency backstop (`detect_emergency` in `routes_agent.ask`) runs first on every patient message and never depends on the model or the guidance memory.
- New tables and columns come from `create_all` on a fresh database and are also added to `sql/schema.sql`.
- Doctor display name comes from `users.display_name`, never from configuration. The demo doctor is username `dr.ting`, display name `Dr. Ting Kai`.
- AI answers never claim to be the doctor and never use the doctor's first person.
- Exact copy pinned by the spec: checkbox label `General advice: the assistant may use this for other patients`; doctor answer block heading `Dr. Ting Kai replied` (rendered from display name); conflict reason `Sources disagree`; source chip `Dr. Ting Kai's advice (<Mon D, YYYY>)`; form note `The assistant follows the discharge document over this answer if they conflict.`; stale entry label `for an earlier document`; unsearchable label `not searchable yet`.
- Test command: `.venv/bin/pytest -q -W error` from the project root. Every task ends with the full suite green.

## Review Focus

1. A general entry whose tag is a substring of an unrelated word (tag `gout` vs document text `gouty`, tag `ace` vs `replace`) must not pass the gate: whole-word matching. Test in Task 3.
2. A doctor token used on a patient route (and the reverse) gets 403, including `/api/ask` and `/api/conversation`; an old token issued before roles existed still works for a patient. Test in Task 1.
3. Two doctor answers racing on the same question: exactly one succeeds, the other gets 409, and only one guidance entry exists. Test in Task 4.
4. Embedding service down when the doctor answers: the answer still reaches the patient and the entry exists with `embedding=None`, and retrieval skips it without error. Test in Task 4 and Task 3.
5. A patient-specific entry after the patient uploads a new document is never retrieved. Test in Task 3.

---

### Task 1: Roles and doctor accounts

**Files:**
- Modify: `backend/models.py` (User), `backend/auth.py`, `backend/routes_documents.py`, `backend/routes_checklist.py`, `backend/routes_care_team.py`, `backend/routes_agent.py`, `sql/schema.sql`, `frontend/login.js`
- Create: `scripts/create_doctor.py`
- Test: `tests/test_roles.py`

**Interfaces:**
- Produces: `User.role: str` (`"patient"` default, `"doctor"`), `User.display_name: Optional[str]` (String(100)). In `auth.py`: `async def current_patient(user=Depends(current_user)) -> User` and `async def current_doctor(user=Depends(current_user)) -> User`, each raising 403 `"This page is for patients."` / `"This page is for doctors."`. `/api/login` and `/api/signup` responses add `"role"`; `/api/me` returns `{"username", "role", "display_name"}`. Token payload adds `"role"` but the server always reads the role from the database. `async def create_doctor(session, username: str, password: str, display_name: str) -> User` in `auth.py` (used by the script and the seed).

- [ ] **Step 1: Write the failing tests** in `tests/test_roles.py`, using the `client` fixture and a helper that creates a doctor via `create_doctor` on `client`'s app session factory:
  - `test_signup_creates_patient`: signup response `role == "patient"`; `/api/me` role patient, display_name None.
  - `test_doctor_login_returns_role_and_display_name`: login as `dr.ting` returns role doctor; `/api/me` display_name `"Dr. Ting Kai"`.
  - `test_doctor_gets_403_on_patient_routes`: for each of `GET /api/documents/active`, `GET /api/checklist`, `GET /api/care-team`, `POST /api/ask`, `GET /api/conversation`, `POST /api/documents` the doctor gets 403.
  - `test_old_token_without_role_still_works`: a token encoded with only `sub`, `username`, `exp` for a patient reaches `/api/care-team` with 200.
  - `test_role_comes_from_database`: a patient token whose payload claims `"role": "doctor"` is still a patient (403 on a doctor-only route; add a throwaway `GET /api/doctor/ping` in this task, returning `{"ok": true}`, removed in Task 4 when real doctor routes exist).
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_roles.py -q` - Expected: FAIL.
- [ ] **Step 3: Implement.** Add the columns; `create_doctor` hashes with `hash_password` and rejects a taken username with `ValueError`. Swap `current_user` for `current_patient` on every existing patient route. `scripts/create_doctor.py USERNAME "Display Name"` prompts for the password with `getpass` (twice, must match, min 8). `login.js`: after login and on the already-logged-in check, go to `/doctor` when role is doctor, else `/app`; store role with the session (`saveSession(token, username, role)` in `api.js`). Add `role VARCHAR(20) NOT NULL DEFAULT 'patient'` and `display_name VARCHAR(100) NULL` to `users` in `schema.sql`.
- [ ] **Step 4: Run** the full suite - Expected: PASS.
- [ ] **Step 5: Commit** `feat: patient and doctor roles, doctor accounts`.

### Task 2: Question lifecycle with answers

**Files:**
- Modify: `backend/models.py` (CareTeamQuestion), `backend/routes_care_team.py`, `sql/schema.sql`, `README.md`, `tests/test_routes_care_team.py`, `tests/test_routes_documents.py:234`
- Test: `tests/test_routes_care_team.py`

**Interfaces:**
- Produces: `CareTeamQuestion.status: str` (`open` | `answered` | `resolved`, default `open`; replaces `resolved`), `answer: Optional[str]` (Text), `answered_by: Optional[int]` (FK users.id SET NULL), `answered_at: Optional[datetime]`. `question_view(q, doctor_names: dict[int, str] | None = None) -> dict` returns `id, question, reason, source, status, answer, answered_by_name, answered_at, created_at, document_id`. `add_question(session, user_id, document_id, question, reason, source, dedupe: bool = False) -> CareTeamQuestion | None`: with `dedupe=True`, returns None without inserting when the user already has an `open` question with the same `question` and `reason`.

- [ ] **Step 1: Update and add tests**: existing assertions on `["resolved"]` become `["status"] == "resolved"` / `"open"`. New: `test_resolve_keeps_answer` (an answered question, resolved by the patient, still returns its answer and status resolved); `test_add_question_dedupe` (second identical open question with `dedupe=True` returns None; after resolving the first, a new one is inserted).
- [ ] **Step 2: Run** - Expected: FAIL.
- [ ] **Step 3: Implement.** `resolve` sets `status="resolved"` from `open` or `answered`. `list_questions` loads display names of `answered_by` users in one query. In `schema.sql` replace `resolved` with `status VARCHAR(20) NOT NULL DEFAULT 'open'`, `answer TEXT NULL`, `answered_by INT NULL` (FK SET NULL), `answered_at DATETIME NULL`. README: a short "Upgrading a milestone 1 database" note: drop and recreate the dev database (as already documented for `active_user_id`).
- [ ] **Step 4: Run** the full suite - Expected: PASS.
- [ ] **Step 5: Commit** `feat: care-team questions carry a status and a doctor answer`.

### Task 3: Guidance memory and the applicability gate

**Files:**
- Modify: `backend/models.py`, `backend/config.py`, `.env.example`, `sql/schema.sql`
- Create: `backend/guidance.py`
- Test: `tests/test_guidance.py`

**Interfaces:**
- Consumes: `normalize` from `extraction.py`; `llm.embed`, `llm.embedding_model`.
- Produces:
  - Model `DoctorGuidance` (`doctor_guidance`): `id, doctor_id (FK users CASCADE), question (Text), answer (Text), scope ("general"|"patient"), patient_id (FK users CASCADE, nullable), document_id (FK discharge_documents SET NULL, nullable), source_question_id (FK care_team_questions SET NULL, nullable), drug_tags (JSON list[str]), condition_tags (JSON list[str]), active (bool, default True), embedding (JSON list[float] | None), embedding_model (String(100) | None), created_at, updated_at`.
  - `Settings.guidance_min_similarity: float = 0.35`, env `GUIDANCE_MIN_SIMILARITY` (Task 8 tunes it).
  - `def tag_names(tag: str) -> list[str]`: split on "/", strip, drop empties.
  - `def eligible(entry: DoctorGuidance, patient_text: str) -> bool`: the gate. True only if `entry.scope == "general"`, it has at least one tag, every drug tag has a name that occurs as a whole word or phrase in `normalize(patient_text)`, and, if it has condition tags, at least one condition tag does. Whole word: regex `(?<![a-z0-9])` + escaped normalized name + `(?![a-z0-9])`.
  - `def patient_text(document: DischargeDocument, items: list[ExtractedItem]) -> str`: document text plus the text of non-rejected items.
  - `async def embed_entry(llm, entry: DoctorGuidance) -> bool`: embeds `question + "\n" + answer`; on `LLMError` leaves `embedding=None` and returns False.
  - `async def retrieve(session, llm, patient: User, document: DischargeDocument, items, query: str, min_similarity: float, limit: int = 5) -> list[tuple[DoctorGuidance, float]]`: candidates are active entries with an embedding from `llm.embedding_model`, either `scope=="patient" and patient_id==patient.id and document_id==document.id`, or general and `eligible(...)`. Cosine score, at or above `min_similarity`, top `limit`, best first. Filtering happens before scoring.

- [ ] **Step 1: Write the failing tests** (unit tests with `FakeLLM` embeddings and an in-memory session):
  - `test_gate_requires_every_drug_and_one_condition`: tags `["colchicine / Colcrys"]`, `["gout"]`; eligible for text containing "Colcrys 0.6 mg ... gout flare"; not for text with gout only; not for text with colchicine only.
  - `test_gate_whole_words`: tag `gout` not eligible for "gouty arthritis history"; tag `ace inhibitor` not for "replace inhibitor".
  - `test_untagged_or_patient_scope_never_eligible`.
  - `test_retrieve_scope_and_document`: patient-specific entry for patient A, document 1 is returned for A on document 1, not for B, not for A on document 2.
  - `test_retrieve_skips_retired_unembedded_and_other_model`: retired entry, `embedding=None`, and `embedding_model="old"` are never returned.
  - `test_embed_entry_failure_returns_false`: an llm whose `embed` raises `LLMError` leaves `embedding` None.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_guidance.py -q` - Expected: FAIL.
- [ ] **Step 3: Implement** `guidance.py`, the model, the setting (documented in `.env.example` and the README settings table), and the `schema.sql` table.
- [ ] **Step 4: Run** the full suite - Expected: PASS.
- [ ] **Step 5: Commit** `feat: doctor guidance memory with applicability gate`.

### Task 4: Doctor API

**Files:**
- Create: `backend/routes_doctor.py`
- Modify: `backend/app.py` (include router), `backend/auth.py` (remove the Task 1 ping route)
- Test: `tests/test_routes_doctor.py`

**Interfaces:**
- Consumes: `current_doctor`, `question_view`, `DoctorGuidance`, `embed_entry`, `active_document`, `load_items`, `recent_history`.
- Produces (all require `current_doctor`):
  - `GET /api/doctor/questions` -> `{"open": [...], "answered": [...]}`; open oldest first; answered newest first, last 50. Each item: `question_view` fields plus `patient: {"id", "username"}` and `context: {"document_paragraphs": list[str] | None, "confirmed_items": [{"category", "text"}], "conversation": [{"role", "content"}]}` (last 6 messages of that document; document fields None when the question has no document).
  - `POST /api/doctor/questions/{id}/answer` body `{"answer": str (1..4000), "general": bool, "saved_question": str | null, "drug_tags": list[str], "condition_tags": list[str]}`. 404 unknown; 409 `"This question already has an answer."` when `answer` is set; 422 `"General advice needs at least one drug or condition tag."` when general and both tag lists are empty after stripping. Sets answer fields; status becomes `answered` unless it is `resolved`. Creates one `DoctorGuidance`: general -> question = `saved_question` or the original, tags kept, no patient; patient -> `patient_id`, `document_id` = the question's document, tags empty. Commits the answer and entry first, then calls `embed_entry` and commits again, so an embedding failure never loses the answer. Response: `{"question": question_view, "guidance": guidance_view}`. The 409 check and update use a conditional `UPDATE ... WHERE id=:id AND answer IS NULL` and check the row count, so concurrent answers cannot both win.
  - `POST /api/doctor/suggest-tags` body `{"question", "answer"}` -> `{"drug_tags": list[str], "condition_tags": list[str]}`. One `llm.chat(json_mode=True)` call; keep only names that occur (case-insensitive, whole word) in question or answer. 503 when no model; 502 on `LLMError`.
  - `GET /api/doctor/guidance` -> `{"entries": [guidance_view]}` newest first. `guidance_view`: `id, question, answer, scope, patient (username or None), drug_tags, condition_tags, active, searchable (embedding present for the current model), stale (patient scope and document_id is not that patient's active document), cited_count, created_at, updated_at`.
  - `PATCH /api/doctor/guidance/{id}` body with any of `question, answer, scope, drug_tags, condition_tags, active`. Switching to general enforces the tag rule (422); switching to patient is allowed only when the entry has a `patient_id` (422 otherwise). Changing question or answer re-embeds (same failure rule).
  - `POST /api/doctor/guidance/{id}/embed` retries embedding; returns `guidance_view`.
  - `cited_count` counts chat message payloads whose `guidance_ids` list contains the id (Task 5 writes `guidance_ids`); computed in Python over assistant messages, fine at demo scale.

- [ ] **Step 1: Write the failing tests**: role 403 for a patient on every route above; answer creates a patient-scope entry tied to the question's document; general answer with no tags -> 422 and nothing saved; second answer -> 409; two concurrent answers on `file_client` via `asyncio.gather` -> one 200, one 409, one entry; answer with embedding raising `LLMError` -> 200, patient sees the answer in `/api/care-team`, entry `searchable` False, then `/embed` with a working fake -> `searchable` True; answering a resolved question keeps status resolved; suggest-tags drops names absent from the text; PATCH to general without tags -> 422; stale flag after the patient uploads a new document.
- [ ] **Step 2: Run** - Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite - Expected: PASS.
- [ ] **Step 5: Commit** `feat: doctor API for answering questions and managing guidance`.

### Task 5: The clone in the agent

**Files:**
- Modify: `backend/agent.py`, `backend/routes_agent.py`
- Test: `tests/test_agent.py`, `tests/test_routes_agent.py`

**Interfaces:**
- Consumes: `guidance.retrieve`, `DoctorGuidance`, `add_question(..., dedupe=True)`.
- Produces:
  - `run_agent(..., guidance_search=None, doctor_name: str = "your doctor")`: `guidance_search` is `async (query: str) -> list[DoctorGuidance]` or None (tool then returns `{"error": "Doctor guidance is not available right now."}`).
  - Tool `search_doctor_guidance(query)` in `TOOLS`. Each hit is evidence kind `doctor`, title `f"{doctor_name}'s advice, {date}"` (date `Mon D, YYYY` from `updated_at`), content and check text = the answer only. The tool result also shows the model `"question_answered": entry.question` (not evidence). `Evidence` gains `guidance_id: Optional[int] = None`.
  - `AgentResult` gains `guidance_ids: list[int]` (ids of cited doctor evidence).
  - `MEDICATION_CHANGE: re.Pattern` and a check in `validate`: kind `answer` whose answer matches it must cite at least one `document` or `doctor` evidence, else `ValueError("A medication change must cite the discharge document or the doctor's advice.")` (repair round, then fallback, as today).
  - Prompt changes: tool listed; rule 1 becomes the spec section 6 authority order (document, this patient's doctor advice, general doctor advice, library, food), with "if a lower source conflicts with a higher one, follow the higher one, say the sources differ, and call add_care_team_question with reason \"Sources disagree\""; new rule: search the doctor's guidance before the library when the document does not answer; a guidance entry about a different drug, condition or situation does not apply; refer to the doctor in the third person by name, never as "I"; rule 4 becomes "Never suggest changing, stopping, adding, or substituting a medication or dose unless a cited document paragraph or doctor's advice says so. Library passages and your own knowledge never justify it."; rule 9 adds doctor advice to the allowed sources; rule 5 says the question was sent to `{doctor_name}`.
  - `routes_agent.ask`: builds `guidance_search` with `retrieve(session, llm, user, doc, items, query, settings.guidance_min_similarity)` and passes `doctor_name` = display name of the (single) doctor account, or "your doctor" if none. Sources in the payload add `"guidance_id"`; payload adds `"guidance_ids"` and `"doctor_name"`. Care-team adds with reason `Sources disagree` use `dedupe=True`. The emergency backstop order is unchanged.

- [ ] **Step 1: Write the failing tests**:
  - `test_doctor_guidance_is_evidence`: scripted tool call -> evidence kind doctor, title with name and date, answer citing it accepted, `guidance_ids` set.
  - `test_doctor_question_text_does_not_ground_numbers`: entry question contains "2 tablets", answer "no"; an answer stating 2 citing it is rejected.
  - `test_medication_change_needs_document_or_doctor`: "Take an extra 0.6 mg colchicine [L]" citing only a library passage containing 0.6 mg -> ValueError; citing a doctor entry -> accepted. Plus `MEDICATION_CHANGE` matches "take an extra", "double your dose", "stop taking", "skip your", "switch to", "increase", "reduce your dose", and does not match "take it with food" or "take your medicine at bedtime".
  - `test_prompt_states_clone_rules`: phrases "Sources disagree", "third person", "unless a cited document paragraph or doctor's advice says so", "does not apply" in the prompt; no em dash.
  - `test_guidance_unavailable_tool_error`: `guidance_search=None` returns the error dict.
  - Route tests: `/api/ask` with a seeded general entry eligible for the patient answers with a `doctor` source and `guidance_ids`; with an ineligible one (document lacks the drug) the tool returns no hits (assert on the tool result the fake LLM received); a doctor-vs-document conflict turn adds one `Sources disagree` question and a second identical turn adds none; the emergency backstop still answers with `guidance_search` raising.
- [ ] **Step 2: Run** - Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite - Expected: PASS.
- [ ] **Step 5: Commit** `feat: the agent answers from the doctor's guidance`.

### Task 6: Patient UI for answers and doctor advice

**Files:**
- Modify: `frontend/careteam.js`, `frontend/ask.js`, `frontend/app.css`, `frontend/app.html` (tab badge)
- Test: `tests/test_frontend_caching.py` only if it enumerates files; otherwise manual check in Task 8.

**Interfaces:**
- Consumes: Task 2 `question_view`, Task 5 payload `sources[].kind == "doctor"`.
- Produces: Care team tab lists open, then answered, then a collapsed resolved list. An answered question shows a distinct block headed `<answered_by_name> replied · <Mon D>` with the answer, plus "Mark resolved". The tab label shows a count of answers whose id is not in `localStorage["aftercare.seenAnswers"]` (try/catch; marked seen when the tab is shown). Ask tab: `sourceLabel` for kind doctor returns the title; answers citing a doctor source show a one-line note above the sources: "The AfterCare assistant answered using <name>'s earlier advice." The not-covered chip text becomes "Sent to <payload.doctor_name>" (falls back to the current copy for old messages without `doctor_name`).

- [ ] **Step 1:** Implement the views. Status strings come from the server; no new endpoints.
- [ ] **Step 2:** Run the full suite - Expected: PASS.
- [ ] **Step 3: Commit** `feat: patients see doctor answers and doctor-advice sources`.

### Task 7: Doctor interface

**Files:**
- Create: `frontend/doctor.html`, `frontend/doctor.js`
- Modify: `backend/app.py` (route `/doctor` -> `doctor.html`), `frontend/app.css` (shared styles; doctor-only rules prefixed `.doctor-`)
- Test: `tests/test_routes_doctor.py::test_doctor_page_served`

**Interfaces:**
- Consumes: Task 4 endpoints, `api.js` helpers.
- Produces: header `<display_name> · AfterCare` with Log out; tabs "Questions (n)" and "My guidance". Questions tab per the spec mockup: username, date, reason, collapsible document / confirmed instructions / recent conversation, answer textarea, the form note, the General checkbox; when ticked: "Question as saved" (prefilled), drug and condition chip editors (add, remove), filled by `POST /api/doctor/suggest-tags` (debounced, on tick and after the answer stops changing for 800 ms; failure shows "Could not suggest tags. Add them yourself."); Send disabled when general and no tags. Below, a collapsed "Answered" list. My guidance tab: filters scope (all, general, patient) and status (active, retired); rows with question, answer, scope, patient, tags, date, `cited_count`, labels `for an earlier document` and `not searchable yet`; actions edit (inline), change scope, retire / re-activate, retry embedding. A patient token on `/doctor` is sent to `/app`, and the reverse on `/app`.

- [ ] **Step 1:** Write `test_doctor_page_served` (GET `/doctor` returns 200 HTML with `Cache-Control: no-cache`). Run - FAIL.
- [ ] **Step 2:** Implement page, script, styles, route; `app.html` boot redirects doctors to `/doctor`.
- [ ] **Step 3:** Run the full suite - Expected: PASS.
- [ ] **Step 4: Commit** `feat: doctor interface`.

### Task 8: Demo data, README and real-model check

**Files:**
- Modify: `scripts/seed_demo.py`, `README.md`, `backend/config.py` (threshold default if tuning changes it)
- Test: `tests/test_seed_demo.py` (create if absent)

**Interfaces:**
- Produces: `seed_demo.py` also (re)creates `dr.ting` / `doctor-password-1` / `Dr. Ting Kai`, a second patient `demo2` / `demo-password-2` with the gout case confirmed, and guidance: one general entry (gout flare colchicine timing; tags `colchicine / Colcrys`, `gout`), one patient-specific entry for `demo2`, one answered question visible to `demo2`, and one open question from `demo` in the queue. All answer text fictional and marked in a code comment as invented for the demo. Embeddings are computed when a model is configured; otherwise entries are saved unsearchable and the script prints how to retry from the doctor page.

- [ ] **Step 1:** Test that seeding twice is idempotent and creates exactly the entries above (with `FakeLLM` embeddings).
- [ ] **Step 2:** Implement; README gets a "Doctor clone" section: roles, `create_doctor.py`, the demo walkthrough from spec section 9, the gate in two sentences, and the new setting.
- [ ] **Step 3:** Real-model check with the configured `.env`: run the spec section 9 walkthrough end to end in the browser; ask the gout question as `demo2` (clone answers citing Dr. Ting) and as the kidney-stones patient (no doctor source, goes to the queue). Tune `GUIDANCE_MIN_SIMILARITY` so the paraphrased gout question hits and an unrelated question does not; record the chosen value and the scores seen in the commit message.
- [ ] **Step 4:** Run the full suite - Expected: PASS.
- [ ] **Step 5: Commit** `feat: doctor clone demo data and docs`.
