# Multi-Doctor AI Clones Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three doctors, each with an AI clone that answers in the first person (labelled as an AI clone) from that doctor's own advice in a per-doctor conversation, with questions routed to that doctor, a direct "ask the real doctor" path, and a more reliable clone turn.

**Architecture:** `doctor_id` is added to chat messages and care-team questions; `guidance.retrieve` and `/api/ask` are scoped to one doctor; the agent prefetches that doctor's guidance as evidence before the first model call and gets a format-only retry; the patient Ask tab gets one tab per doctor and the Care team tab a direct-question form; doctor routes only show the doctor's own questions and guidance.

**Tech Stack:** FastAPI, async SQLAlchemy (SQLite tests), OpenAI-compatible chat/embeddings via `LLMClient`, plain HTML/JS ES modules, pytest + `FakeLLM`, `node --check` via tests/test_frontend_syntax.py.

**Spec:** `docs/superpowers/specs/2026-11-13-multi-doctor-clones-design.md` (builds on `docs/superpowers/specs/2026-11-13-doctor-clone-design.md`; the newer spec wins on conflicts).

## Global Constraints

- No em dash (U+2014) in any code string, prompt, UI copy, or doc. Use "-".
- Demo doctors: `dr.ting` "Dr. Ting Kai", `dr.renny` "Dr. Aaron Renny", `dr.patel` "Dr. Parin Patel", all password `doctor-password-1`. All data fictional.
- A clone never says or implies it is the human doctor; every clone answer in the UI shows the label `AI clone of <doctor display name>`.
- A clone only ever sees its own doctor's guidance entries; the applicability gate, scope and document rules from the first spec still apply.
- The R19 emergency backstop runs first on every patient message (Ask and direct questions) and never depends on the model or the guidance memory.
- Exact copy: tab lead `Answers come from your discharge document and <name>'s earlier advice, given by <name>'s AI clone.`; direct action `Ask <name> directly`; Care team form heading `Ask a doctor directly`; doctor-queue label for source `patient`: `Asked by the patient`; per-question line `For <name>`; not-covered/fallback answer `I couldn't answer this from your discharge document or my earlier advice, so I sent it to <name>.`
- Unassigned care-team questions (e.g. from a rejected extracted item) appear in every doctor's queue; answering assigns them to the answering doctor.
- Test command: `.venv/bin/pytest -q -W error`. Every task ends with the full suite green.

## Review Focus

1. A patient passes another doctor's id (or a patient's id) as `doctor_id` to `/api/ask`, `/api/conversation` or `POST /api/care-team`: 422 for a non-doctor id; nothing leaks. Test in Task 1 and Task 3.
2. Switching doctor tabs mid-conversation: each tab shows only its own history, and a clone's prompt history never contains another doctor's messages. Test in Task 3.
3. A doctor opens PATCH/embed on another doctor's guidance id: 404, entry unchanged. Test in Task 1.
4. Prefetch retrieval raises (DB or embedding failure) on an emergency message: the emergency answer still returns. Test in Task 2/3.
5. A model reply that is prose only, then a content failure: the format retry and the repair round are both used before fallback, never more model calls than max_rounds + 3. Test in Task 2.

---

### Task 1: Doctor assignment in the data model and APIs

**Files:**
- Modify: `backend/models.py`, `sql/schema.sql`, `backend/routes_care_team.py`, `backend/routes_doctor.py`
- Create: `backend/doctors.py`
- Test: `tests/test_routes_care_team.py`, `tests/test_routes_doctor.py`, `tests/test_doctors.py`

**Interfaces:**
- Produces:
  - `ChatMessage.doctor_id: Optional[int]` (FK users.id CASCADE, index). `CareTeamQuestion.doctor_id: Optional[int]` (FK users.id SET NULL, index).
  - `backend/doctors.py`: `async def list_doctors(session) -> list[User]` (role doctor, ordered by id); `def doctor_label(user: User) -> str` (display_name stripped, else username); `async def require_doctor(session, doctor_id: int) -> User` raising 422 `"Choose one of the doctors."` when the id is not a doctor.
  - `GET /api/doctors` (current_patient) -> `{"doctors": [{"id", "name"}]}`.
  - `add_question(..., doctor_id: int | None = None)`; dedupe also matches `doctor_id`.
  - `question_view` adds `doctor_id` and `doctor_name` (the assigned doctor's label or None); `list_questions` loads names for both `answered_by` and `doctor_id` in one query.
  - `POST /api/care-team` (current_patient) body `{"doctor_id": int, "question": str (1..2000, stripped)}` -> `{"question": question_view, "emergency": {"terms": [...], "banner": bool}}`. Source `patient`, reason `Asked by the patient`, document = the patient's active document if any (else None). Runs `detect_emergency(question, warning_terms(items of the active document or []))` first; saves the question either way.
  - Doctor routes: queue = questions where `doctor_id == me or doctor_id is None`; answering sets `doctor_id = me` when it was None (in the same conditional UPDATE); `GET /api/doctor/guidance` lists only `doctor_id == me`; PATCH and `/embed` return 404 for another doctor's entry.

- [ ] **Step 1: Write the failing tests**: `test_doctors_endpoint_lists_doctors_in_id_order` (two doctors via `doctor_headers` with different usernames; patient sees both, 403 for a doctor); `test_direct_question_assigned_and_labelled` (POST /api/care-team -> source `patient`, `doctor_id`, `doctor_name`, reason `Asked by the patient`); `test_direct_question_rejects_non_doctor_id` (a patient's own id -> 422 with the exact detail, nothing saved); `test_direct_question_emergency_still_saved` ("I have chest pain" -> banner true and the question exists); `test_queue_shows_own_and_unassigned_only` (doctor A sees A's and unassigned, not B's); `test_answering_unassigned_assigns_it`; `test_guidance_of_another_doctor_is_404` (PATCH and embed; entry unchanged); `test_dedupe_is_per_doctor`.
- [ ] **Step 2: Run** them - Expected: FAIL.
- [ ] **Step 3: Implement.** Register the `/api/doctors` route on the care-team router (or a new router included in app.py). Add both columns to `schema.sql`.
- [ ] **Step 4: Run** the full suite - Expected: PASS.
- [ ] **Step 5: Commit** `feat: questions and messages belong to a doctor; patients can ask a doctor directly`.

### Task 2: The clone agent: own doctor only, first person, prefetch, format retry

**Files:**
- Modify: `backend/guidance.py`, `backend/agent.py`
- Test: `tests/test_guidance.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: Task 1 columns.
- Produces:
  - `retrieve(..., doctor_id: int, ...)` (new required keyword): candidates filtered to `DoctorGuidance.doctor_id == doctor_id` before the gate and scoring.
  - `run_agent(..., doctor_name: str, prefetch: Optional[list[DoctorGuidance]] = None)`: `doctor_names` is removed (one doctor per turn). Prefetched entries are numbered as `doctor` evidence before the first model call (same `_number` key `("doctor", id)` so a later tool hit reuses the number) and listed in the user message after the document as "My earlier advice (evidence numbers)" with each entry's `applies_to`, `question_answered` (context only) and `advice`.
  - Model-facing label for doctor evidence: `My earlier advice, <Mon D, YYYY>`; `Evidence.title` (what the patient's chip shows) stays `guidance_title(doctor_name, entry)` -> `Dr. Ting Kai's advice, Nov 12, 2026`.
  - Prompt: replace the third-person sentence in rule 2 with the spec section 6 first-person AI-clone rule (verbatim: "You are {DOCTOR}'s AI clone. Speak in the first person as {DOCTOR}'s AI clone ('my earlier advice', 'I advised'). Never say or imply you are the human doctor; when you introduce yourself, say you are {DOCTOR}'s AI clone."); other rules that say "{DOCTOR}'s advice" become "my earlier advice" where they address the patient; authority order unchanged.
  - `fallback_answer(doctor_name)` returns the Global Constraints not-covered copy; rule 6 tells the model to use the same sentence for kind `not_covered`.
  - Loop: a final message with no JSON object at all (`_final_object` raises) gets one format-only retry with `FORMAT_NUDGE = "Reply again with only the final JSON object described in the output contract, and no other text."` that does not set `repaired`; any other validation failure uses the existing single repair round. Emergency never retries (unchanged).

- [ ] **Step 1: Write the failing tests**: `test_retrieve_only_own_doctor` (identical general entries from two doctors; each doctor_id gets only its own); `test_prefetched_guidance_is_evidence_before_any_tool_call` (first chat call's user message contains "My earlier advice" and the evidence number; an answer citing it with no tool call is accepted; `guidance_ids` set); `test_prefetch_and_tool_share_numbers`; `test_doctor_evidence_titles` (model sees "My earlier advice, Nov 12, 2026", `Evidence.title` is "Dr. Ting Kai's advice, Nov 12, 2026"); `test_prompt_states_ai_clone_rule` (the verbatim sentence; no "third person"); `test_prose_reply_gets_format_retry_then_repair` (script: prose, then JSON with an ungrounded number, then valid -> answer; 3 chat calls; second call's last message is FORMAT_NUDGE, third is the repair nudge); `test_second_prose_reply_uses_repair_then_falls_back` (prose, prose, prose -> fallback after 3 calls); `test_fallback_copy_first_person`. Update existing tests that assert third-person wording, `doctor_names`, or call counts.
- [ ] **Step 2: Run** `tests/test_guidance.py tests/test_agent.py` - Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite - Expected: PASS (route tests calling `retrieve`/`run_agent` are updated in Task 3; keep them passing here by giving the route the doctor id of the lowest-id doctor until Task 3 replaces it).
- [ ] **Step 5: Commit** `feat: each clone uses only its own doctor's advice, speaks as an AI clone, and prefetches guidance`.

### Task 3: Per-doctor Ask and conversation routes

**Files:**
- Modify: `backend/routes_agent.py`
- Test: `tests/test_routes_agent.py`

**Interfaces:**
- Consumes: Task 1 `require_doctor`, `doctor_label`, `add_question(doctor_id=)`; Task 2 `retrieve(doctor_id=)`, `run_agent(doctor_name=, prefetch=)`.
- Produces:
  - `AskBody.doctor_id: int` (required). `/api/ask` resolves the doctor with `require_doctor` before anything else that touches the model (the emergency backstop still runs before the model; a bad id is 422 even for an emergency message). History = last `HISTORY_LIMIT` messages with this document and doctor. Prefetch = `guidance_search(body.question)` before `run_agent`, through the same failure-safe closure (a failure means no prefetch; the turn continues). Care-team questions get `doctor_id`. Messages saved with `doctor_id`. Payload adds `doctor_id` and `doctor_name` (this doctor). `doctor_directory` is removed.
  - `GET /api/conversation?doctor_id=<id>` (required, 422 for a non-doctor) returns only that doctor's messages for the active document.

- [ ] **Step 1: Write the failing tests**: `test_ask_requires_a_doctor` (missing -> 422; patient id -> 422); `test_history_is_per_doctor` (ask doctor A then doctor B; B's first chat call has no A messages; `/api/conversation?doctor_id=A` shows only A's two messages); `test_not_covered_goes_to_that_doctor` (question saved with B's id, appears only in B's queue); `test_ask_uses_only_that_doctors_guidance` (entries from A and B both eligible; asking B, the prefetched evidence is B's entry only); `test_prefetch_failure_keeps_emergency` (retrieve patched to raise on an emergency message -> emergency payload); `test_payload_names_the_doctor`. Update existing route tests to send `doctor_id`.
- [ ] **Step 2: Run** - Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite - Expected: PASS.
- [ ] **Step 5: Commit** `feat: one conversation per doctor in Ask`.

### Task 4: Patient UI: doctor tabs, clone label, ask directly

**Files:**
- Modify: `frontend/ask.js`, `frontend/careteam.js`, `frontend/app.css`, `frontend/app.html` (only if boot needs the doctor list)
- Test: `tests/test_frontend_syntax.py` (existing, covers the changed scripts)

**Interfaces:**
- Consumes: `GET /api/doctors`, `GET /api/conversation?doctor_id=`, `POST /api/ask {question, doctor_id}`, `POST /api/care-team {doctor_id, question}` and its `emergency` field, `question_view.doctor_name`.
- Produces: Ask tab per spec section 7: a doctor tab row (wraps on narrow screens, no horizontal page overflow), the selected doctor remembered in `localStorage["aftercare.doctorId"]` (try/catch; first doctor by default; ignored if no longer in the list), the lead copy, the `AI clone of <name>` label on every clone answer, the `Ask <name> directly` inline form prefilled with the patient's last question in this conversation, and a confirmation line linking to `#careteam`. Care team tab: the `Ask a doctor directly` form (doctor select, question, Send) above the list; each question shows `For <name>` when `doctor_name` is set. If a direct question returns `emergency.banner`, show the existing emergency banner and copy (reuse ask.js's emergency rendering or the same CSS) above the confirmation. Emergency answers in Ask keep their current rendering.

- [ ] **Step 1:** Implement. Escape every server string.
- [ ] **Step 2:** Run the full suite (includes the syntax test) - Expected: PASS.
- [ ] **Step 3: Commit** `feat: patients choose a doctor's AI clone and can ask the doctor directly`.

### Task 5: Doctor UI labels

**Files:**
- Modify: `frontend/doctor.js`, `frontend/app.css` (only if needed)
- Test: `tests/test_frontend_syntax.py`

**Interfaces:**
- Consumes: Task 1 queue semantics and `source == "patient"`.
- Produces: the queue card shows `Asked by the patient` for source `patient` (in place of the "Why it came here" reason line), and `Unassigned` for questions with `doctor_id` null.

- [ ] **Step 1:** Implement.
- [ ] **Step 2:** Run the full suite - Expected: PASS.
- [ ] **Step 3: Commit** `feat: doctor queue labels direct and unassigned questions`.

### Task 6: Three demo doctors, README and real-model acceptance

**Files:**
- Modify: `scripts/seed_demo.py`, `README.md`, `tests/test_seed_demo.py`

**Interfaces:**
- Consumes: everything above.
- Produces: seed creates the three doctors from Global Constraints; keeps the existing Dr. Ting Kai entries and questions (now with `doctor_id`); adds for Dr. Aaron Renny and Dr. Parin Patel at least one general entry each that passes the gate for `demo` or `demo2` (fictional, plausible, numbers in the answer text, tags present in that patient's document) and one open question each from a demo patient assigned to them; seeded chat messages (if any) carry `doctor_id`. README walkthrough per spec section 10 (doctor tabs, ask a clone, ask a doctor directly, log in as that doctor, answer).

- [ ] **Step 1:** Extend `tests/test_seed_demo.py`: three doctors with exact usernames and display names; each has at least one general entry eligible for demo or demo2; each has exactly one open question assigned; idempotent re-run.
- [ ] **Step 2:** Run - Expected: FAIL. Implement. Run the full suite - Expected: PASS.
- [ ] **Step 3:** Real-model acceptance (spec section 8) against a seeded temporary copy of the dev DB (not the user's DB) on port 8014: demo2 asks Dr. Ting Kai's clone "My big toe still hurts a lot after my colchicine, can I take one more?" 10 times in fresh conversations (clear that doctor's chat messages between runs) -> at least 9 answers cite a `doctor` source; demo asks the same 3 times -> never a doctor source. Also one question per other doctor that should hit their entry. Record counts in the commit message and report. If below 9/10, report BLOCKED with the failure reasons logged by tracing `agent.validate`, rather than tuning silently.
- [ ] **Step 4: Commit** `feat: three demo doctors and docs` (with the acceptance counts).
