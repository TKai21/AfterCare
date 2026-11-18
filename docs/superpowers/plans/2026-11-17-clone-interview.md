# Doctor clone interview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the list-of-questions intake with a program-driven interview: items built from the doctor's habits on similar cases, one question per round, a "So far" line, a "Not sure" button, and denied danger signs that no longer end the interview.

**Architecture:** A new module `backend/interview.py` holds the item state (pure functions) and the three small model calls (build items, extract, write a question). `routes_agent.ask` gains an interview branch: on a first turn whose agent result would start an intake, it builds items and asks the first question; on a reply to an item-shape intake it runs the interview round without calling the agent until the interview ends. The denial rule lives in `safety.py`; the route applies it only on interview reply turns.

**Tech Stack:** FastAPI, async SQLAlchemy, SQLite, plain ES modules, pytest with FakeLLM, OpenAI gpt-4o-mini.

**Spec:** `docs/superpowers/specs/2026-11-17-clone-interview-design.md` (amends 4.7 of `docs/superpowers/specs/2026-11-15-clone-habits-design.md`).

## Global Constraints

- Settings with defaults: `INTERVIEW_CASES=4`, `INTERVIEW_MAX_ITEMS=5`, `INTERVIEW_MAX_SAFETY=3`, `INTERVIEW_MAX_ROUNDS=6` (in `config.py`, same pattern as `clarify_max_rounds`).
- Item label: at most 6 words, no digits, no drug names outside the turn's allowed names, no advice words (reuse `_CLONE_BANNED`).
- Clone-written question: the existing 4.7 checks (`_check_clarify` clone branch: one sentence ending "?", at most 25 words, no numbers, links, advice or suggestion words, allowed drugs only, no repeat).
- Item attempts at most 2; a quote over 60 characters is cut at a word boundary with "..." for display only (stored verbatim).
- Fixed "Not sure" patient text: `I'm not sure.`
- Self-harm and suicide labels are never relaxed.
- Fail closed everywhere: any model or parse failure in the interview -> today's 4.7 flow on a first turn, handoff on a reply turn.
- Never write to `aftercare-new.sqlite3` from tests or evals. No em dash. Repo text in English. Commit messages without AI co-author lines.
- The intake JSON keeps reading the old list shape (4.7); the new shape is a dict `{"items": [...], "rounds": [...]}`.

## Review Focus

1. A patient reply that affirms a danger sign while denying another ("no fever but I can't breathe") must show the emergency answer - Task 4 test `test_denial_mixed_affirmed_term_still_emergency`.
2. An extractor quote that is not in the patient's words must not fill an item - Task 2 test `test_extract_rejects_invented_quote`.
3. A double-tapped "Not sure" must not skip two items - Task 3 test `test_not_sure_stale_round_is_ignored`.
4. The interview must end even when every remaining item is unaskable - Task 1 test `test_next_item_none_when_all_skipped`.
5. A question the patient asks mid-interview (not a reply) ends the interview as today - Task 3 test `test_new_question_mid_interview_hands_off`.

---

### Task 1: Interview state (pure)

**Files:**
- Create: `backend/interview.py`
- Test: `tests/test_interview.py`

**Interfaces:**
- Produces:
  - `@dataclass Item(label: str, habits: list[int], safety: bool, status: str = "unknown", quote: str = "", attempts: int = 0)`; `status` in `ITEM_STATUSES = ("unknown", "answered", "unsure", "skipped")`.
  - `@dataclass Interview(items: list[Item], rounds: list[dict])`; round dict `{"asked": str, "item": int, "reply": str}`.
  - `is_interview(intake) -> bool` (dict with an `items` list).
  - `load_interview(intake) -> Optional[Interview]`, `dump_interview(iv: Interview) -> dict`.
  - `next_item(iv: Interview) -> Optional[int]`: first unknown safety item, else first unknown item; `None` when no unknown item.
  - `done(iv: Interview, max_rounds: int) -> bool`: `next_item is None or len(rounds) >= max_rounds`.
  - `fill(iv, index: int, quote: str, sources: list[str]) -> bool`: sets answered only if the normalised quote (case, whitespace) is a substring of one of `sources`.
  - `mark(iv, index: int, status: str)`.
  - `so_far(iv) -> list[dict]` -> `[{"label", "status", "quote"}]` for answered and unsure items, quote shortened per Global Constraints.
  - `summary_text(iv) -> str`: "Known: ...\nNot sure: ...\nNot answered: ..." (lines omitted when empty).

- [ ] **Step 1: Write the failing tests**: `test_round_trip_dump_load`, `test_old_list_intake_is_not_interview`, `test_next_item_safety_first`, `test_next_item_none_when_all_skipped`, `test_done_at_max_rounds`, `test_fill_accepts_substring_case_and_space_insensitive`, `test_fill_rejects_quote_not_in_sources`, `test_so_far_shortens_long_quote_at_word_boundary` (a 70-character quote -> ends with "..." and is a prefix of the quote), `test_summary_text_lists_three_groups`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_interview.py -q` - FAIL (module missing).
- [ ] **Step 3: Implement** the interfaces above in `backend/interview.py`.
- [ ] **Step 4: Run** the same command - PASS.
- [ ] **Step 5: Commit** `feat: interview item state`.

### Task 2: Interview model calls

**Files:**
- Modify: `backend/interview.py`, `backend/config.py`
- Test: `tests/test_interview.py`

**Interfaces:**
- Consumes: Task 1; `agent._check_clarify` clone-question rules (factor the clone-question check into `agent.check_clone_question(text: str, asked_before: list[str], allowed_drugs: frozenset[str]) -> None` raising `ValueError`, and have `_check_clarify` call it); `rules.find_quote`; the retrieval used by `retrieve` (`search.py`) for similar guidance entries.
- Produces:
  - `async similar_case_habits(session, llm, doctor_id: int, question: str, limit: int) -> tuple[list[GuidanceRule], list[GuidanceRule]]` -> (ask habits, warning habits) of the `limit` guidance entries most similar to `question`, non-suppressed, ordered by similarity.
  - `async build_items(llm, question: str, ask_habits, warning_habits, document_terms: list[str], allowed_drugs, *, max_items: int, max_safety: int) -> Optional[tuple[bool, list[Item]]]` -> `(reports_symptom, items)`; one JSON call `{"reports_symptom": bool, "items": [{"label", "habits": [ids]}]}`; drops items with unknown ids or a bad label; safety items appended per spec 4.1 step 4 and 4.7 item 7 (warning terms of the warning habits, then document terms, deduped, not already named by an item label, at most `max_safety`); `None` on a model or parse failure.
  - `async extract(llm, iv: Interview, original: str, reply: str) -> dict[int, str]` -> item index -> quote for non-safety unknown items; one JSON call `{"items": [{"index", "status", "quote"}]}`; every quote goes through `fill`, so an invented quote is dropped.
  - `async write_question(llm, item: Item, habit_texts: dict[int, str], asked_before: list[str], allowed_drugs, doctor_name: str, rephrase: bool) -> Optional[tuple[str, bool]]` -> `(question, is_doctor_words)`; first attempt prefers a habit question verbatim (a habit text ending "?", not suppressed); otherwise one call writing one question about the item, checked with `check_clone_question`; one repair call on failure; then the first habit question verbatim; else `None` (caller marks `skipped`, or `unsure` when `rephrase`).

- [ ] **Step 1: Write the failing tests** with FakeLLM: `test_build_items_drops_unknown_habit_ids`, `test_build_items_rejects_label_with_digit`, `test_build_items_adds_safety_items_from_warning_terms_capped_at_3`, `test_build_items_none_on_bad_json`, `test_extract_rejects_invented_quote`, `test_extract_fills_from_original_question`, `test_write_question_prefers_habit_verbatim`, `test_write_question_rejects_advice_then_falls_back_to_habit`, `test_write_question_none_without_habit_question`, `test_similar_case_habits_skips_suppressed` (DB fixture as in `tests/test_rules.py`).
- [ ] **Step 2: Run** - FAIL.
- [ ] **Step 3: Implement**; add the four settings to `config.py` with env names `INTERVIEW_CASES`, `INTERVIEW_MAX_ITEMS`, `INTERVIEW_MAX_SAFETY`, `INTERVIEW_MAX_ROUNDS`.
- [ ] **Step 4: Run** `tests/test_interview.py tests/test_agent.py -q` - PASS.
- [ ] **Step 5: Commit** `feat: interview items, extraction and questions from the doctor's habits`.

### Task 3: Route integration and "Not sure"

**Files:**
- Modify: `backend/routes_agent.py`, `backend/agent.py` (only to expose what the route needs), `backend/closing.py` (`intake_rounds` reads both shapes)
- Test: `tests/test_routes_agent.py`

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces:
  - `AskBody` gains `not_sure: bool = False` and `round: Optional[int] = None`.
  - Clarify payload for an interview question: `kind="clarify"`, `pending=True`, `questions=[{"text", "doctor": bool}]` (one entry), `interview={"so_far": so_far(iv), "round": n, "max_rounds": INTERVIEW_MAX_ROUNDS, "item": label}`.
  - First turn: when the agent result is the 4.7 ask-first clarify or a `not_covered` the picker would act on (same skip rules as today: warning terms in the message, medication change, conflicts), the route calls `similar_case_habits` and `build_items`; with `reports_symptom` true and at least one item, extract from the question, pick, write, and store the waiting row with `intake=dump_interview(iv)`; otherwise keep the agent's result unchanged.
  - Reply turn on an item-shape intake: emergency backstop first (Task 4 decides relaxation); otherwise run the reply judge (`_reply_judge`, with terms = safety items' terms + terms matched in this reply) and `extract`; apply safety items from `warning_signs` (present -> end with handoff; absent -> answered; unclear -> re-ask once as a rephrase, then `unsure`); if `replies_to_questions` is false -> today's "Patient did not answer the clone's questions" path; if not `done` -> next question (no agent call); if `done` -> call `run_agent` with the rounds as `pending` and clarify disallowed, then today's settle path (close gate, answer, handoff). The handoff body starts with `summary_text(iv)`.
  - `not_sure=True`: requires an item-shape pending intake whose `len(rounds)` equals `round`, else returns the current clarify payload unchanged and adds no messages; stores patient text `I'm not sure.`, marks the asked item `unsure`, no judge call.

- [ ] **Step 1: Write the failing tests** (FakeLLM scripted): `test_interview_first_turn_asks_one_question_and_stores_items`, `test_interview_reply_fills_item_and_asks_next_without_agent_call`, `test_interview_ends_and_closes_when_all_answered_and_no_signs`, `test_interview_safety_item_present_hands_off_with_summary`, `test_interview_unclear_safety_item_reasked_once_then_unsure`, `test_not_sure_marks_item_and_moves_on`, `test_not_sure_stale_round_is_ignored`, `test_new_question_mid_interview_hands_off`, `test_old_shape_intake_still_uses_4_7_flow`, `test_doctor_without_ask_habits_uses_4_7_flow`, `test_interview_build_failure_keeps_agent_result`.
- [ ] **Step 2: Run** `tests/test_routes_agent.py -q` - new tests FAIL.
- [ ] **Step 3: Implement.** Keep `ask` readable: put the interview turn in `async def interview_turn(...)` in `routes_agent.py` (or `interview.py` if it needs no route objects).
- [ ] **Step 4: Run** the full suite `.venv/bin/python -m pytest -q` - PASS.
- [ ] **Step 5: Commit** `feat: the clone interviews the patient one question at a time`.

### Task 4: Denied danger signs on interview replies

**Files:**
- Modify: `backend/safety.py`, `backend/routes_agent.py`
- Test: `tests/test_safety.py`, `tests/test_routes_agent.py`

**Interfaces:**
- Consumes: `detect_emergency`, `negated_terms`, the reply judge (Task 3 wiring).
- Produces: `denial_clears(message: str, matched: list[str], extra_terms: list[str], judged_absent: set[str]) -> bool` in `safety.py`: true only when no matched label is self-harm or suicide, every matched term is in `negated_terms(message, matched)`, every matched term is in `judged_absent`, and `detect_emergency(message_without_denied_clauses, extra_terms)` is empty. The route calls it only on an item-shape, unexpired pending intake; when it returns true the turn continues as an interview round; otherwise the fixed emergency answer as today.

- [ ] **Step 1: Write the failing tests**: `test_denial_clears_no_trouble_breathing`, `test_denial_mixed_affirmed_term_still_emergency` ("no fever but I can't breathe"), `test_denial_never_clears_self_harm` ("I don't want to hurt myself"), `test_denial_retraction_still_emergency` ("I did not faint. Well I did"), `test_denial_judge_unclear_still_emergency`; route: `test_interview_reply_denying_breathing_continues`, `test_first_message_denial_is_not_relaxed`, `test_old_shape_intake_denial_is_not_relaxed`. Keep every M4 probe string from `tests/test_safety.py` passing.
- [ ] **Step 2: Run** - FAIL.
- [ ] **Step 3: Implement.** The judge call for this check happens before the backstop decision on interview turns; a judge failure means not cleared.
- [ ] **Step 4: Run** the full suite - PASS.
- [ ] **Step 5: Commit** `feat: a denied danger sign does not end the interview`.

### Task 5: Patient page and doctor inbox

**Files:**
- Modify: `frontend/ask.js`, `frontend/app.css`, `frontend/api.js` (if the ask call is there), `frontend/doctor.js`
- Test: browser check with Playwright on a throwaway server and a DB copy (see the repro pattern in the ledger), plus any existing frontend tests under `tests/`

**Interfaces:**
- Consumes: Task 3 payload (`interview.so_far`, `round`, `max_rounds`) and the care-team row's `intake` dict.
- Produces: the clarify card shows "So far" (label: quote; "Not sure" items marked), one question with the doctor/clone tag as today, "Question N of up to M", and a "Not sure" button beside Send while the question is pending, which posts `{not_sure: true, round, doctor_id, question: ""}`; the doctor inbox shows Known / Not sure / Not answered above the rounds.

- [ ] **Step 1: Implement** matching the existing clarify card styles; no new colours.
- [ ] **Step 2: Verify** on a DB copy at port 8003: runny nose to Dr. Helen Okafor, answer two rounds, press "Not sure" once, deny breathing trouble; screenshots at 1100 and 390 px wide show no overlap, no horizontal scroll; doctor inbox shows the summary.
- [ ] **Step 3: Commit** `feat: interview card with So far line and Not sure button`.

### Task 6: PortalPal re-evaluation

**Files:**
- Modify: `scripts/eval_portalpal.py` (simulated patient answers only the asked item using `replies.json` facts; "Not sure" when the fact is missing), `README.md`
- Test: `tests/test_eval_portalpal.py`

- [ ] **Step 1:** Adjust the simulator and its tests; run them - PASS.
- [ ] **Step 2:** Run `--split test --repeats 1`, with memory only, gpt-4o-mini judge, wrapped in `scripts/run_with_spend.py`, cap $3.
- [ ] **Step 3:** Put warning handoff, paraphrase, benign closed, questions per conversation and drug violations next to the 2026-11-16 numbers in the README; commit `eval: PortalPal with the interview`.
