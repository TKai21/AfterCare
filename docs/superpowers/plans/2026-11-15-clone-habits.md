# Clone Habits, Ask-First Turns and Safety Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The clone learns each doctor's habits (what they ask, how they close, which signs need them) from their answers, asks the patient first, closes safe cases itself and hands the rest off with the answers collected; no delivered text names a drug the patient neither takes nor asked about; retrieval stops losing correctly ranked rules to a fixed floor.

**Architecture:** Habits are a second kind of `guidance_rules` row (`kind = "habit"`), read by the existing rules reader, never tag-gated, quote-only as evidence, all given to the clone each turn. Two new answer kinds, `clarify` and `close`, are validated by the server; whether a `close` is delivered is decided by a pure function over the reply, the habits and the document, failing closed into a handoff. A committed RxNorm-derived lexicon backs a drug-name check inside `validate`.

**Tech Stack:** Python 3, FastAPI, async SQLAlchemy (SQLite dev, MySQL prod), plain ES modules frontend, pytest with `tests/fake_llm.py`, OpenAI-compatible chat and embeddings.

**Spec:** `docs/superpowers/specs/2026-11-15-clone-habits-design.md`

## Global Constraints

- Work in the worktree `/Users/wutingkai/Desktop/USC/EE542/Final-clone-rules`, branch `clone-rules`; Python is `.venv/bin/python`, tests `.venv/bin/pytest -q -W error`.
- The emergency backstop (`detect_emergency`) runs first on every message, and an emergency answer never waits on a model call or on memory. Do not change how the backstop itself matches.
- New columns go in `db.ADDED_COLUMNS` (SQLite and MySQL) and in `sql/schema.sql`.
- Never use the em dash character in code, docs, UI text or commit messages; use "-".
- Commit messages: no AI co-author line.
- Every model call in a test uses `FakeLLM` (route a system prompt with `llm.route(PROMPT, responder)` when the call is not the clone's own turn).
- Evaluation runs need the real library index: `knowledge/index.sqlite3` must report 49 documents (`backend/manage_knowledge.py status`). Never commit `knowledge/index.sqlite3`.
- Defaults from the spec: `HABITS_IN_CONTEXT = 30`, clarify question cap `4`, `CLARIFY_TIMEOUT_HOURS = 24`, `GUIDANCE_TOP_K = 6`; handoff reasons exactly "Needs doctor after questions" and "Patient did not answer the clone's questions".
- PortalPal data is CC BY-NC 4.0: keep the citation and licence beside it in `eval/portalpal/`; it only ever enters temporary databases.

## Review Focus

1. A reply that negates one warning sign and reports another ("no fever, but I threw up twice") must hand off. Test in Task 11.
2. Drug names in real patient spelling: lower case, possessive, plural, brand in capitals ("tylenol", "Advil's", "ADVIL") are found; excluded diet words ("vitamin D", "water", "potassium") never are. Test in Task 2.
3. A habit quote that carries the original patient's name ("Hi Mike, have you...") is dropped. Test in Task 6.
4. A pending clarify belongs to one patient-doctor conversation: a message to another doctor's tab is not its reply. Test in Task 11.
5. An emergency term in the reply to a clarify gives the fixed emergency answer and turns the waiting question into an open one. Test in Task 11.

---

## Phase 0: True baseline

### Task 1: Calibration baseline with the real library index

**Files:**
- Modify: `scripts/calibrate_clone.py` (add `--out-dir`, default `eval/`, so a run never overwrites the committed v1/v2 files by accident; refuse to run when the library index has 0 documents)
- Create: `eval/calibration/baseline-index/` (runs, calibration, review copy, figure)
- Test: `tests/test_calibrate_clone.py`

**Interfaces:**
- Produces: `calibrate_clone.py --out-dir DIR` writing `DIR/clone_runs.json`, `DIR/clone_calibration.json`, `DIR/clone_risk_coverage.svg`. Later tasks rerun it into their own directories.

- [ ] **Step 1: Write the failing tests** `test_out_dir_writes_there_only(tmp_path)` (the default files under `eval/` are untouched, the three files exist in `tmp_path`) and `test_refuses_an_empty_library_index(tmp_path, capsys)` (exit code 1, message names the index path and "0 documents").
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_calibrate_clone.py` - Expected: the two new tests FAIL.
- [ ] **Step 3: Implement** the flag and the check (use `KnowledgeStore.status()`; its document count).
- [ ] **Step 4: Run** the test file, then the full suite. Expected: PASS.
- [ ] **Step 5: Run the real baseline** `.venv/bin/python scripts/calibrate_clone.py --out-dir eval/calibration/baseline-index`, apply the existing review overrides (`eval/clone_review.json`) as the script does; read 5 answers that now cite a library passage and note in the commit message whether the judge was right on them.
- [ ] **Step 6: Commit** `eval: clone calibration baseline with the real library index`

## Phase 1: Strict drug names

### Task 2: Drug lexicon

**Files:**
- Create: `scripts/build_drug_names.py` (network, run once), `data/drug_names.json`, `data/drug_name_exclusions.json`, `backend/drugnames.py`
- Test: `tests/test_drugnames.py`

**Interfaces:**
- Produces in `backend/drugnames.py`:
  - `class DrugLexicon` with `@classmethod load(path=DATA / "drug_names.json", exclusions=DATA / "drug_name_exclusions.json") -> DrugLexicon`
  - `DrugLexicon.find(text: str) -> set[str]`: canonical lower-case names found as whole words (case-insensitive; trailing "'s" and a plural "s" forgiven).
  - `DrugLexicon.equivalents(name: str) -> set[str]`: the name, its ingredients (for a brand) and its brands (for an ingredient), lower case.
  - `DrugLexicon.names_for_rxcuis(rxcuis: Collection[str]) -> set[str]`.
  - `get_lexicon() -> DrugLexicon` (module-level cached load).
- `data/drug_names.json` shape: `{"source": "...", "date": "...", "names": {"<lower name>": {"rxcui": "...", "tty": "IN"|"BN", "ingredients": ["<rxcui>", ...]}}}`.

- [ ] **Step 1: Write the failing tests**:
  - `test_every_food_table_nutrient_is_excluded`: for each nutrient key in `data/foods.json` (`sodium_mg` -> "sodium", `vitamin_d_mcg` -> "vitamin d"), the name is not found by `find`.
  - `test_measured_collisions_are_excluded`: `find("drink water, limit sodium, take vitamin D, iron and zinc; Today I feel ok; alli")` is empty.
  - `test_real_spelling_is_found` (Review Focus 2): `find("tylenol")`, `find("Advil's")`, `find("ADVIL")`, `find("two aspirins")` give `{"tylenol"}`, `{"advil"}`, `{"advil"}`, `{"aspirin"}`.
  - `test_brand_and_ingredient_equivalents`: "acetaminophen" in `equivalents("tylenol")`; "colcrys" in `equivalents("colchicine")`.
  - `test_whole_words_only`: `find("naproxenate")` and `find("colchicines are")` behave as whole words (the second finds "colchicine").
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_drugnames.py` - Expected: FAIL (module missing).
- [ ] **Step 3: Implement `scripts/build_drug_names.py`**: RxNav `GET /REST/Prescribe/allconcepts.json?tty=IN+BN`, then for each BN `GET /REST/rxcui/{rxcui}/related.json?tty=IN` (cache responses on disk under the scratch dir; generic User-Agent, no personal data). Write `data/drug_names.json`. Write `data/drug_name_exclusions.json` as `{"nutrients": [...from foods.json], "words": ["water", "calcium", "potassium", "sodium", "caffeine", "oxygen", "vitamin d", "iron", "magnesium", "glucose", "zinc", "today", "alli", ...]}`; review the list by grepping the lexicon for every word in `backend/safety.py` STOP_WORDS and every food name in `data/foods.json` and add each collision.
- [ ] **Step 4: Implement `backend/drugnames.py`** as in Interfaces (one compiled alternation over names sorted longest first).
- [ ] **Step 5: Run** the test file and the full suite. Expected: PASS.
- [ ] **Step 6: Commit** `feat: drug name lexicon from RxNorm's prescribable subset`

### Task 3: Drug names in validation

**Files:**
- Modify: `backend/agent.py` (`validate`, `run_agent`, `_Turn`), `backend/routes_agent.py` (pass allowed names)
- Test: `tests/test_agent.py`, `tests/test_routes_agent.py`

**Interfaces:**
- Consumes: `get_lexicon()`, `DrugLexicon.find/equivalents/names_for_rxcuis` (Task 2).
- Produces:
  - `allowed_drug_names(items: list[ExtractedItem], rxcuis: Collection[str], question: str, lexicon: DrugLexicon) -> set[str]` in `backend/drugnames.py`: medication items' names (non-rejected), names of the RxCUIs, names the question uses; each expanded with `equivalents`.
  - `prescribed_drug_names(items, rxcuis, lexicon) -> set[str]`: the same without the question (used by Task 11).
  - `validate(raw, evidence, allowed_drugs: frozenset[str] = frozenset(), lexicon: DrugLexicon | None = None)`: when `lexicon` is given, every lexicon name in the answer must be in `allowed_drugs` or appear in a cited `document` evidence's content; otherwise `ValueError(f"Names a drug the patient does not take: {name}. ...")`.
  - `run_agent(..., allowed_drugs: frozenset[str] = frozenset())` passes them to `validate` with `get_lexicon()`.

- [ ] **Step 1: Write the failing tests** in `tests/test_agent.py`:
  - `test_unprescribed_drug_from_doctor_advice_is_repaired_then_falls_back`: evidence has a doctor rule "I prefer naproxen first"; allowed `{"colchicine"}`; the model answers "I prefer naproxen [1]." twice; result is the fallback, and the repair message names "naproxen".
  - `test_drug_the_patient_asked_about_may_be_named`: allowed built by `allowed_drug_names(..., question="Can I take aspirin?")`; answer "No aspirin while taking naproxen [1]." with naproxen prescribed passes.
  - `test_brand_in_question_allows_its_ingredient`: question "Can I take Tylenol?", answer names "acetaminophen" from a library passage; passes.
  - `test_drug_named_in_a_cited_document_paragraph_passes`: document paragraph "Do not take ibuprofen."; cited; passes.
  - `test_fixed_texts_are_not_checked`: the emergency and fallback answers pass through unchanged.
- [ ] **Step 2: Run** them - Expected: FAIL.
- [ ] **Step 3: Implement** as in Interfaces; `routes_agent.ask` computes `allowed_drug_names(items, library_rxcuis, body.question, get_lexicon())` before the model calls.
- [ ] **Step 4: Add** `test_ask_blocks_volunteered_naproxen` to `tests/test_routes_agent.py` (end to end through `/api/ask` with FakeLLM scripted to name naproxen twice; the reply is the fallback and a care-team question exists).
- [ ] **Step 5: Run** the suite. Expected: PASS.
- [ ] **Step 6: Run evaluations**: `scripts/eval_clone.py --out-dir eval/clone/drug-check` and `scripts/calibrate_clone.py --out-dir eval/calibration/drug-check`. Expected: unprescribed recommendations 0 in the preference report; list calibration questions whose outcome changed against `eval/calibration/baseline-index` in the commit message.
- [ ] **Step 7: Commit** `feat: answers may not name a drug the patient neither takes nor asked about`

## Phase 2: Retrieval floor

### Task 4: Top K with a floor and a margin

**Files:**
- Modify: `backend/config.py` (`guidance_floor`, `guidance_margin`, `guidance_top_k`; `GUIDANCE_MIN_SIMILARITY` is read as `GUIDANCE_FLOOR` when the new variable is unset, so existing `.env` files keep working), `backend/guidance.py` (`retrieve`), `backend/routes_agent.py`, `backend/drafts.py`, `scripts/eval_clone.py` (`retrieval_check`; `RETRIEVAL_LIMIT` keeps reading the default `limit`), `scripts/calibrate_clone.py` if it calls `retrieve` (every caller), `README.md`
- Create: `scripts/sweep_guidance_floor.py`, `eval/calibration/floor-sweep.json`
- Test: `tests/test_guidance.py`, `tests/test_sweep_guidance_floor.py`

**Interfaces:**
- Produces: `retrieve(session, llm, patient, document, items, query, *, doctor_id, floor: float, margin: float, limit: int = 6) -> list[tuple[Unit, float]]` (advice rules and whole entries only; habits never). Keeps hits with `score >= floor and score >= best - margin`, best first, at most `limit`.

- [ ] **Step 1: Write the failing tests**: `test_margin_drops_hits_far_below_the_best` (scores 0.40, 0.38, 0.20 with floor 0.1, margin 0.1 keep the first two), `test_floor_still_applies` (best 0.08 with floor 0.1 returns nothing), `test_habit_rules_are_never_retrieved` (a habit rule with the highest score is absent; needs Task 5's column, so mark it `xfail(strict=True)` until Task 5 lands and remove the mark there), `test_old_env_name_sets_the_floor` (config).
- [ ] **Step 2: Run** - Expected: FAIL.
- [ ] **Step 3: Implement** `retrieve` and config; update both callers.
- [ ] **Step 4: Implement `scripts/sweep_guidance_floor.py`**: on `eval/clone_questions.json` split `calibration`, in a temporary world built like `calibrate_clone.py`'s (memory from `eval/clone_guidance.json`, read into rules), for floor in 0.10..0.45 step 0.05 and margin in 0.05..0.30 step 0.05: for `expect == "answer"` questions whose facts come from guidance, whether a rule of the needed entry is returned; for `expect == "escalate"` questions, the mean number of advice rules returned. Choose the pair that keeps every answerable question's rule with the fewest escalate rules (ties: higher floor). Print and write `eval/calibration/floor-sweep.json` with the same numbers on the `test` split and on `eval/clone/cases.json` (own-dimension rule returned, per case and doctor). Test with FakeLLM: the chosen pair follows the rule on a two-question fixture.
- [ ] **Step 5: Run the sweep** with the real models; set the defaults in `config.py` to the chosen pair; README "Doctor clone" section documents the three variables and the sweep numbers.
- [ ] **Step 6: Run evaluations** into `eval/clone/floor` and `eval/calibration/floor`.
- [ ] **Step 7: Commit** `feat: guidance retrieval keeps the top rules near the best score instead of a fixed floor`

## Phase 3: Habits

### Task 5: Habit columns

**Files:**
- Modify: `backend/models.py` (`GuidanceRule`), `backend/db.py` (`ADDED_COLUMNS`), `sql/schema.sql`
- Test: `tests/test_db.py` (or the existing column-upgrade test file)

**Interfaces:**
- Produces on `GuidanceRule`: `kind: str` (`"advice"` default, server default `'advice'`), `habit_type: Optional[str]` (`ask` | `close` | `warning`), `warning_terms: list` (JSON, default `[]`), `suppressed: bool` (default False: a doctor-deleted habit kept so a re-read does not bring it back).

- [ ] **Step 1: Write the failing test** `test_existing_guidance_rules_table_gains_habit_columns`: create a SQLite DB with the old `guidance_rules` DDL and one row, run `init_db`, the row reads back with `kind == "advice"`, `warning_terms == []`, `suppressed is False`.
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement**. **Step 4: Run** the suite (remove Task 4's xfail mark; it must now PASS). 
- [ ] **Step 5: Commit** `feat: guidance rules can be habits`

### Task 6: Reading habits

**Files:**
- Modify: `backend/rules.py` (`RULES_PROMPT`, `RuleDraft`, `validate_rules`, `build_rules`, `find_conflicts`), `tests/fake_llm.py` (`sentence_rules` returns `"kind": "advice"`)
- Test: `tests/test_rules.py`

**Interfaces:**
- Produces: `RuleDraft(statement, quote, kind="advice", habit_type=None, warning_terms=())`; `validate_habit(draft: RuleDraft, names: Collection[str], lexicon: DrugLexicon) -> bool`; `build_rules` stores habits; a suppressed habit whose normalised statement equals a new draft's is kept suppressed and the draft is not added again. `find_conflicts` compares habits only with the same doctor's habits and never with labels.
- Prompt reply shape: `{"rules": [{"kind": "advice"|"habit", "habit_type": "ask"|"close"|"warning"|null, "statement": "...", "quote": "...", "warning_terms": ["..."]}]}`.

- [ ] **Step 1: Write the failing tests** (FakeLLM route for `RULES_PROMPT` returning the JSON under test):
  - `test_habits_are_stored_with_their_type`: an `ask`, a `close` and a `warning` habit plus one advice rule are stored with the right `kind`/`habit_type`; warning terms that are not spans of the quote are removed.
  - `test_habit_with_a_number_is_dropped` ("call if fever above 101"); `test_habit_naming_a_drug_is_dropped`; `test_habit_changing_a_medication_is_dropped` ("stop your naproxen"); `test_medication_change_question_is_allowed` ("Have you had any recent changes in your medications?").
  - `test_habit_quoting_a_patient_name_is_dropped` (Review Focus 3): quote "Hi Mike, have you had any fever?" with the patient's username "mike" in `names`.
  - `test_suppressed_habit_is_not_read_back`.
  - `test_habits_are_not_checked_against_labels`.
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement**; `build_rules` gets `names` from the source question's patient (`User.username`, `User.display_name`) and the doctor. **Step 4: Run** the suite - PASS.
- [ ] **Step 5: Commit** `feat: the rules reader learns the doctor's habits`

### Task 7: Habits in the clone's context

**Files:**
- Modify: `backend/guidance.py` (new `doctor_habits`), `backend/agent.py` (`Evidence.kind` gains `"habit"`, `_Turn.habit_entry`, a "How I usually respond" block in the first message, system prompt), `backend/routes_agent.py`
- Test: `tests/test_agent.py`, `tests/test_routes_agent.py`

**Interfaces:**
- Produces: `doctor_habits(session, doctor_id: int, limit: int = 30) -> list[GuidanceRule]` (active entries only, not suppressed, deduplicated by normalised statement, most recently updated entry first); `run_agent(..., habits: list[GuidanceRule] = ())`; evidence for a habit: kind `"habit"`, title "Dr. X's usual practice", content and `check_text` = the quote only, `guidance_id` = its entry, and a new `Evidence.habit_type: Optional[str] = None` set from the rule (Task 10 validates on it).

- [ ] **Step 1: Write the failing tests**: `test_habits_are_listed_with_quote_only` (the first user message contains the habit quote and not the rest of the entry's answer, which holds "500 mg"); `test_habit_from_another_patients_entry_reaches_this_patient`; `test_habit_evidence_grounds_no_numbers` (an answer citing only a habit with "2" fails number grounding); `test_inactive_entry_habits_are_left_out`.
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement**. **Step 4: Run** the suite - PASS.
- [ ] **Step 5: Commit** `feat: the clone sees its doctor's habits on every turn`

### Task 8: Habits on the doctor page

**Files:**
- Modify: `backend/routes_doctor.py` (guidance views carry habits; `DELETE /api/doctor/guidance/rules/{rule_id}` suppresses a habit of the doctor's own; lift the two `GENERAL_TAG_ERROR` checks), `frontend/doctor.js`, `frontend/app.css`, `README.md`
- Create: `scripts/reread_rules.py USERNAME` (re-reads every active entry of one doctor with `rules.index_entry`; prints rules and habits per entry; one model call per entry, plus conflict checks)
- Test: `tests/test_routes_doctor.py`

**Interfaces:**
- Produces: guidance view field `habits: [{"id", "habit_type", "statement", "quote", "warning_terms"}]`; `GET /api/doctor/guidance` also returns `habits_by_type` for the "How I respond" section.

- [ ] **Step 1: Write the failing tests**: `test_general_answer_without_tags_is_saved` (200, entry stored, its advice rules never retrieved for a patient); `test_doctor_suppresses_own_habit` (204; another doctor gets 404); `test_guidance_list_groups_habits_by_type`; `test_reread_rules_reads_every_active_entry` (FakeLLM; inactive entries untouched).
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement** the routes, then the UI: a "How I respond" section above the entries with three groups (Questions I ask / How I close / Signs I want to hear about), each habit with its quote and a delete button; the answer form shows "Without tags, only your habits from this answer will be used." when General is ticked and no tag is set.
- [ ] **Step 4: Run** the suite; then run the app (`seed_demo.py --confirm`, log in as `dr.ting`, answer the open question with a habit-style answer) and check the section renders at 375 px and 1280 px wide with no overlap.
- [ ] **Step 5: Commit** `feat: doctors see and remove the habits their clone learned`

## Phase 4: Ask first, then close or hand off

### Task 9: Term matching with negation

**Files:**
- Modify: `backend/safety.py`
- Test: `tests/test_safety.py`

**Interfaces:**
- Produces: `match_terms(message: str, terms: list[str]) -> list[str]` (the second half of `detect_emergency`, which now calls it; behaviour unchanged) and `negated_terms(message: str, terms: list[str]) -> set[str]`: the matched terms whose every occurrence sits in a clause (split on `.;,` and "but") that starts with or contains "no", "not", "never", "denies", "without", "n't" directly before the term's words (at most 2 words between).

- [ ] **Step 1: Write the failing tests**: `test_detect_emergency_unchanged` (run the existing emergency tests; add one asserting `detect_emergency` equals the old output on 5 messages); `test_simple_negations` ("No fever", "I don't have any vomiting", "without chest pain" negate their term); `test_mixed_clause_is_not_negated` (Review Focus 1: "no fever, but I threw up twice" negates "fever" and not "vomiting"/"threw up"); `test_negation_far_away_does_not_count` ("No, the pain got worse and I have a fever" does not negate "fever").
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement**. **Step 4: Run** the suite - PASS.
- [ ] **Step 5: Commit** `feat: match warning terms and recognise plain negations`

### Task 10: `clarify` and `close` in the agent

**Files:**
- Modify: `backend/agent.py` (`KINDS`, `SYSTEM_PROMPT` rules 9 and new rules for clarify and close, output contract with `warning_signs`, `validate`, `AgentResult.warning_signs`, `run_agent(..., pending: PendingClarify | None = None)`)
- Test: `tests/test_agent.py`

**Interfaces:**
- Produces:
  - `@dataclass PendingClarify: question: str; asked: str` (the original question and the clarify text).
  - `KINDS = ("answer", "not_covered", "emergency", "out_of_scope", "clarify", "close")`.
  - `validate` rules: `clarify` cites at least one `habit` evidence whose `habit_type == "ask"` (carry `habit_type` on `Evidence`) and only habit evidence; every sentence but at most one ends with "?"; at most 4 questions; no numbers; drug check; rejected when `pending` is set. `close` cites at least one `close` habit; same checks as `answer`; only allowed when `pending` is set.
  - `AgentResult.warning_signs: list[dict]` parsed from the final JSON (`[{"term": str, "status": "present"|"absent"|"unclear"}]`, missing means `[]`).
  - `AgentResult.replies_to_questions: Optional[bool]` from the final JSON field `"replies_to_questions"`, required (a bool) when `pending` is set, absent otherwise; a missing or non-bool value when pending fails validation.
  - Rule 9 wording: a question about the patient's own health or a new symptom is in scope; only questions unrelated to their health are out of scope.
- [ ] **Step 1: Write the failing tests**: `test_clarify_cites_ask_habits_only`, `test_clarify_with_five_questions_is_repaired`, `test_clarify_with_a_number_is_repaired`, `test_second_clarify_is_rejected_when_pending`, `test_close_needs_pending_and_a_close_habit`, `test_warning_signs_are_parsed`, `test_replies_to_questions_required_when_pending`, `test_new_symptom_is_in_scope` (prompt contains the rule 9 wording; a scripted `out_of_scope` for "My stomach hurts" still validates, so assert on the prompt text only), `test_pending_context_lists_original_question_and_asked`.
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement**. **Step 4: Run** the suite - PASS.
- [ ] **Step 5: Commit** `feat: the clone can ask first and close with the doctor's own words`

### Task 11: The reply turn, the close decision and the waiting question

**Files:**
- Create: `backend/closing.py`
- Modify: `backend/models.py` (`CareTeamQuestion.status` gains `waiting_on_patient`; new nullable `asked TEXT`, `patient_reply TEXT`, `stopped_by VARCHAR(100)` columns via `ADDED_COLUMNS` and `schema.sql`), `backend/routes_agent.py`, `backend/routes_care_team.py` (`question_view` carries the three fields)
- Test: `tests/test_closing.py`, `tests/test_routes_agent.py`

**Interfaces:**
- Consumes: `match_terms`, `negated_terms` (Task 9), `prescribed_drug_names`, `get_lexicon` (Tasks 2-3), `doctor_habits` (Task 7), `PendingClarify`, `AgentResult.warning_signs` (Task 10), `MEDICATION_CHANGE` (`agent.py`).
- Produces:
  - `close_blocker(original: str, reply: str, emergency_terms: list[str], doctor_warning_terms: list[str], document_warning_terms: list[str], warning_signs: list[dict], prescribed: set[str], lexicon: DrugLexicon) -> str | None` in `backend/closing.py`: `None` when a close may be delivered, else the reason shown to the doctor: "Emergency terms", "Warning sign: <term>" (a matched term not absent by both checks, in the original question or the reply), "Medication change reported", "Drug not on the prescription: <name>".
  - `async def pending_clarify(session, document_id: int, doctor_id: int) -> tuple[ChatMessage, CareTeamQuestion] | None`.
  - Route behaviour (in `ask`): see steps.

- [ ] **Step 1: Write the failing pure tests** in `tests/test_closing.py`: one per reason; `test_benign_reply_closes` ("Two days, no fever, no medication changes" with doctor terms fever/vomiting and the model saying both absent returns None); Review Focus 1 (`"no fever, but I threw up twice"` returns "Warning sign: ..."); `test_model_and_program_must_agree` (program negates, model says `unclear`: blocked); `test_missing_model_status_blocks`; `test_warning_in_first_message_blocks`; `test_prescribed_drug_in_reply_does_not_block`; `test_unprescribed_drug_in_reply_blocks`.
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement** `closing.py`.
- [ ] **Step 4: Write the failing route tests** in `tests/test_routes_agent.py`:
  - `test_clarify_creates_a_waiting_question` (status `waiting_on_patient`, `asked` set, payload `pending: true`, the confidence gate not called).
  - `test_benign_reply_closes_and_resolves` (status `resolved`, answer is the close text, payload kind `close`, gate called once).
  - `test_warning_reply_hands_off_with_summary` (status `open`, reason "Needs doctor after questions", `patient_reply` and `stopped_by` set, patient sees the fixed handoff sentence).
  - `test_new_question_hands_off_the_symptom` (reason "Patient did not answer the clone's questions"; the new question is answered).
  - Review Focus 4: `test_other_doctors_tab_is_not_the_reply`.
  - Review Focus 5: `test_emergency_in_reply_gives_emergency_answer_and_opens_question`.
- [ ] **Step 5: Run** - FAIL. **Step 6: Implement** the route: detect pending first (after the backstop); pass `pending` to `run_agent`; for `clarify` add the waiting question and set `payload.pending`; on a reply turn clear `pending` on the clarify message's payload, then: `close` and `close_blocker(...) is None` -> deliver and resolve; `close` blocked, `not_covered` or fallback -> handoff (update the waiting question to `open` with the summary fields; never add a second question); `answer` -> deliver and resolve the waiting question with reason kept; `replies_to_questions is False` (Task 10) -> deliver the turn's answer and hand off the waiting question with the second reason.
- [ ] **Step 7: Run** the suite - PASS.
- [ ] **Step 8: Commit** `feat: after its questions the clone closes safe cases and hands the rest to the doctor`

### Task 12: Doctor inbox for waiting and handed-off questions

**Files:**
- Modify: `backend/routes_doctor.py` (`list_questions`), `backend/drafts.py` (draft context), `frontend/doctor.js`, `frontend/app.css`
- Test: `tests/test_routes_doctor.py`, `tests/test_drafts.py`

**Interfaces:**
- Produces: `GET /api/doctor/questions` returns `{"open": [...], "waiting": [...], "answered": [...]}`; a `waiting_on_patient` question older than `CLARIFY_TIMEOUT_HOURS` (setting `clarify_timeout_hours`, default 24) is returned in `open` with `stopped_by = "Patient did not answer the clone's questions"` and is written back as `open` on that read.

- [ ] **Step 1: Write the failing tests**: `test_waiting_questions_are_listed_apart`, `test_waiting_question_times_out_on_read` (freeze `utcnow`), `test_open_count_excludes_waiting`, `test_draft_context_has_asked_and_reply`.
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement**; UI: a collapsed "Clone is asking (N)" list below the open questions; an open handed-off question shows three labelled blocks (Patient asked / My clone asked / Patient replied) and the stop reason as a small note.
- [ ] **Step 4: Run** the suite; check the inbox in the browser at 375 px and 1280 px.
- [ ] **Step 5: Commit** `feat: the doctor's inbox shows what the clone asked and what the patient replied`

### Task 13: Patient Ask page

**Files:**
- Modify: `frontend/ask.js`, `frontend/app.css`, `backend/routes_agent.py` (`answer_payload` sources for habit evidence)
- Test: `tests/test_routes_agent.py` (payload shape), browser check

**Interfaces:**
- Consumes: payload kinds `clarify` (with `pending`) and `close`; source kind `habit`.

- [ ] **Step 1: Write the failing test** `test_payload_marks_habit_sources` (source `kind == "habit"`, title "Dr. X's usual practice", content is the quote).
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement** the payload, then the UI: a pending clarify renders as the clone's questions with a quiet "Reply below" hint and the habit source chip; once answered the hint is gone; `close` renders like an answer. Use the existing answer article styles.
- [ ] **Step 4: Run** the suite; in the browser as `demo2` with Dr. Ting, report a new symptom, reply, and check both outcomes at 375 px and 1280 px.
- [ ] **Step 5: Commit** `feat: patients see the clone's questions and its close`

## Phase 5: Evaluation

### Task 14: Preference evaluation reads ask-first turns

**Files:**
- Modify: `scripts/eval_clone.py`, `scripts/clone_eval_score.py`, `tests/test_eval_clone.py`, `tests/test_clone_eval_score.py`

**Interfaces:**
- Produces: records carry `kind` values `clarify` and `close`; a `clarify` answer is judged for `first_ask` (its text is the answer under test) and counts as not blocked; the report adds a "Kinds" table (per doctor and dimension: answer, clarify, not_covered, other) and lists habits read per doctor (from the world).

- [ ] **Step 1: Write the failing tests**: `test_clarify_is_judged_for_first_ask`, `test_report_has_kinds_table`, `test_world_reports_habits_read`.
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement**. **Step 4: Run** the suite - PASS.
- [ ] **Step 5: Commit** `feat: the clone preference evaluation scores ask-first turns`

### Task 15: PortalPal evaluation

**Files:**
- Create: `eval/portalpal/syppm.csv` (downloaded once from `https://huggingface.co/datasets/PortalPal-AI/Patient-Message-Response-Drafting/resolve/main/syppm_huggingface.csv`), `eval/portalpal/README.md` (citation from the dataset card, CC BY-NC 4.0, what was changed), `eval/portalpal/split.json`, `eval/portalpal/replies.json`, `scripts/eval_portalpal.py`, `scripts/portalpal_data.py`
- Test: `tests/test_portalpal_data.py`, `tests/test_eval_portalpal.py`

**Interfaces:**
- Produces in `scripts/portalpal_data.py`: `load_rows() -> list[dict]` (`id` = row index as "pp-000".."pp-099", `message`, `ehr`, `response`); `split(rows, seed=542) -> {"memory": [60 ids], "tune": [15], "test": [25]}`; `ehr_document(ehr: str) -> tuple[str, list[dict]]` (document text and items: one `medication` item per active medication line, one `diagnosis`/problem item per problem).
- `replies.json`: `{"<id>": {"benign": str, "warning": str, "warning_sign": str}}` for tune and test ids; written once by `scripts/eval_portalpal.py --write-replies` with the strong model from the message and the physician's response (the warning reply must contain one sign the physician's response names, given as `warning_sign`), then read and fixed by hand; the run never generates replies.
- `scripts/eval_portalpal.py`: `--split tune|test`, `--repeats N` (default 3), `--out-dir` (default `eval/portalpal/results`), `--without-memory` also runs the no-memory condition (default on). World: one doctor `eval-portal` displayed "Portal physician" (never a real name), patients `eval-pp-NNN` with their EHR as the confirmed document; memory = each memory row's response saved through the doctor answer path (`POST /api/doctor/questions/{id}/answer` after creating the question, general, untagged), so rules and habits are read exactly as in the product. Turn 1 = the message; turn 2 = the benign or the warning reply (both, in separate conversations).
- Metrics in `results.md`: habits read (listed for review); turn 1 kind counts and ask-first rate; question recall (judge, `gpt-4o`: map the physician's questions and the clone's questions to categories duration, associated_symptoms, medication_changes, triggers, history, other; recall = shared / physician's); benign close rate; warning handoff rate (target 25/25 per repeat; every miss listed with its text); close fidelity (judge: same close as the physician: schedule and when to seek urgent care); drug check violations; each with and without memory.

- [ ] **Step 1: Write the failing data tests**: `test_split_is_fixed_and_disjoint` (60/15/25, same ids for the same seed), `test_ehr_document_lists_active_medications` (a fixture EHR string with two active medications gives two medication items; "*No Active Medications*" gives none), `test_replies_cover_tune_and_test_and_name_their_sign` (every id present; `warning_sign` occurs in `warning`; skipped until `replies.json` exists).
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement** `portalpal_data.py`; download the CSV; write `eval/portalpal/README.md`.
- [ ] **Step 4: Write the failing runner tests** (FakeLLM and a fake judge, three-row fixture): `test_memory_goes_through_the_answer_path` (habits exist for the eval doctor after the world is built), `test_warning_reply_must_hand_off` (a scripted `close` on a warning reply is counted as a miss and listed), `test_run_never_opens_the_dev_database` (as `test_eval_clone.py` does).
- [ ] **Step 5: Run** - FAIL. **Step 6: Implement** the runner and `--write-replies`.
- [ ] **Step 7: Write the replies** (`--write-replies`, real API), read all 40 pairs, fix any warning reply whose sign is not one the physician named and any benign reply that slips one in; commit.
- [ ] **Step 8: Tune** on `--split tune` (clarify cap, prompt wording only); then run `--split test` once.
- [ ] **Step 9: Run** the suite - PASS.
- [ ] **Step 10: Commit** `feat: PortalPal evaluation of the clone against one physician's replies`

### Task 16: Final runs, gate recalibration, README

**Files:**
- Create: `eval/clone/habits/`, `eval/calibration/habits/`
- Modify: `eval/clone_calibration.json` (recalibrated), `README.md` ("Doctor clone", "Doctor clone evaluation", "Calibrated confidence gate", "Status")

- [ ] **Step 1: Run** `scripts/eval_clone.py --out-dir eval/clone/habits`, `scripts/calibrate_clone.py --out-dir eval/calibration/habits`, and PortalPal test (already done in Task 15).
- [ ] **Step 2: Recalibrate** the gate on the calibration split of the new runs (alpha 0.1), write `eval/clone_calibration.json`, and report the test-split numbers.
- [ ] **Step 3: README**: one table per evaluation with every stage (baseline-index, drug-check, floor, habits); the floor sweep; PortalPal metrics with counts; the warning handoff result stated plainly; habits read for review; what is not measured.
- [ ] **Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `eval: clone habits results, recalibrated confidence gate, README`
