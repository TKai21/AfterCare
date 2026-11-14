# Drug Label Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a patient confirms a discharge document, look up each medication's FDA label (RxNorm + DailyMed) in the background, let a doctor approve it, and then make approved label text searchable only for patients who take that drug.

**Architecture:**
- A new `backend/druglabels.py` holds three things:
  - the outside-service client (`LabelClient`, with an injectable `httpx` transport);
  - the background jobs that write two new app-DB tables, `drug_labels` and `document_drugs`;
  - the helper that builds a patient's RxCUI set.
- The confirm route schedules the job as a FastAPI background task.
- Doctor routes approve labels, which calls `KnowledgeStore.add` with a new `rxcui` column.
- `KnowledgeStore.retrieve` filters label rows by the patient's RxCUI set in SQL, before ranking.
- The doctor page gets a "Drug labels" tab in a new `frontend/labels.js`.

**Tech Stack:** FastAPI, async SQLAlchemy (SQLite dev/tests, MySQL supported), httpx, sqlite3 + NumPy library store, plain JS ES modules, pytest with FakeLLM and `httpx.MockTransport`.

**Spec:** `docs/superpowers/specs/2026-11-13-drug-label-library-design.md` (read sections 4, 6, 7, 8 and 13 before any task).

## Global Constraints

- Never use the em dash character in code, copy, comments or docs; use "-".
- Tests never touch the real network. Every `create_app` in tests gets a `LabelClient` on a fixture or failing transport. `tests/conftest.py` must pass one so no existing test can reach RxNav or DailyMed.
- Only the medication name text, the dose text and RxCUIs/setids go to outside services. Never send usernames, document text or ids.
- Label text whose `drug_labels.status` is not `approved` never enters the library store.
- The library id of an approved label is `dailymed-rxcui-<rxcui>`. Its title is `DailyMed label: <ingredient> (<product title>)`. Its topic is `Medications`. Its URL is `https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=<setid>`.
- `drug_labels.status` is one of `fetching`, `pending`, `approved`, `rejected`, `no_label`, `failed`.
- `document_drugs.status` is one of `fetching`, `resolved`, `unresolved`, `failed`.
- The interrupted-job error text is exactly `Interrupted, try again`.
- The empty-section copy is exactly `No drug labels waiting.`
- Candidates: at most 5, newest first. Titles naming another active ingredient are excluded.
- Doctor label endpoints are doctor-only (403 for patients). Every doctor sees every label. A concurrent review conflict returns 409.
- `sql/schema.sql` stays in sync with `backend/models.py`.
- RxNav allows at most 20 requests per second. Use a 15 s timeout per request and run lookups one at a time.
- Frontend: plain ES modules, no new dependencies. `tests/test_frontend_syntax.py` must pass.

## Review Focus

1. **A medication item with an empty or odd `details.name`** (for example "Multivitamin" or "Insulin per sliding scale"). The row should become `unresolved` with no crash, and it should show in the doctor tab with "Try again". Test in Task 2.
2. **The same ingredient on two patients' documents, confirmed one after the other.** There should be exactly one `drug_labels` row, and both patients' sets should contain the RxCUI. Test in Task 2.
3. **A patient with no confirmed document, or a pending one, asks a question.** The set should be empty, no label passages should appear, and manifest passages should still appear. Test in Task 3.
4. **Approve while the embedding service fails.** The row should stay `pending`, the doctor should get a 502 with a readable message, and nothing should be added to the store. Test in Task 4.
5. **The doctor picks another candidate on a `rejected` label.** The label should be fetched again and become `pending` with the new setid and title. A pick on an `approved` label should return 409. Test in Task 4.

---

### Task 1: Label service client and fixtures

**Files:**
- Create: `backend/druglabels.py` (client part only)
- Create: `demo/labels/` (fixture responses, see Step 1)
- Modify: `backend/knowledge.py` (reuse `dailymed_sections`, `DAILYMED_SPL_XML`; no behavior change)
- Test: `tests/test_druglabels.py`

**Interfaces:**
- Produces:
  - `class LabelServiceError(Exception)`.
  - `@dataclass Resolution: ingredients: list[tuple[str, str]]` (rxcui, ingredient name, lower case) `; strength_rxcui: str | None`.
  - `class LabelClient(transport: httpx.AsyncBaseTransport | None = None)` with:
    - `async resolve(name: str, dose: str | None) -> Resolution`. Empty ingredients means unresolved.
    - `async candidates(rxcui: str, ingredient: str) -> list[dict]`. Each dict has `setid`, `title`, `published` (ISO date) and `labeler`.
    - `async label_text(setid: str) -> str`.
    - `aclose()`.
  - `LABEL_SECTIONS: list[str]` (LOINC codes).
  - `fixture_transport(directory: Path) -> httpx.MockTransport`.
  - `failing_transport() -> httpx.MockTransport` (every request raises `httpx.ConnectError`).

- [ ] **Step 1: Capture fixtures.**
  - Use the real network once, from a throwaway script that is not committed.
  - Save responses for the four demo drugs ("Tamsulosin" 0.4 mg, "Ibuprofen" 600 mg, "Colchicine" 0.6 mg, "Naproxen" 500 mg) under `demo/labels/`. Cover:
    - RxNav `approximateTerm.json` (term with the dose, and name alone);
    - `rxcui/{id}/related.json?tty=IN`;
    - DailyMed `spls.json?rxcui=` (first page);
    - one label XML per drug.
  - Trim each XML to the sections in `LABEL_SECTIONS` plus their parents, keeping it valid. Each fixture file stays under 60 KB.
  - Write `demo/labels/index.json` mapping the request path and query to the file. `fixture_transport` serves from it, and answers 404 for anything unmapped.
- [ ] **Step 2: Write failing tests** in `tests/test_druglabels.py`:
  - `test_resolve_colchicine_with_dose`: `Resolution.ingredients == [("2683", "colchicine")]` and `strength_rxcui` is not None.
  - `test_resolve_falls_back_to_name_only`: the dose term gets no candidate from a MockTransport, so the name-only term is used.
  - `test_resolve_unknown_name_is_empty`: `ingredients == []`.
  - `test_resolve_service_down_raises`: `failing_transport()` raises `LabelServiceError`.
  - `test_candidates_newest_first_max_five_no_combinations`: a MockTransport with 7 labels, including "IBUPROFEN AND FAMOTIDINE TABLET". The result has 5 labels in date order and none of them is the combination.
  - `test_label_text_has_dosage_and_warnings`: the colchicine fixture text contains "DOSAGE" and "WARNINGS" (case-insensitive) and is non-empty.
  - `test_only_names_leave_the_server`: record every request URL. Each query value is in {name, name plus dose, an RxCUI, a setid}.
- [ ] **Step 3: Run** `.venv/bin/pytest tests/test_druglabels.py -v`. Expect a FAIL (import error).
- [ ] **Step 4: Implement the client.** Rules:
  - Resolve uses `approximateTerm.json?term=<name> <dose>&maxEntries=1`, then the name alone if that finds nothing.
  - Take the candidate's rxcui, then `related.json?tty=IN` for the ingredients.
  - `strength_rxcui` is the candidate's rxcui when it differs from every ingredient rxcui.
  - `candidates` queries `spls.json?rxcui=` with the strength RxCUI when one was given. The caller passes it; see Task 2.
  - Drop a title that contains " AND " or "/" around another ingredient word, or any comma-separated list of ingredients.
  - Sort by `published_date` descending and keep 5.
  - `labeler` is the bracketed text at the end of the title.
  - Every HTTP or JSON error becomes `LabelServiceError`.
- [ ] **Step 5: Run the tests.** Expect a PASS. Also run the full suite: `.venv/bin/pytest -q`.
- [ ] **Step 6: Commit.** `git add backend/druglabels.py backend/knowledge.py demo/labels tests/test_druglabels.py && git commit -m "feat: RxNorm and DailyMed label client with offline fixtures"`

### Task 2: Tables, background jobs and the confirm hook

**Files:**
- Modify: `backend/models.py`, `sql/schema.sql`, `backend/druglabels.py`, `backend/routes_documents.py` (`confirm`), `backend/app.py` (`create_app(..., label_client=None)`, startup sweep), `tests/conftest.py`
- Test: `tests/test_druglabels.py`, `tests/test_routes_documents.py`

**Interfaces:**
- Consumes: Task 1 `LabelClient`, `fixture_transport`, `failing_transport`.
- Produces:
  - Models `DrugLabel` (`drug_labels`) and `DocumentDrug` (`document_drugs`), with the columns in spec section 6.
    - `DrugLabel.rxcui` is unique, String(20). `candidates` is JSON.
    - `DocumentDrug` is unique on (`extracted_item_id`, `rxcui`). `rxcui` is nullable.
  - `async resolve_document(session_factory, client: LabelClient, document_id: int) -> None`: creates and updates the `document_drugs` rows, and creates labels for new ingredients.
  - `async resolve_document_drug(session_factory, client, document_drug_id: int) -> None`: the retry of one row.
  - `async fetch_label(session_factory, client, label_id: int, setid: str | None = None) -> None`: fills candidates if they are empty, fetches the text, and sets `pending`, `no_label` or `failed`.
  - `async mark_interrupted(session_factory) -> int`: any `fetching` row in either table becomes `failed` with error `Interrupted, try again`.
  - `async patient_rxcuis(session, document_id: int | None) -> frozenset[str]`: the resolved RxCUIs of a confirmed document; empty for None or not confirmed.
  - `app.state.label_client`. `create_app` defaults it to `LabelClient()`.
- [ ] **Step 1: Write failing tests.**
  - `test_confirm_resolves_medications`:
    - Confirm the seed-style gout document in a test client with the fixture transport.
    - The `document_drugs` rows are colchicine and naproxen, both `resolved`.
    - The two `drug_labels` rows are `pending` with a title and text.
  - `test_confirm_succeeds_when_services_down`:
    - The client uses `failing_transport()`. Confirm returns 200 with the checklist.
    - The `document_drugs` rows are `failed` with an error.
  - `test_same_ingredient_two_patients_one_label`:
    - Two patients each confirm a colchicine document, giving one colchicine `drug_labels` row.
    - `patient_rxcuis` for both documents contains "2683".
  - `test_rejected_ingredient_not_requeued`: a label is set `rejected`, and a new patient confirms the same drug. The label stays `rejected`, and the new `document_drugs` row is still `resolved`.
  - `test_unresolvable_name_is_unresolved`: an item with name "Multivitamin" gives one `document_drugs` row, `unresolved`, rxcui None, and no label.
  - `test_patient_rxcuis_empty_without_confirmed_document`: covers `None` and a `pending_review` document.
  - `test_mark_interrupted`: rows set to `fetching` become `failed` with the error `Interrupted, try again`.
  - `test_no_real_network_in_default_client_fixture`: the conftest `client` app has a label client whose transport is not the default network transport.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_druglabels.py tests/test_routes_documents.py -v`. Expect a FAIL.
- [ ] **Step 3: Implement.**
  - The models, plus the matching `CREATE TABLE` statements in `sql/schema.sql`.
  - The job functions, each using its own session from `session_factory` and committing per row. Status `fetching` is written before each outside call.
  - In `confirm`, add a `background: BackgroundTasks` parameter. After the commit, call `background.add_task(resolve_document, request.app.state.session_factory, request.app.state.label_client, doc.id)`. Read `request` from the `Request` parameter.
  - `scripts/seed_demo.py` calls `confirm` directly. Give `background` a default of `None`. When it is None, skip scheduling; the seed handles lookups in Task 6.
  - Call `mark_interrupted` in the lifespan after `init_db`.
  - In `tests/conftest.py`, pass `label_client=LabelClient(fixture_transport(DEMO_LABELS))` in every `create_app`.
  - Jobs catch `LabelServiceError`, record the error on the row and never raise.
- [ ] **Step 4: Run the tests.** The targeted tests pass, then `.venv/bin/pytest -q` passes in full.
- [ ] **Step 5: Commit.** `git commit -m "feat: resolve confirmed medications to drug labels in the background"`

### Task 3: Library rxcui column, patient filter, agent wiring

**Files:**
- Modify: `backend/knowledge.py` (`KnowledgeStore.__init__`, `add`, `retrieve`, `index_sources`), `backend/manage_knowledge.py`, `backend/agent.py` (`_Turn`, `run_agent`, `search_education_library`), `backend/routes_agent.py`
- Test: `tests/test_knowledge.py`, `tests/test_agent.py`, `tests/test_routes_agent.py`

**Interfaces:**
- Consumes: Task 2 `patient_rxcuis`, `DrugLabel`.
- Produces:
  - `KnowledgeStore.add(record)` accepts an optional `record["rxcui"]`.
  - `KnowledgeStore.retrieve(query, limit=5, rxcuis: Collection[str] = ())`. Rows with a non-null rxcui that is not in `rxcuis` are excluded in SQL, before scoring.
  - `index_sources(store, directory=DATA, extra_records: list[dict] = ())`. Extra records are indexed alongside the reviewed snapshots.
  - `label_record(label: DrugLabel) -> dict` in `druglabels.py`. It builds the library record with the id, title, topic and URL from Global Constraints, plus `text`, `sha256`, `collected_at` and `rxcui`.
  - `async approved_label_records(database_url: str) -> list[dict]` in `druglabels.py`, for the CLI.
  - `run_agent(..., library_rxcuis: frozenset[str] = frozenset())` passes the set to `_Turn` and on to `store.retrieve`.
- [ ] **Step 1: Write failing tests.**
  - `test_store_adds_rxcui_column_to_old_index`:
    - Create an index file with the old 6-column `documents` table.
    - Open a `KnowledgeStore`; the column exists and the old rows are retrievable.
  - `test_retrieve_filters_labels_before_ranking`:
    - Index 6 label chunks for rxcui "2683" that score highest, and 1 manifest chunk.
    - `retrieve(q, limit=5, rxcuis=())` returns the manifest chunk.
    - `retrieve(q, rxcuis={"2683"})` returns label chunks.
  - `test_index_sources_keeps_extra_records`: a full rebuild with one extra record keeps it, and keeps its rxcui.
  - `test_search_library_passes_patient_rxcuis` (agent): a stub store records the `rxcuis` argument, which equals `library_rxcuis`.
  - `test_label_passage_numbers_are_checked` (agent): a label passage "0.6 mg" is cited by an answer saying "1.2 mg". The answer fails grounding, the same as for any library passage.
  - `test_ask_uses_active_document_rxcuis` (route): a patient with resolved colchicine gets `rxcuis == {"2683"}` on the store call. A patient without a confirmed document gets an empty set.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_knowledge.py tests/test_agent.py tests/test_routes_agent.py -v`. Expect a FAIL.
- [ ] **Step 3: Implement.**
  - For the migration, add the column after `CREATE TABLE IF NOT EXISTS` when `PRAGMA table_info(documents)` lacks `rxcui`.
  - `status()` is unaffected.
  - Filter SQL: `WHERE c.model=? AND (d.rxcui IS NULL OR d.rxcui IN (...))`.
  - `manage_knowledge.py index` passes `extra_records=await approved_label_records(settings.database_url)`. It prints `dailymed-rxcui-*` ids like the others.
  - The ask route computes `patient_rxcuis` before the read transaction ends.
- [ ] **Step 4: Run the tests.** The targeted tests pass, then the full suite passes.
- [ ] **Step 5: Commit.** `git commit -m "feat: per-patient drug label filter in the education library"`

### Task 4: Doctor label routes

**Files:**
- Modify: `backend/routes_doctor.py` (or create `backend/routes_labels.py` and include it in `app.py`, the implementer's choice)
- Test: `tests/test_routes_labels.py`

**Interfaces:**
- Consumes: Tasks 2-3 (`fetch_label`, `resolve_document_drug`, `label_record`, `app.state.store`, `app.state.label_client`).
- Produces the endpoints. Each label view is `{id, rxcui, ingredient, status, title, url, labeler, published, text, error, candidates, sources: [{medication, username}], reviewed_by_name, reviewed_at}`.
  - `GET /api/doctor/labels` returns `{"labels": [...], "unresolved": [{id, medication, username, status, error}]}`.
    - `labels` holds every row except `approved` and `rejected` ones older than 30 days. Pending rows come first.
    - `unresolved` holds the `document_drugs` rows with status `unresolved` or `failed`.
  - `POST /api/doctor/labels/{id}/approve` returns the view. It is 409 unless the row is `pending`, and 502 with "Could not add the label to the library. Try again." if embedding fails. The conditional update to `approved` happens only after `store.add` succeeds.
  - `POST /api/doctor/labels/{id}/reject` returns the view. It is 409 unless the row is `pending`.
  - `POST /api/doctor/labels/{id}/candidate` with `{setid}` returns the view after a synchronous `fetch_label`. It is 422 if the setid is not in `candidates`, and 409 if the row is `approved` or `fetching`.
  - `POST /api/doctor/labels/{id}/retry` (only `failed` or `no_label`, else 409) and `POST /api/doctor/document-drugs/{id}/retry` (only `failed` or `unresolved`, else 409). Both run synchronously and return the refreshed view.
- [ ] **Step 1: Write failing tests:**
  - `test_patient_gets_403` on every endpoint.
  - `test_list_shows_pending_with_sources`.
  - `test_approve_adds_to_store_and_is_retrievable_for_patient_only`: after approval, a kidney-stone patient's ask does not get colchicine text, and the gout patient's does.
  - `test_approve_embedding_failure_stays_pending` (Review Focus 4).
  - `test_concurrent_approve_and_reject_one_wins_409`: uses `file_client`.
  - `test_pick_candidate_on_rejected_refetches` and `test_pick_candidate_on_approved_409` (Review Focus 5).
  - `test_retry_failed_document_drug_resolves`: the row was failed with `failing_transport`, then the app's client is swapped to the fixture transport.
  - `test_unknown_id_404`.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_routes_labels.py -v`. Expect a FAIL.
- [ ] **Step 3: Implement.** Use the conditional-update pattern from `answer_question`: `update(...).where(id==, status=="pending")` and check `rowcount`.
- [ ] **Step 4: Run the tests.** The targeted tests pass, then the full suite passes.
- [ ] **Step 5: Commit.** `git commit -m "feat: doctor routes to review drug labels"`

### Task 5: Doctor page "Drug labels" tab

**Files:**
- Create: `frontend/labels.js`
- Modify: `frontend/doctor.html` (third tab `#labels`, with an icon in the same style), `frontend/doctor.js` (routing and badge only; rendering lives in `labels.js`), `frontend/app.css`
- Test: `tests/test_frontend_syntax.py` (picks up the new file automatically), `tests/test_frontend_caching.py` if it lists static files

**Interfaces:**
- Consumes: Task 4 endpoints.
- Produces: `export function showLabels(main, {onCount})` in `labels.js`. It renders and binds the tab. `onCount(n)` reports the pending count for the tab badge.

- [ ] **Step 1: Implement.** Requirements:
  - **Pending card.** It shows:
    - the ingredient (as the heading) and the product title;
    - the labeler, the dose form (from the title) and the published date;
    - "From: <medication line> (<username>)" for each source;
    - a `<details>` with the label text (pre-wrapped, max-height with scroll);
    - the buttons Approve (primary), Reject, and "Pick another label" (a select of the other candidates showing title and date, plus a Use button).
  - **Failed and unresolved entries.** They show the medication, the username, the error and "Try again".
  - **Section empty.** Show exactly `No drug labels waiting.`
  - Buttons are disabled while a request runs. Errors show inline on the card.
  - The tab badge shows the pending count, the same way the Questions badge does.
  - At 360 px wide there is no horizontal overflow, and the candidate select wraps under the title.
  - Escape every server string with `escapeHtml`.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_frontend_syntax.py tests/test_frontend_caching.py -q`. Expect a PASS.
- [ ] **Step 3: Check it by hand.** Run the server with a seeded DB (after Task 6, or a temporary seed), log in as `dr.ting` and open `/doctor#labels`. Check that a pending colchicine card renders at both desktop and 360 px widths. Record what was checked in the report.
- [ ] **Step 4: Commit.** `git commit -m "feat: drug labels review tab on the doctor page"`

### Task 6: Gout sources, demo seed, README

**Files:**
- Modify: `knowledge/manifest.json`, `backend/knowledge.py` (`ALLOWED_HOSTS` adds `www.niams.nih.gov`), `scripts/seed_demo.py`, `README.md`
- Test: `tests/test_seed_demo.py`, `tests/test_knowledge.py`

**Interfaces:**
- Consumes: Task 2 `resolve_document`, `fixture_transport`.
- [ ] **Step 1: Write failing tests:**
  - `test_seed_records_demo_drugs_offline`. Run `run_seed` against a temp DB with a `failing_transport` network guard, so any real request would fail the test. The seed must use `fixture_transport(DEMO_LABELS)` itself. Then:
    - `demo2` has resolved colchicine and naproxen;
    - the labels are `pending`;
    - `demo` (when `--confirm`) has tamsulosin and ibuprofen.
  - `test_manifest_has_gout_sources`: the ids `medlineplus-gout` and `niams-gout` are present, with URLs `https://medlineplus.gov/gout.html` and `https://www.niams.nih.gov/health-topics/gout`, and `official_url` accepts both.
- [ ] **Step 2: Run** them. Expect a FAIL.
- [ ] **Step 3: Implement.**
  - Add `run_seed(..., label_client=None)`, defaulting to `LabelClient(fixture_transport(DEMO_LABELS))`. After each confirm, await `resolve_document`.
  - Add the two manifest entries, with selectors picked by inspecting the pages.
  - Add a README section "Drug labels" covering:
    - the flow (confirm, review on `/doctor#labels`, approve, ask);
    - that labels apply only to patients taking the drug;
    - why UpToDate and Micromedex are not used (licence terms; their APIs are sold to hospitals and EHR vendors);
    - the walkthrough: `dr.ting` approves colchicine, then `demo2` asks "I forgot my colchicine this morning, what should I do?".
- [ ] **Step 4: Collect and index for real.**
  - Run `.venv/bin/python backend/manage_knowledge.py collect`.
  - Review the two gout snapshots and set `"reviewed": true`, keeping only the summary text.
  - Run `.venv/bin/python backend/manage_knowledge.py index`.
  - Record the chunk counts in the report.
- [ ] **Step 5: Run the full suite.** `.venv/bin/pytest -q` passes.
- [ ] **Step 6: Commit.** `git commit -m "feat: gout library sources, offline demo label seed, README"`
