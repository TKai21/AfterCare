# AfterCare

AfterCare is an English web app for patients who go home after a hospital stay. The patient uploads the discharge document their doctor wrote (family medicine), reviews and confirms what the app extracted from it, and then asks an agent lifestyle and medication questions such as "is almond butter OK, and how much?". Every answer carries citations, and the doctor's document is always the top authority: the agent explains and applies it, and never overrides it. A daily checklist generated from the confirmed document is a secondary feature. Questions the agent cannot answer from approved sources are saved to a Care team list the patient can take to the next appointment.

AfterCare is a class project, not medical advice. In an emergency, call 911.

This is EE542 final project, Team 2, milestone 1. The "Status" section below lists plainly what is not built or not verified yet.

## Architecture

| Part | What it is |
|------|-----------|
| Backend | FastAPI, async SQLAlchemy. MySQL 8 in production (`asyncmy`), SQLite (`aiosqlite`) for local dev and tests |
| Auth | Sign up / log in with JWT; every query is scoped to the current user, another user's IDs return 404 |
| Frontend | Plain HTML/CSS/JS in `frontend/`, served by FastAPI. No framework, no build step |
| Models | Any OpenAI-compatible server: `/chat/completions` with standard `tools`/`tool_calls`, and `/embeddings` |
| Education library | Curated, allowlisted official sources (MedlinePlus, NIDDK, CDC, DailyMed, kidney.org), chunked and embedded into a SQLite vector index, exact cosine search with NumPy |
| Food table | `data/foods.json`: USDA FoodData Central nutrients plus oxalate categories from the UChicago oxalate table |
| Grounding | The server validates every answer. Citations must exist in the evidence of that request, and every number in an answer must appear in the cited evidence. A rejected answer gets one repair attempt that tells the model what failed (never for an emergency); if that fails too, the user gets a safe fallback and the question is added to the Care team list automatically |
| Safety | Local emergency-symptom detection (a static symptom list plus the document's own warning signs) shows an emergency banner above the answer. It never depends on the model: with no model configured, or when the model fails, an emergency message still gets the call-911 answer and banner |

The app has four tabs: Instructions (upload, then review and confirm the extracted items), Today (daily checklist), Ask (the agent), Care team (questions the app could not answer).

## Run the website

1. Install Python 3.10+. MySQL 8+ is only needed for the production setup; for a quick local run use SQLite (below).
2. Get the code. The project lives on the `aftercare` branch, and the repository root is the project root:

   ```bash
   git clone https://github.com/renny475/EE542-Labs.git
   cd EE542-Labs
   git checkout aftercare
   ```

3. From the project root:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   cp .env.example .env
   ```

4. In `.env`, set a unique `JWT_SECRET` (startup fails if it is empty or still the placeholder). Keep `.env` private; it is gitignored.
5. Choose a database.

   Quick local run with SQLite - set this in `.env`:

   ```
   DATABASE_URL=sqlite+aiosqlite:///./aftercare-dev.sqlite3
   ```

   An existing local SQLite dev DB from an earlier version must be deleted and recreated after upgrading (the new guidance tables and columns are not migrated).

   MySQL - edit the password in `sql/schema.sql`, run `mysql -u root -p < sql/schema.sql`, and put the same password in `DATABASE_URL`. The app also creates missing tables on startup. `schema.sql` uses `CREATE TABLE IF NOT EXISTS`, so a database created by an earlier version will not gain new columns (for example `discharge_documents.active_user_id`, or `care_team_questions.status`, `answer`, `answered_by` and `answered_at`, which replace the old `resolved` column). Drop and recreate it (upgrading a milestone 1 database).
6. Start the app from the project root. A relative SQLite path in `DATABASE_URL` resolves against the current directory; `RAG_DB_PATH` (`knowledge/index.sqlite3`) and `FOODS_PATH` (`data/foods.json`) resolve against the project root wherever you start it:

   ```bash
   uvicorn --app-dir backend app:app --host 0.0.0.0 --port 8000
   ```

7. Open `http://localhost:8000`, create an account, and log in.

## Model configuration

All settings live in `.env` and are read by `backend/config.py`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATABASE_URL` | MySQL on localhost | SQLAlchemy async URL |
| `JWT_SECRET` | none, required | Token signing secret |
| `JWT_EXPIRE_MINUTES` | 43200 | Token lifetime |
| `LLM_API_BASE` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | empty | Chat model name, for example `gpt-4.1-mini` (see `.env.example`) |
| `LLM_API_KEY` | empty | May stay empty for a self-hosted server |
| `EMBEDDING_API_BASE` | `LLM_API_BASE` | Embeddings endpoint base |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_API_KEY` | `LLM_API_KEY` | Embeddings key |
| `RAG_DB_PATH` | `knowledge/index.sqlite3` | Education library index |
| `RAG_MIN_SIMILARITY` | 0.25 | Minimum cosine similarity for a passage |
| `RERANK_ENABLED`, `RERANK_MODEL`, `RAG_MIN_RERANK` | off, `ms-marco-MiniLM-L-6-v2`, -4.5 | Library reranker, see [Library search](#library-search) |
| `GUIDANCE_MIN_SIMILARITY` | 0.45 | Minimum cosine similarity for a doctor guidance entry |
| `FOODS_PATH` | `data/foods.json` | Food table |
| `LABEL_FIXTURES` | empty | Dev only: `1` answers RxNav and DailyMed from the recorded `demo/labels` fixtures (offline). Leave it unset in production |

Without `LLM_MODEL` the app still starts and you can sign up, but uploading a document (extraction) and Ask return a "not configured" error (HTTP 503). The one exception is an emergency message in Ask: it still gets the call-911 answer and banner. A temporary model failure returns HTTP 502 and the page asks the patient to try again.

To use a self-hosted vLLM server, set `LLM_API_BASE` to its OpenAI-compatible URL (for example `http://host:8000/v1`) and `LLM_MODEL` to the served model name; `LLM_API_KEY` can be empty (no `Authorization` header is sent then, and `manage_knowledge.py index` only requires a key for `api.openai.com`). If you change `EMBEDDING_MODEL` or the embeddings endpoint, rebuild the index with `python backend/manage_knowledge.py index` (`status` shows whether the index matches the configured model).

## Knowledge library

`knowledge/manifest.json` lists the approved sources and `knowledge/sources/` holds one snapshot per source with its provenance. Only HTTPS sources on the allowlisted official hosts are accepted. Run from the project root:

```bash
python backend/manage_knowledge.py status    # what is indexed, and whether it matches EMBEDDING_MODEL
python backend/manage_knowledge.py collect   # fetch every manifest URL into knowledge/sources/ (network)
python backend/manage_knowledge.py index     # embed reviewed snapshots and curated summaries (paid embedding calls)
```

Collected snapshots are unreviewed; review each one and set `"reviewed": true` before indexing, as unreviewed snapshots are skipped.

The index `knowledge/index.sqlite3` is not in git: build it locally with `index`, which needs a working embeddings endpoint. Without it the agent still answers from the patient's own document and the food table, but not from the education library.

## Library search

A library search (the agent's `search_education_library` tool) works in four steps:

1. **Filter.** Only manifest sources and the approved drug labels for this patient's own drugs are eligible. The RxCUI filter runs in SQL before any scoring, so another patient's drug label never reaches BM25 or the reranker.
2. **Two candidate lists.** The top 20 passages by embedding cosine similarity, and the top 20 by BM25 keyword score (`backend/search.py`; IDF and average length come from the whole index and are rebuilt when the index changes).
3. **Fusion.** The two lists are merged by reciprocal rank fusion (k = 60) into one list without duplicates.
4. **Rerank.** With the reranker on, the first 20 fused passages are scored by a local cross-encoder (`ms-marco-MiniLM-L-6-v2`, run through ONNX Runtime, no PyTorch, no patient text leaves the server). The best 5 with a score of at least `RAG_MIN_RERANK` are returned.

**Fallback.** With the reranker off, missing, or failing (an exception, the wrong number of scores, a NaN), the search returns the fused order above the cosine floor `RAG_MIN_SIMILARITY` (this is the "hybrid" mode). A failure is logged once per store and never fails the patient's turn. `manage_knowledge.py status` and the startup log say whether the reranker is on, and why not.

The model is downloaded once, by hand; the server never downloads anything:

```bash
python backend/manage_knowledge.py fetch-reranker   # into models/<RERANK_MODEL>/, pinned revision, sha256-checked
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `RERANK_ENABLED` | off | `1` turns the reranker on (it needs the model files from `fetch-reranker`); any other value, or unset, leaves it off |
| `RERANK_MODEL` | `ms-marco-MiniLM-L-6-v2` | Reranker name from `reranker.RERANKERS` |
| `RAG_MIN_RERANK` | -4.5 | Lowest reranker score (a logit) a passage needs to be returned |

### Evaluation

`eval/library_questions.json` has 40 English questions: 32 that a library source answers (gout, kidney stones, heart failure and sodium, COPD, pneumonia, cellulitis, UTI, diabetes, blood pressure, drug doses from the manifest's DailyMed labels, test names, and patient-style paraphrases), and 8 the library cannot answer. Each answerable question lists every source that answers it, checked against the snapshot in `knowledge/sources/`. The questions alternate between a `tune` split (20, used only to pick `RAG_MIN_RERANK`) and a `test` split (20). Run it on demand against the real index and embedding API (it is not part of pytest):

```bash
.venv/bin/python scripts/eval_library.py --sweep         # calibrate RAG_MIN_RERANK on the tune split
.venv/bin/python scripts/eval_library.py --split test    # or --split all
```

hit@k and MRR are over the answerable questions (a hit is any listed source in the top k). "false" counts the no-answer questions that returned any passage. The rerank time is measured around the cross-encoder call only (20 passages), on the dev laptop (macOS, Apple silicon). Measured on 2026-11-14 with `text-embedding-3-small` and the 49 manifest documents (no drug labels approved, so no RxCUI was passed):

Test split (16 answerable, 4 no-answer):

| mode | hit@1 | hit@5 | MRR | false | rerank ms (median / worst) |
|------|------:|------:|----:|------:|---------------------------:|
| embedding | 0.750 | 1.000 | 0.856 | 4/4 | - |
| hybrid | 0.812 | 0.938 | 0.844 | 4/4 | - |
| rerank | 0.875 | 0.938 | 0.906 | 0/4 | 253 / 301 |

All 40 questions (32 answerable, 8 no-answer):

| mode | hit@1 | hit@5 | MRR | false | rerank ms (median / worst) |
|------|------:|------:|----:|------:|---------------------------:|
| embedding | 0.844 | 1.000 | 0.912 | 8/8 | - |
| hybrid | 0.875 | 0.969 | 0.898 | 8/8 | - |
| rerank | 0.938 | 0.969 | 0.953 | 0/8 | 244 / 351 |

**`RAG_MIN_RERANK` = -4.5.** The rule: the highest cut, from -10 to 10 in steps of 0.5, whose tune-split hit@5 equals hit@5 with no cut. On the tune split hit@5 is 1.000 from no cut up to -4.5 and drops to 0.938 at -4.0. No-answer false hits fall from 4/4 to 0/4 between -8.5 and -7.0, so -4.5 removes them with some margin.

**Ship decision: the reranker is off by default.** The ship rule (from the design spec) keeps rerank on only if, on the test split, it is not worse than embedding-only search on hit@5 and MRR and has no more no-answer false hits. Rerank wins on MRR (0.906 against 0.856), hit@1 and false hits (0 against 4), but loses on hit@5 (0.938 against 1.000): one question, "I get lightheaded when I stand up since starting my water pill", where embedding search ranks the furosemide (Lasix) label fifth and neither hybrid nor rerank returns it. So `RERANK_ENABLED` defaults to off and library search runs in hybrid mode. Set `RERANK_ENABLED=1` to use the calibrated reranker. Rerank did not lose to hybrid on the test split (equal hit@5, better MRR and false hits), so the alternative model `bge-reranker-base` was not tried.

Note that hybrid, today's default, also scores below embedding-only on the test split (hit@5 0.938, MRR 0.844).

Limits of this evaluation:

- The builder wrote the questions and labels; the team has not reviewed them yet. Some patient wording is borrowed from the teammate's 15 gout questions, which are about a discharge PDF that AfterCare does not search.
- 20 tune and 20 test questions are few: one question moves hit@5 by 0.06. The split limits overfitting of `RAG_MIN_RERANK` but does not remove it, since the same person wrote both halves.
- The no-answer questions are on topics far from the library (a cast, chemotherapy, cataract surgery). A near miss is harder: "What should I do if I miss a colchicine dose?" has no answer in today's index (no colchicine label is approved), yet rerank at -4.5 still returns other drugs' labels (best score -0.05). In the end-to-end check the agent correctly answered `not_covered` and sent the question to the doctor, with the reranker on and off.
- The worst reranker time (351 ms) is just above the 300 ms budget.
- The doctor guidance memory (`GUIDANCE_MIN_SIMILARITY`) is out of scope; it is unchanged and not evaluated here.

## Drug labels

When a patient confirms a document, each medication is resolved to its active ingredient (RxNav) and the current FDA label for that ingredient is fetched from DailyMed. A label starts as `pending`. A doctor reviews it on the "Drug labels" tab of the doctor page (`/doctor#labels`), can switch to another candidate label, and approves or rejects it. Only approved labels are used by the library. Approving a label adds it to the index at once, with no rebuild. When the server starts, it removes from the index any label that is not approved in the database. A full rebuild with `python backend/manage_knowledge.py index` re-adds every approved label; it reads them from `DATABASE_URL` and fails loudly if it cannot.

A label is shared by all patients but is used only for a patient whose confirmed document includes that drug. An approved colchicine label never reaches a patient who is not taking colchicine.

UpToDate and Micromedex are not used. Their licence terms do not allow reuse of the content in this way, and their APIs are sold to hospitals and EHR vendors, not to a project like this. DailyMed labels are public FDA text.

`scripts/seed_demo.py` resolves the demo drugs from recorded fixtures in `demo/labels`, so seeding needs no network. `demo2` gets colchicine and naproxen, and `demo` (with `--confirm`) gets tamsulosin and ibuprofen. All of them start `pending`, and every run of the seed resets them to `pending` again (also removing them from the index). The gout education pages (`medlineplus-gout`, `niams-gout`) are in the knowledge manifest. The NIAMS snapshot covers only the overview and causes of gout; treatment comes from the MedlinePlus page. To collect only some sources, run `python backend/manage_knowledge.py collect --only ID [ID ...]`; collecting without `--only` re-fetches every source and marks every snapshot unreviewed.

Demo walkthrough:

1. Run `python scripts/seed_demo.py --confirm`.
2. Log in as `dr.ting` (password `doctor-password-1`), open `/doctor#labels`, and approve colchicine.
3. Log in as `demo2` and ask "I forgot my colchicine this morning, what should I do?". The answer can cite the colchicine label.

To try the "Drug labels" tab without the network, start the server with `LABEL_FIXTURES=1`; lookups then come from `demo/labels`, which covers only the four demo drugs.

## Food data

`data/foods.json` is generated by `data/build_foods.py` from USDA FoodData Central (SR Legacy items, patient-facing servings) and `data/oxalate.json`, whose category words are copied verbatim from the UChicago oxalate table. Foods missing from that table get oxalate "unknown" and the agent says it does not know rather than estimating.

```bash
FDC_API_KEY=... python data/build_foods.py [--cache DIR]      # from the FDC API
python data/build_foods.py --sr-legacy-csv DIR                # offline, from the unzipped SR Legacy CSV download
```

## Demo documents

`demo/cases/*.json` holds eight fictional family-medicine discharge cases: pneumonia, cellulitis, uti, copd, heart_failure, diabetes_htn, kidney_stones, gout. `demo/out/` has the generated `.txt` and `.pdf` for each, marked "FICTIONAL - FOR DEMO". No real patient data is used anywhere.

Dates in the documents are relative to the day you generate them. Regenerate before a demo:

```bash
python demo/generate_demo.py --days-ago 2
```

To see the Instructions review and Today screens without a model, `scripts/seed_demo.py` creates the user `demo` / `demo-password-1` in the database named by `DATABASE_URL`, with a fictional kidney-stones document waiting for review. It also seeds the doctor clone demo (see below). It is dev only and resets everything the demo users own when re-run.

```bash
python scripts/seed_demo.py            # pending review
python scripts/seed_demo.py --confirm  # confirmed, checklist generated
```

## Doctor clone

Accounts have a role: `patient` (sign up on the site) or `doctor`. A doctor cannot sign up; create one with `scripts/create_doctor.py`. There are three fictional doctors, each with an AI clone, and each doctor uses the doctor page at `/doctor`.

In the Ask tab the patient sees one tab per doctor. Each tab is its own conversation with that doctor's AI clone, and the clone's replies are labeled "AI clone". The clone answers only from the patient's discharge document, the education sources, and that doctor's own earlier advice (cited as "Dr. Name's advice"). A question the clone cannot answer goes to that doctor's queue, and the patient can also ask that doctor directly. Each answer a doctor gives becomes guidance for that doctor's clone and shows up in the patient's Care team tab.

```bash
python scripts/create_doctor.py USERNAME "Display Name"   # prompts for the password twice
```

`scripts/seed_demo.py` creates three doctors, all with the password `doctor-password-1`:

| Username | Doctor |
| --- | --- |
| `dr.ting` | Dr. Ting Kai |
| `dr.renny` | Dr. Aaron Renny |
| `dr.patel` | Dr. Parin Patel |

It also creates `demo` / `demo-password-1` (kidney stones) and `demo2` / `demo-password-2` (gout case, always confirmed). Each doctor has one general entry (Dr. Ting: colchicine timing; Dr. Renny: tamsulosin and stent discomfort; Dr. Patel: naproxen with food) and one open question in their queue; Dr. Ting also has a patient-specific entry for `demo2`. All seeded answers are invented for the demo. Embeddings need a configured model; without one the entries are saved unsearchable and can be retried from the doctor page.

Upgrading from the single-doctor version: questions and messages now belong to a doctor, so recreate the dev database (delete `aftercare-dev.sqlite3`) and run `python scripts/seed_demo.py --confirm` again.

On MySQL, the same upgrade adds the columns `chat_messages.doctor_id` and `care_team_questions.doctor_id`, which `schema.sql` (`CREATE TABLE IF NOT EXISTS`) does not add to existing tables. Drop and recreate the database with `sql/schema.sql`, then run the seed again.

Demo walkthrough:

1. Run `python scripts/seed_demo.py --confirm` so `demo`'s document is confirmed (asking needs a confirmed document).
2. Log in as `demo`, open the Ask tab, and pick a doctor tab. Ask a question nobody has answered; it goes to that doctor's queue. You can also press "Ask Dr. Name directly" under a doctor tab, or use "Ask a doctor directly" in the Care team tab.
3. Log in as that doctor (`dr.ting`, `dr.renny` or `dr.patel`, password `doctor-password-1`) at `/doctor`, answer it, and tick "General advice" (suggested tags can be edited).
4. Log in as `demo2`, pick the Dr. Ting Kai tab, and ask a similar gout question, for example "My big toe still hurts a lot after my colchicine, can I take one more?". The AI clone answers and cites Dr. Ting's advice. The same question from `demo` gets no doctor source.

The gate: a general entry only reaches a patient whose discharge document or confirmed instructions mention every drug tag and at least one condition tag as whole words (alternative names such as "colchicine / Colcrys" match), and an untagged general answer is refused. Anything unclear fails closed, and a patient-specific entry is only ever used for that patient and document.

The setting `GUIDANCE_MIN_SIMILARITY` (default 0.45) is the minimum cosine similarity for retrieving an entry that passes the gate.

## Tests

```bash
pytest
```

The unit suite uses SQLite and a scripted fake model, so it needs no network and no key. Currently 664 tests pass (`pytest -q -W error`).

## Status: not done or not verified yet

- The automated tests use a scripted fake model. The real chat and embedding models have only been tried by hand (for example the library evaluation and the colchicine question in [Library search](#library-search)).
- The education index (`knowledge/index.sqlite3`) is built locally only (it is not in git) and holds no approved drug labels yet.
- The library evaluation set (see [Library search](#library-search)) has not been reviewed by the team yet. The end-to-end browser test is not written.
- The MySQL path (`sql/schema.sql`, `asyncmy`) has not been run against a real MySQL server; local runs and tests use SQLite.
- The Android TWA wrapper from the reference project is not ported.
- Out of scope for this milestone: follow-up summary generation, OCR for scanned PDFs (they return "No readable text found. Scanned PDFs need OCR first."), and languages other than English.

## Project layout

```
README.md  pytest.ini  .env.example
backend/     FastAPI app: app.py (create_app), config.py, auth.py, models.py, db.py,
             documents.py, extraction.py, checklist.py, agent.py, foods.py, safety.py,
             knowledge.py, manage_knowledge.py, llm.py, routes_*.py, requirements.txt
frontend/    index.html + login.js (login), app.html + app.css + api.js (shell),
             review.js, today.js, ask.js, careteam.js (tabs)
knowledge/   manifest.json, sources/*.json (index.sqlite3 is built locally)
data/        foods.json, oxalate.json, build_foods.py
demo/        generate_demo.py, cases/*.json, out/ (generated .txt and .pdf)
scripts/     seed_demo.py, eval_library.py (library search evaluation)
eval/        library_questions.json
sql/         schema.sql (MySQL)
tests/       unit tests and the fake LLM
docs/        design spec
```
