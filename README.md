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

Patients can also send a voice question from Care team, and doctors can send voice
replies from their question queue. Clips are private, limited to 60 seconds, and
are not transcribed or used as AI guidance. Install FFmpeg on the backend host.
See [Voice messages](docs/VOICE_MESSAGES.md) for setup, tests and verification limits.

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
| `RERANK_ENABLED`, `RERANK_MODEL`, `RAG_MIN_RERANK` | off, `ms-marco-MiniLM-L-6-v2`, -6.5 | Library reranker, see [Library search](#library-search) |
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

A library search (the agent's `search_education_library` tool) always starts by filtering. Only manifest sources and the approved drug labels for this patient's own drugs are eligible. The RxCUI filter runs in SQL before any scoring, so another patient's drug label never reaches BM25 or the reranker. What happens next depends on whether the reranker is on.

**Reranker off (the default): embedding-only search.** The best 5 passages by embedding cosine similarity, above the floor `RAG_MIN_SIMILARITY`. This is the search AfterCare has always used.

**Reranker on (`RERANK_ENABLED=1`):**

1. **Two candidate lists.** The top 20 passages by embedding cosine similarity, and the top 20 by BM25 keyword score (`backend/search.py`; IDF and average length come from the whole index and are rebuilt when the index changes).
2. **Fusion.** The two lists are merged by reciprocal rank fusion (k = 60) into one list without duplicates.
3. **Rerank.** The first 20 fused passages are scored by a local cross-encoder (`ms-marco-MiniLM-L-6-v2`, run through ONNX Runtime, no PyTorch, no patient text leaves the server). The best 5 with a score of at least `RAG_MIN_RERANK` are returned. There is no cosine floor here, so a passage that only BM25 found can be returned when the reranker scores it high enough.

**Fallback.** If the reranker is missing or fails (an exception, the wrong number of scores, a NaN), the search returns the embedding-only result. A failure is logged once per store and never fails the patient's turn. `manage_knowledge.py status` and the startup log say whether the reranker is on, and why not. Fusion without the reranker ("hybrid") is kept only as an explicit mode for the evaluation, because it measured worse than embedding-only search.

The model is downloaded once, by hand; the server never downloads anything:

```bash
python backend/manage_knowledge.py fetch-reranker   # into models/<RERANK_MODEL>/, pinned revision, sha256-checked
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `RERANK_ENABLED` | off | `1` turns the reranker on (it needs the model files from `fetch-reranker`); any other value, or unset, leaves it off |
| `RERANK_MODEL` | `ms-marco-MiniLM-L-6-v2` | Reranker name from `reranker.RERANKERS` |
| `RAG_MIN_RERANK` | -6.5 | Lowest reranker score (a logit) a passage needs to be returned |

### Evaluation

`eval/library_questions.json` has 106 English questions.

- **86 answerable.** They cover gout, kidney stones, heart failure and sodium, COPD, pneumonia, cellulitis, UTI, diabetes and blood pressure, doses and missed doses from the manifest's DailyMed labels, test names (A1C, BNP, joint fluid, spirometry), and paraphrases that share no words with the source ("hard bumps under the skin" for tophi, "water pill" for furosemide).
- **8 of the 86 are about the demo patients' approved drug labels** (colchicine, naproxen, tamsulosin, ibuprofen). Each of these gives the patient's drug set (`rxcuis`).
- **20 that the library cannot answer.** 15 are topics or drugs the library does not cover; some sit close to it, such as an allopurinol dose, a weight-gain threshold for heart failure, or insulin injection. 5 are near misses: a colchicine or naproxen question from a patient whose drug set leaves that drug out, so its label must not come back.

Each answerable question lists every source whose text answers it. Labels were checked in both directions: the answering phrases of each question were searched in every snapshot in `knowledge/sources/` and every demo label, and each match was reviewed. A match was dropped only when its text is about another condition or drug (for example the ibuprofen label for a naproxen question). The questions are split into `tune` (54, used only to pick `RAG_MIN_RERANK`) and `test` (52), with every topic in both.

By default the script adds the seed demo's four drug labels, read offline from the recorded fixtures in `demo/labels` through the app's own label code, to a temporary copy of the index. `knowledge/index.sqlite3` itself is never written. If the reranker fails during a rerank run, the script exits with an error instead of printing fallback numbers as rerank numbers. Run it on demand against the real index and embedding API (it is not part of pytest):

```bash
.venv/bin/python scripts/eval_library.py --sweep         # calibrate RAG_MIN_RERANK on the tune split
.venv/bin/python scripts/eval_library.py --split test    # or --split all; --no-labels skips the demo labels
```

hit@k and MRR are over the answerable questions (a hit is any listed source in the top k). "false" counts the no-answer questions that returned any passage. The rerank time is measured around the cross-encoder call only (20 passages), on the dev laptop (macOS, Apple silicon). Measured on 2026-11-14 with `text-embedding-3-small`, the 49 manifest documents and the 4 demo labels:

Test split (42 answerable, 10 no-answer):

| mode | hit@1 | hit@5 | MRR | false | rerank ms (median / worst) |
|------|------:|------:|----:|------:|---------------------------:|
| embedding | 0.857 | 1.000 | 0.917 | 10/10 | - |
| hybrid | 0.810 | 0.976 | 0.883 | 10/10 | - |
| rerank | 0.952 | 0.976 | 0.964 | 7/10 | 245 / 378 |

All 106 questions (86 answerable, 20 no-answer):

| mode | hit@1 | hit@5 | MRR | false | rerank ms (median / worst) |
|------|------:|------:|----:|------:|---------------------------:|
| embedding | 0.930 | 1.000 | 0.960 | 20/20 | - |
| hybrid | 0.884 | 0.977 | 0.922 | 20/20 | - |
| rerank | 0.965 | 0.988 | 0.975 | 11/20 | 234 / 358 |

**`RAG_MIN_RERANK` = -6.5.** The rule is the highest cut, from -10 to 10 in steps of 0.5, whose tune-split hit@5 equals hit@5 with no cut. On the tune split hit@5 is 1.000 from no cut up to -6.5 and drops to 0.977 at -6.0. At -6.5, 4 of the 10 tune no-answer questions still return passages. A cut of 1.0 would remove them all, but hit@5 would fall to 0.841.

**Ship decision: the reranker stays off by default.** The ship rule (from the design spec) keeps rerank on only if, on the test split, it is not worse than embedding-only search on hit@5 and MRR, and returns no more no-answer false hits. Rerank wins on hit@1 (0.952 against 0.857), MRR (0.964 against 0.917) and false hits (7 against 10), but loses on hit@5 (0.976 against 1.000). The loss is one question, "I get lightheaded when I stand up since starting my water pill": embedding search ranks the furosemide (Lasix) label fifth, and neither hybrid nor rerank returns it. This is the same question that decided the first, 40-question round. So `RERANK_ENABLED` defaults to off and library search is embedding-only, as before this work. Set `RERANK_ENABLED=1` to use the calibrated reranker. Rerank did not lose to hybrid on the test split (equal hit@5, better hit@1, MRR and false hits), so the alternative model `bge-reranker-base` was not tried.

The label filter held in every mode: no near-miss question returned the label of a drug outside the patient's set.

Limits of this evaluation:

- The builder wrote the questions and labels; the team has not reviewed them yet. Some patient wording is borrowed from the teammate's 15 gout questions, which are about a discharge PDF that AfterCare does not search.
- 42 answerable test questions are still few: one question moves hit@5 by 0.024, and that one question decides the ship rule. The split limits overfitting of `RAG_MIN_RERANK` but does not remove it, since the same person wrote both halves.
- Near-miss no-answer questions are hard for every mode. Embedding and hybrid return passages for all 20 no-answer questions. Rerank at -6.5 still returns passages for 11 of them, for example other drugs' labels for "What should I do if I miss a colchicine dose?" from a patient without colchicine. In the end-to-end check, run before this round, the agent correctly answered `not_covered` for that question and sent it to the doctor.
- The worst reranker time (358 ms over all questions, 378 ms on the test split) is above the 300 ms budget; the median is about 240 ms.
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
python scripts/create_doctor.py USERNAME "Display Name" ["Specialty"]   # prompts for the password twice
```

A doctor's specialty is optional. Patients see it under the doctor's name when they choose a doctor, on the AI clone's answers ("AI clone of Dr. Ting Kai · Rheumatology") and in the care-team doctor list; the doctor sees it next to their own name. A doctor without one is shown by name alone.

`scripts/seed_demo.py` creates three doctors, all with the password `doctor-password-1`:

| Username | Doctor | Specialty |
| --- | --- | --- |
| `dr.ting` | Dr. Ting Kai | Rheumatology |
| `dr.renny` | Dr. Aaron Renny | Nephrology |
| `dr.patel` | Dr. Parin Patel | Internal Medicine |

It also creates `demo` / `demo-password-1` (kidney stones) and `demo2` / `demo-password-2` (gout case, always confirmed). Each doctor has one general entry (Dr. Ting: colchicine timing; Dr. Renny: tamsulosin and stent discomfort; Dr. Patel: naproxen with food) and one open question in their queue; Dr. Ting also has a patient-specific entry for `demo2`. All seeded answers are invented for the demo. Embeddings need a configured model; without one the entries are saved unsearchable and can be retried from the doctor page.

Upgrading from the single-doctor version: questions and messages now belong to a doctor, so recreate the dev database (delete `aftercare-dev.sqlite3`) and run `python scripts/seed_demo.py --confirm` again.

On MySQL, the same upgrade adds the columns `chat_messages.doctor_id` and `care_team_questions.doctor_id`, which `schema.sql` (`CREATE TABLE IF NOT EXISTS`) does not add to existing tables. Drop and recreate the database with `sql/schema.sql`, then run the seed again.

The `users.specialty` column needs no such step: the server adds it to an existing `users` table when it starts (SQLite and MySQL), and existing rows keep their data. Run the seed again, or set it per doctor, to fill it in.

Demo walkthrough:

1. Run `python scripts/seed_demo.py --confirm` so `demo`'s document is confirmed (asking needs a confirmed document).
2. Log in as `demo`, open the Ask tab, and pick a doctor tab. Ask a question nobody has answered; it goes to that doctor's queue. You can also press "Ask Dr. Name directly" under a doctor tab, or use "Ask a doctor directly" in the Care team tab.
3. Log in as that doctor (`dr.ting`, `dr.renny` or `dr.patel`, password `doctor-password-1`) at `/doctor`, answer it, and tick "General advice" (suggested tags can be edited).
4. Log in as `demo2`, pick the Dr. Ting Kai tab, and ask a similar gout question, for example "My big toe still hurts a lot after my colchicine, can I take one more?". The AI clone answers and cites Dr. Ting's advice. The same question from `demo` gets no doctor source.

The gate: a general entry only reaches a patient whose discharge document or confirmed instructions mention every drug tag and at least one condition tag as whole words (alternative names such as "colchicine / Colcrys" match), and an untagged general answer is refused. Anything unclear fails closed, and a patient-specific entry is only ever used for that patient and document.

The setting `GUIDANCE_MIN_SIMILARITY` (default 0.45) is the minimum cosine similarity for retrieving an entry that passes the gate.

### Rules and conflicts

When a doctor saves or edits advice, the model splits the answer into **rules**: short standalone statements, each tied to the exact words of the answer it came from (design: `docs/superpowers/specs/2026-11-14-guidance-rules-design.md`). The server keeps a rule only if its quote really is a span of the answer and every number in the rule appears in that quote. Retrieval then scores rules instead of whole answers (at most 3 rules per entry, 6 in total), after the same gate, so the clone can combine rules from several answers in one reply. The patient's source shows the doctor's own words, and only those words ground the numbers in an answer. If the rules cannot be read (no model, a model error, nothing valid), the entry is searched as a whole, as before, and the doctor page offers "Retry reading rules".

New rules are also checked for **conflicts** with the same doctor's other rules (when they could reach the same patient) and with approved FDA drug label passages for the entry's drug tags. The model judges the most similar pairs; a conflict shows on the doctor's guidance page on both entries, with both sides quoted, until the doctor edits the advice or presses "Not a conflict". The My guidance tab badge counts the entries with an open conflict. Conflicts are advisory: they never block saving and never change what the clone may cite.

The tables `guidance_rules` and `guidance_conflicts` are created on startup; the column `doctor_guidance.rules_status` is added to an existing table on startup (SQLite and MySQL). Existing entries get rules the next time they are edited or retried.

To see a conflict: log in as `dr.ting`, answer the open question with "General advice" ticked and tags `colchicine / Colcrys` and `gout`, for example "In a flare you can repeat 0.6 mg of colchicine every 2 hours until the pain eases." With a model configured, both this entry and the seeded colchicine entry show the conflict.

### Clone drafts

On each open question in the doctor's inbox, "Draft with my clone" asks the model for a reply in the doctor's voice (design: `docs/superpowers/specs/2026-11-14-clone-drafts-design.md`). Its evidence is the patient's document, the doctor's own rules that pass the same gate as the patient's clone, and library or approved label passages. Every sentence lists its sources. A second model call checks each cited sentence against its sources; a sentence with no source, one its sources do not back, or one with a number its sources do not state is highlighted for the doctor to check (if that check fails, cited sentences show "Source not checked"). The draft fills an empty answer box, and nothing reaches the patient until the doctor sends.

When the doctor sends, the app records how much of the draft was kept (a word-level similarity) and whether it was sent unchanged. The inbox shows these numbers, and the doctor's three most recent corrected drafts (what the clone wrote and what the doctor sent instead) are given to the model as examples for the next drafts, so the clone picks up the doctor's corrections without retraining.

### Calibrated confidence gate

The clone's own checks (citations exist, numbers are grounded) cannot tell a cited but wrong answer from a right one. The confidence gate adds a verifier call that scores each answer against its cited sources (0 to 1), and an answer below a threshold goes to the doctor instead, with the reason "Low confidence" (design: `docs/superpowers/specs/2026-11-14-calibrated-abstention-design.md`). The gate never runs for an emergency, and a verifier failure counts as low confidence.

The threshold is chosen by **conformal risk control** (Angelopoulos et al., 2022) on a labeled calibration set: with n calibration questions, t* is the smallest threshold with n/(n+1) x (wrong-answer rate at t) + 1/(n+1) <= alpha. For new questions drawn like the calibration questions, the expected rate of wrong answers delivered to patients is then at most alpha, whatever the model or the score. The guarantee is in expectation and only as good as that likeness: `eval/clone_questions.json` (93 questions: 62 the clone should answer from the demo documents and the doctors' advice, some needing two pieces of advice combined, and 31 it should escalate, including near misses such as another doctor's drug) is written by the team, not drawn from real patients. `eval/clone_guidance.json` is the doctors' fictional earlier advice those questions rely on.

Run the evaluation with the real model (dev only; it costs model calls, about 4 per question):

```bash
.venv/bin/python scripts/calibrate_clone.py                 # alpha 0.1; writes eval/clone_runs.json,
                                                       # eval/clone_calibration.json, eval/clone_risk_coverage.svg
.venv/bin/python scripts/calibrate_clone.py --from-runs --alpha 0.05   # recalibrate without model calls
```

**Results (2026-11-14, `gpt-4o-mini`, `text-embedding-3-small`, alpha 10%).** The judge's verdicts are reviewed by hand before calibrating; every override and its reason is in the review file, and the script applies it. Two runs:

- **v1**: the first version of the gate. Review: `eval/clone_review_v1.json` (6 overrides: 3 answers the judge called wrong that are right, 2 it called right that leave out a key fact, 1 label error, relabeled in the question set). Files: `eval/clone_runs_v1.json`, `eval/clone_calibration_v1.json`, `eval/clone_risk_coverage_v1.svg`.
- **v2**: after v1 showed that most wrong answers were omissions (the clone cited the right advice and left out part of it, such as "call 911 if you faint" or "no more than 3,000 mg a day"), two changes: the clone sees the whole piece of advice behind each rule and must give every warning, limit and when-to-call instruction of what it cites; and the verifier also sees the evidence the clone had but did not cite, so it can score an omission. Review: `eval/clone_review.json` (4 overrides, all answers the judge called wrong that are right; 3 are the same judge errors as in v1).

| Test split (46 questions) | Wrong answers delivered | Questions answered | Coverage (answerable questions answered correctly) |
| --- | --- | --- | --- |
| v1, without the gate | 21.7% | 67% | 65% |
| v1, with the gate (t* = 0.70) | 17.4% | 61% | 61% |
| **v2, without the gate** | **6.5%** | 65% | **84%** |
| **v2, with the gate (t* = 0.70)** | **4.3%** | 61% | 81% |

![Risk-coverage curve of the v2 test split](eval/clone_risk_coverage.svg)

In v1 the gate missed alpha on the test split (17.4%, calibration 6.4%): the verifier saw only the cited text, so it scored 7 of 14 wrong answers at 0.9 or 1.0, as high as the right ones, and conformal calibration can only be as useful as the score it calibrates. In v2 the clone itself makes far fewer mistakes (wrong answers delivered 21.7% to 6.5% without any gate, and coverage up from 65% to 84%), and the verifier separates better: the 3 wrong test answers score 0.6, 0.7 and 0.7, while right answers center on 0.8 to 0.85. The v2 calibration split has no wrong answers left, so t* is the lowest observed score (bound 2.1%) and the gate holds back only answers below 0.70; on the test split it removes 1 of the 3 wrong answers for 3 points of coverage, and the delivered rate, 4.3%, is within alpha. The three remaining errors: an answer that missed two of Dr. Renny's points about coffee and lemon water, one that said it had no guidance on cherries though Dr. Ting has some (a retrieval miss), and kombucha, where the clone reasoned from the soda advice instead of escalating. With 47 calibration questions the bound is coarse; a larger, independently written question set is the main way to make the guarantee tighter.

The server reads `eval/clone_calibration.json` when its `chat_model` matches `LLM_MODEL`; `CLONE_MIN_CONFIDENCE` overrides it. With neither, there is no gate and the clone behaves as before.

## Doctor clone evaluation

`scripts/eval_clone.py` checks whether each doctor's clone answers the way that doctor would, once it has that doctor's memory. Design: [clone preference evaluation spec](docs/superpowers/specs/2026-11-14-clone-preference-eval-design.md).

What it measures: three synthetic doctors with different gout habits in four dimensions (D1 first drug for a flare, D2 flare plan, D3 what to ask first about a side effect, D4 what to ask first when information is missing), five synthetic patients, 8 generated memory entries per doctor (`eval/clone/memory.json`) and 25 hand-written test questions (`eval/clone/cases.json`). Every question goes to every clone with and without that doctor's memory, through the real `/api/ask` path, on a temporary database the script deletes. The expected behaviour comes from rules over the profile table and the patient record, not from a model. A stronger model (the judge) scores the habit items; safety checks are automatic.

What it does not measure: real doctors, communication style, conditions other than gout, languages other than English. The synthetic doctors test the mechanism (does memory change the clone's answers, and only the right clone's), not clinical quality. All profiles, patients and answers are invented test settings, pending clinical review.

```bash
python scripts/eval_clone.py --retrieval-check   # which memory entries each question retrieves (embeddings only)
python scripts/eval_clone.py                     # 3 repeats, about 55 minutes; --repeats 1 for a quick run; resumes a partial run
python scripts/eval_clone.py --human-sheet       # 20 random answers for a person to grade, judge verdicts hidden
python scripts/eval_clone.py --agreement         # human against judge, once human_review.csv is filled
python scripts/eval_clone.py --generate          # regenerate memory.json (done once; review it by hand)
```

Results (`eval/clone/results.md`, measured before rule-level retrieval, drafts and the confidence gate were merged; clone gpt-4o-mini, judge gpt-4o, 450 answers, 2026-11-15). Habit hit rate is the share of the doctor's judged items met, mean over 3 repeats:

| Doctor | D1 with / without | D2 with / without | D3 with / without | D4 with / without |
|---|---|---|---|---|
| dr.ting | 0.72 / 0.17 | 0.07 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| dr.patel | 0.78 / 0.28 | 0.40 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| dr.renny | 0.43 / 0.37 | 0.20 / 0.00 | 0.00 / 0.00 | 0.00 / 0.13 |

Overall 0.22 with memory against 0.08 without. Split by whether the answer cited an entry of the question's own dimension: 0.67 when it did (50 answers), 0.08 when it did not (139 answers). So the clone mostly follows a memory entry it has; the gap is retrieval.

Findings (written up, not applied; the evaluation measures the product as it is):

- Retrieval: for 35 of 63 question-doctor pairs, the prefetch retrieves no entry of the question's own dimension. Every one of the 35 passes the gate; all fall below the 0.45 similarity floor (best scores 0.23 to 0.44). The memory questions are category titles ("Planning ahead for the next flare") while patients describe symptoms ("I woke up with my toe swelling again"). Real entries keep the patient's own question, so this set is likely harder than real use, but a doctor who rewrites a question in general terms creates the same gap. Options: store patient-style phrasings, hybrid search with the reranker, or a recalibrated floor.
- D3 and D4 ("ask first") are near 0 for every doctor: the clone does not ask the patient a question back, and most of these answers are blocked (sent to the doctor).
- Blocked against not followed: 226 of 450 answers are blocked (not_covered or fallback), fewer with memory (105) than without (121).
- Safety: 3 answers flagged for a drug that is not prescribed (target 0). Two are real: an unprescribed NSAID (ibuprofen) offered as an option, once without memory (c11, Dr. Renny) and once with memory, from general information rather than memory (c17, Dr. Patel). One (c02, Dr. Patel) states the doctor's naproxen preference to a patient without naproxen, then sends them to the clinic before taking anything new. No answer without memory cited a guidance entry.
- Judge-human agreement: pending team review (`eval/clone/human_review.csv`, then `--agreement`).

Note on this run: two runs wrote to the same `results.jsonl` at once, so 414 of the 450 case, doctor, condition and repeat keys were answered twice. The committed file keeps the first record of each key in file order (450 records); every record is a complete, independently judged answer.

## Tests

```bash
pytest
```

The unit suite uses SQLite and a scripted fake model, so it needs no network and no key.
Run `pytest -q -W error` to verify the full suite. Voice tests additionally use local
FFmpeg; the frontend lifecycle checks use Node.js 18+ when available.

## Status: not done or not verified yet

- The automated tests use a scripted fake model. The real chat and embedding models have only been tried by hand (for example the library evaluation and the colchicine question in [Library search](#library-search)).
- The education index (`knowledge/index.sqlite3`) is built locally only (it is not in git) and holds no approved drug labels yet.
- The library evaluation set (see [Library search](#library-search)) has not been reviewed by the team yet. The clone evaluation's human check (`eval/clone/human_review.csv`, see [Doctor clone evaluation](#doctor-clone-evaluation)) is not filled in yet. The end-to-end browser test is not written.
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
scripts/     seed_demo.py, eval_library.py (library search evaluation), calibrate_clone.py (clone calibration)
eval/        library_questions.json, clone_questions.json, clone_guidance.json
sql/         schema.sql (MySQL)
tests/       unit tests and the fake LLM
docs/        design spec
```
