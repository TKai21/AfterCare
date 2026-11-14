# Hybrid Library Search With a Local Reranker - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Education-library search finds candidates with BM25 and embeddings, merges them, reranks them with a local ONNX cross-encoder, and proves on an evaluation set that this is at least as accurate as today.

**Architecture:**
- `backend/search.py` (new, pure functions): tokenizer, BM25, reciprocal rank fusion.
- `backend/reranker.py` (new): the ONNX cross-encoder, its pinned download, and `load_reranker`.
- `KnowledgeStore.retrieve` gains three modes: `embedding` (today), `hybrid` (BM25 + embeddings, RRF) and `rerank` (hybrid then the cross-encoder). Production uses `rerank` when a reranker is loaded and `hybrid` otherwise. The doctor guidance memory is not touched.

**Tech Stack:** Python 3.13, FastAPI, SQLite library store, NumPy, `onnxruntime==1.32.0`, `tokenizers==0.23.3`, pytest with FakeLLM.

**Spec:** `docs/superpowers/specs/2026-11-14-hybrid-search-rerank-design.md`

**Branch:** create `library-rerank` from the current `drug-labels` HEAD; do not push.

## Global Constraints

- Scope is the education library only. `backend/guidance.py`, its settings (`GUIDANCE_MIN_SIMILARITY` 0.45) and its tests stay unchanged.
- The drug-label RxCUI filter runs before candidates are gathered. A label whose RxCUI is not in the patient's set is never returned in any mode.
- No patient text leaves the server for BM25 or reranking. The model download sends no patient data and uses a generic User-Agent.
- Default model `ms-marco-MiniLM-L-6-v2`, from `cross-encoder/ms-marco-MiniLM-L-6-v2` at revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, run through ONNX (no PyTorch).
- The server never downloads a model. Missing model files mean the server starts, search uses `hybrid`, and status reports the reranker off with a reason.
- Reranking adds at most about 300 ms per library search on the dev laptop. Measured: 20 pairs of about 400 characters take about 180 ms. So at most 20 candidates are reranked.
- A reranker failure never fails a search: fall back to `hybrid`, and log a warning once per store.
- Settings: `RERANK_MODEL` (default `ms-marco-MiniLM-L-6-v2`), `RERANK_ENABLED` (default on; `0` turns it off), `RAG_MIN_RERANK` (calibrated in Task 5).
- `status()` gains `"reranker": "on" | "off"` and `"reranker_note"`.
- Tests are offline and never load a real model, except tests marked to skip when `models/` lacks the files.
- House style: no em dash in code, comments, docs or commit messages. No co-author lines in commits.

## Review Focus

1. **The index is rebuilt by the CLI while the server runs.** The server's cached BM25 statistics must notice and rebuild on the next search, not keep serving stale or missing chunk ids. Test in Task 3.
2. **A query with no usable words** ("?", "the and of", only punctuation) gives no BM25 candidates. Search must still return embedding results, with no division by zero. Tests in Tasks 1 and 3.
3. **The reranker returns the wrong number of scores, NaN, or raises.** Treat it as a failure and fall back. Test in Task 3.
4. **A very long query** (thousands of characters) must be truncated, not crash ONNX. Test in Task 2 (real-model test, skipped without model files).
5. **Zero eligible candidates** (empty index, or everything filtered out) must return `[]` without calling the reranker. Test in Task 3.

---

### Task 1: BM25, tokenizer and RRF (`backend/search.py`)

**Files:**
- Create: `backend/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Produces:
  - `tokenize(text: str) -> list[str]`
  - `class BM25` with `BM25(docs: dict[int, str], k1: float = 1.5, b: float = 0.75)` and `scores(query: str, ids: Collection[int] | None = None) -> dict[int, float]`. `docs` maps chunk id to text. IDF and average length come from all `docs`. `ids`, when given, limits which documents are scored. Only documents with a score > 0 are returned.
  - `rrf(*rankings: Sequence[int], k: int = 60) -> list[int]`: fused order, best first, no duplicates. Ties keep the first appearance.

- [ ] **Step 1: Write the failing tests** in `tests/test_search.py`:

```python
def test_tokenize_lowercases_drops_stopwords_and_keeps_number_units():
    assert tokenize("Take 0.6 mg of Colchicine, then the next dose") == \
        ["take", "0.6", "mg", "0.6mg", "colchicine", "next", "dose"]
    assert tokenize("0.6mg") == ["0.6", "mg", "0.6mg"]
    assert tokenize("?? the and of") == []

def test_bm25_ranks_exact_term_first():
    bm = BM25({1: "drink water to prevent kidney stones",
               2: "colchicine 0.6 mg missed dose take when you remember",
               3: "gout is a kind of arthritis"})
    s = bm.scores("missed colchicine dose")
    assert max(s, key=s.get) == 2 and 1 not in s

def test_bm25_ids_limit_scoring_but_not_statistics():
    docs = {1: "colchicine dose", 2: "colchicine label", 3: "water"}
    assert set(BM25(docs).scores("colchicine", ids={2})) == {2}
    assert BM25(docs).scores("colchicine", ids={2})[2] == BM25(docs).scores("colchicine")[2]

def test_bm25_empty_query_and_empty_corpus():
    assert BM25({1: "water"}).scores("?? the") == {}
    assert BM25({}).scores("water") == {}

def test_rrf_merges_and_dedupes():
    assert rrf([1, 2, 3], [3, 4]) == [3, 1, 2, 4]
    assert rrf([], [5]) == [5]
```

- [ ] **Step 2: Run and see them fail.** Run `.venv/bin/pytest tests/test_search.py -q`. Expected: ImportError.

- [ ] **Step 3: Implement `backend/search.py`.**
  - Tokenizer:
    - Regex tokens `\d+(?:\.\d+)?|[a-z]+` on the lowercased text.
    - A number directly followed by a unit word (`mg mcg g kg ml l mmol meq iu units unit oz lb mmhg`) also yields the joined token (`0.6mg`), placed after the unit.
    - Drop stopwords: `a an and are as at be by for from has have i if in is it its me my of on or our so that the their them then there these they this to was we were what when where which who will with you your`.
  - Standard Okapi BM25.
  - A document of zero length is allowed. An empty corpus or an empty query returns `{}`.

- [ ] **Step 4: Run the tests again.** Run `.venv/bin/pytest tests/test_search.py -q`. Expected: all pass.

- [ ] **Step 5: Commit.** `git commit -m "feat: BM25, tokenizer and rank fusion for library search"`

---

### Task 2: The ONNX reranker, its pinned download, and dependencies

**Files:**
- Create: `backend/reranker.py`
- Modify:
  - `backend/requirements.txt` (add `onnxruntime==1.32.0`, `tokenizers==0.23.3`)
  - `.gitignore` (add `models/`)
  - `backend/manage_knowledge.py` (new `fetch-reranker` command)
  - `backend/config.py` (three settings)
- Test: `tests/test_reranker.py`

**Interfaces:**
- Produces:
  - `RERANKERS: dict[str, dict]`. Exactly one entry here:

```python
RERANKERS = {
    "ms-marco-MiniLM-L-6-v2": {
        "repo": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        "files": {  # repo path -> (local name, sha256)
            "onnx/model.onnx": ("model.onnx", "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a"),
            "tokenizer.json": ("tokenizer.json", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
        },
    },
}
```

  - `MODELS_DIR = PROJECT_DIR / "models"`; the local folder is `MODELS_DIR / <name>`.
  - `class RerankerError(Exception)`.
  - `class OnnxReranker` with `OnnxReranker(model_dir: Path)` and `score(query: str, passages: list[str]) -> list[float]`. It is synchronous, and a single batch scores all passages.
  - `load_reranker(settings) -> tuple[OnnxReranker | None, str]`. The note is one of:
    - `"on: ms-marco-MiniLM-L-6-v2"`
    - `"off: RERANK_ENABLED=0"`
    - `"off: unknown RERANK_MODEL <name>"`
    - `"off: model files missing, run python backend/manage_knowledge.py fetch-reranker"`
    - `"off: failed to load (<ExceptionType>)"`

    It never raises.
  - `async fetch_reranker(name: str, dest: Path = MODELS_DIR, transport=None) -> Path`
  - Settings: `rerank_model: str = "ms-marco-MiniLM-L-6-v2"`, `rerank_enabled: bool = True` (env `RERANK_ENABLED` != `"0"`), `rag_min_rerank: float = 0.0`. Task 5 sets the calibrated default.

- [ ] **Step 1: Install the pinned packages.** Add both pins to `backend/requirements.txt`, then run `.venv/bin/pip install onnxruntime==1.32.0 tokenizers==0.23.3`.

- [ ] **Step 2: Write the failing tests** in `tests/test_reranker.py`:
  - **`test_fetch_writes_verified_files`**
    - Use a `httpx.MockTransport` that serves bytes for each repo path.
    - Patch `RERANKERS` to a fake entry whose sha256 values match those bytes.
    - Assert that both files exist under `dest/<name>/`, that no `.part` files remain, and that the requested URLs are `https://huggingface.co/<repo>/resolve/<revision>/<path>`.
  - **`test_fetch_sha_mismatch_raises_and_leaves_nothing`**
    - Use the wrong bytes.
    - Assert that `RerankerError` names the file, and that `dest/<name>/` has neither that file nor a `.part` file.
    - A previously verified file is left untouched.
  - **`test_load_reranker_notes`**
    - Disabled: `(None, "off: RERANK_ENABLED=0")`.
    - Unknown name: an `off: unknown RERANK_MODEL` note.
    - Missing files: the `fetch-reranker` hint.
    - A corrupt `model.onnx` (garbage bytes): `off: failed to load (...)`.
    - None of these cases raises.
  - **`test_cli_fetch_reranker_unknown_model_exits_2`**: `manage_knowledge.run(["fetch-reranker"], settings=Settings(..., rerank_model="nope"))` returns 2.
  - **Real-model tests.** Mark them `pytest.mark.skipif(not (MODELS_DIR / "ms-marco-MiniLM-L-6-v2" / "model.onnx").exists(), reason=...)`:
    - `test_real_reranker_orders_relevant_first`:
      - Query: "What should I do if I miss a dose of colchicine?"
      - The passage "If you miss a dose of colchicine, take it as soon as you remember." scores above "Drink plenty of water to prevent kidney stones."
    - `test_real_reranker_truncates_long_query`: a 5,000-character query with 3 passages returns 3 finite floats.
    - `test_real_reranker_empty_list`: `score(q, []) == []`.

- [ ] **Step 3: Run the tests and see them fail.** Run `.venv/bin/pytest tests/test_reranker.py -q`.

- [ ] **Step 4: Implement.**
  - **Tokenizer:**
    - `Tokenizer.from_file(model_dir / "tokenizer.json")`.
    - `enable_truncation(max_length=512)` and `enable_padding()`.
    - Use `encode_batch` on `(query, passage)` pairs.
  - **ONNX session:**
    - `onnxruntime.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])`.
    - Feed int64 `input_ids`, `attention_mask` and `token_type_ids`, but only the inputs the session declares.
    - The score is `logits[:, 0]`.
  - **Downloads:**
    - Use `httpx.AsyncClient(transport=transport, follow_redirects=True, timeout=60, headers={"User-Agent": "AfterCare-ClassProject/1.0"})`.
    - Stream each file to `<local>.part` while hashing.
    - On a match, `replace` the `.part` file to its final name; on a mismatch, delete it and raise.
    - Skip a file that already exists with the right hash.
  - **The `fetch-reranker` command:**
    - Add it to the choices in `manage_knowledge.run`. It needs no API key.
    - It prints the folder and returns 0.
    - Unknown model: returns 2. `RerankerError` or `httpx.HTTPError`: prints `Error: ...` to stderr and returns 1.

- [ ] **Step 5: Run the tests again.** Run `.venv/bin/pytest tests/test_reranker.py -q`. Expected: pass, with the real-model tests skipped.

- [ ] **Step 6: Run the real command and the real-model tests.**
  - Run `.venv/bin/python backend/manage_knowledge.py fetch-reranker`, then `.venv/bin/pytest tests/test_reranker.py -q`.
  - Expected: the files land in `models/ms-marco-MiniLM-L-6-v2/` and the real-model tests pass.
  - `git status` shows no `models/` files.

- [ ] **Step 7: Commit.** `git commit -m "feat: local ONNX reranker with a pinned, verified download"`

---

### Task 3: Hybrid and rerank modes in `KnowledgeStore.retrieve`

**Files:**
- Modify: `backend/knowledge.py` (`KnowledgeStore.__init__`, `add`, `prune_labels`, `status`, `retrieve`)
- Test: `tests/test_knowledge.py`

**Interfaces:**
- Consumes: `search.tokenize`, `search.BM25`, `search.rrf` (Task 1); an object with `score(query, passages) -> list[float]` (Task 2's `OnnxReranker` or a test fake).
- Produces:
  - `KnowledgeStore(path, llm, min_similarity=0.25, reranker=None, min_rerank=0.0, reranker_note="off: not configured")`
  - `retrieve(query, limit=5, rxcuis=(), mode=None) -> list[dict]`
    - `mode` is one of `"embedding"`, `"hybrid"`, `"rerank"`, or `None`.
    - `None` means `"rerank"` if a reranker is set, else `"hybrid"`.
    - `"rerank"` without a reranker raises `ValueError`.
    - The result keys are unchanged: `source_id, title, url, topic, content, score`.
  - Module constants: `CANDIDATES = 20` (per method), `RERANK_CANDIDATES = 20` (maximum pairs sent to the reranker).

**Behaviour:**
1. Fetch the eligible rows with the existing SQL filter, adding `c.id`. If there are none, return `[]` without calling the reranker.
2. `embedding` mode returns exactly what today's code returns: cosine order, cosine floor, top `limit`.
3. `hybrid` mode:
   - Take the embedding top `CANDIDATES` and the BM25 top `CANDIDATES` over the eligible ids.
   - Order them with `rrf`.
   - Keep the rows with cosine ≥ `min_similarity`; `score` is the cosine.
   - Return the top `limit`.
4. `rerank` mode:
   - Send the first `RERANK_CANDIDATES` of the RRF order to the reranker, in a worker thread (`asyncio.to_thread`).
   - Each passage text is `title + "\n" + content`.
   - Keep the candidates with a reranker score ≥ `min_rerank`, ordered by that score. `score` is the reranker score rounded to 4 decimals.
   - Return the top `limit`.
   - The cosine floor is not applied, so BM25-only candidates can win.
5. Reranker failure:
   - Any exception, a result of the wrong length, or a non-finite value counts as a failure.
   - Use the `hybrid` result instead.
   - Log `logger.warning` once per store instance (an instance flag), with the exception type.
6. BM25 text for a chunk is `title + "\n" + content`.
7. BM25 cache:
   - One `BM25` over all chunks of the current model, built lazily.
   - Keyed by a fingerprint: the index file's `(st_ino, st_mtime_ns)` plus `SELECT count(*), max(id) FROM chunks`.
   - `add`, `prune_labels` and a fingerprint change all invalidate it.
8. `status()` adds `"reranker": "on"` or `"off"`, plus `"reranker_note"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_knowledge.py`. Use a `FakeReranker` defined in the test file that scores by a dict or a function and records its calls; some variants raise or return the wrong length.
  - `test_hybrid_finds_exact_term_that_embeddings_miss`:
    - Build a store where FakeLLM ranks a passage containing "allopurinol" below other chunks for the query "allopurinol".
    - `retrieve(q, limit=1, mode="hybrid")` returns it.
    - Then build a case where the BM25-only chunk's cosine is under the floor: it is absent in `hybrid` but returned in `rerank` when FakeReranker scores it highest.
  - `test_embedding_mode_matches_previous_behaviour`: `mode="embedding"` gives the same ids and scores as the current test expectations.
  - `test_rerank_orders_by_reranker_and_applies_min_rerank`:
    - FakeReranker scores B > A > C, with C under `min_rerank`.
    - The result order is [B, A] and the scores are the reranker scores.
    - Each passage sent to the reranker starts with the title and a newline.
  - `test_rerank_sends_at_most_twenty_candidates`: with 30 matching chunks, the reranker receives 20 passages.
  - `test_reranker_failure_falls_back_to_hybrid_and_logs_once` (caplog):
    - Run it once each with a reranker that raises, one that returns the wrong length, and one that returns NaN.
    - Each result equals the `hybrid` result.
    - The warning is logged once over two searches on the same store.
  - `test_label_outside_patient_set_never_returned_in_any_mode`:
    - A label chunk for rxcui "2683" matches the query best on BM25, embeddings and FakeReranker.
    - With `rxcuis=()` it is absent in all three modes.
    - With `rxcuis={"2683"}` it is present.
  - `test_no_candidates_returns_empty_without_calling_reranker`: on an empty eligible set, `[]` comes back and the reranker has 0 calls.
  - `test_query_without_usable_words_still_returns_embedding_hits`:
    - Query `"the of ?"` in `hybrid` mode.
    - No exception, and the result equals the `embedding` result filtered by the floor.
  - `test_bm25_cache_notices_rebuild_by_another_store`:
    - Store A answers a query.
    - Store B (same path) runs `index_sources` with new text.
    - Store A's next `hybrid` search finds a term that only appears in the new text.
  - `test_default_mode_and_status_reflect_reranker`:
    - With no reranker, `status()["reranker"] == "off"` and the default mode equals `hybrid`.
    - With FakeReranker, `status()["reranker"] == "on"` and the reranker is called.
  - `test_rerank_mode_without_reranker_raises_value_error`

- [ ] **Step 2: Run the new tests and see them fail.** Run `.venv/bin/pytest tests/test_knowledge.py -q`.

- [ ] **Step 3: Implement** the behaviour above in `backend/knowledge.py`. Add `logger = logging.getLogger(__name__)`.

- [ ] **Step 4: Run the full suite.** Run `.venv/bin/pytest -q`. Expected: everything passes, including the unchanged agent, guidance and route tests.

- [ ] **Step 5: Commit.** `git commit -m "feat: hybrid BM25 and embedding library search with optional reranking"`

---

### Task 4: Wire the reranker into the app and the CLI

**Files:**
- Modify: `backend/app.py`, `backend/manage_knowledge.py`, `backend/config.py` (env parsing, if Task 2 left it incomplete)
- Test: `tests/test_app_reranker.py` (new)

**Interfaces:**
- Consumes: `load_reranker(settings)` (Task 2); `KnowledgeStore(..., reranker=, min_rerank=, reranker_note=)` (Task 3).
- Produces:
  - `knowledge.open_store(settings, llm) -> KnowledgeStore`. It calls `load_reranker` and passes `settings.rag_min_rerank`. `create_app` uses it when it builds the store itself, and so does `manage_knowledge` for `status` and `index`.
  - `app.state.reranker` is `store.reranker`, or `None` when there is no store. A store passed in by a test keeps its own reranker.
  - At startup in `lifespan`:
    - `logger.info("Library reranker: %s", note)` when it is on.
    - `logger.warning("Library reranker: %s", note)` when it is off.

- [ ] **Step 1: Write the failing tests:**
  - **`test_settings_parse_rerank_env`**
    - Use monkeypatched env with `get_settings.cache_clear()`. Restore it after the test so the real `.env` does not leak into other tests.
    - `RERANK_ENABLED=0` gives `rerank_enabled is False`.
    - `RAG_MIN_RERANK=-2.5` gives `-2.5`.
    - `RERANK_MODEL=x` gives `"x"`.
  - **`test_app_starts_without_model_files_and_reports_off`**
    - Build `create_app(settings, llm=FakeLLM())` with:
      - `rag_db_path` under `tmp_path`;
      - `rerank_model` set to a name that is in `RERANKERS` but whose folder does not exist (monkeypatch `reranker.MODELS_DIR` to `tmp_path / "models"`).
    - The lifespan runs and `app.state.reranker is None`.
    - The store's `status()["reranker"] == "off"`, and the startup warning names `fetch-reranker` (caplog).
  - **`test_cli_status_includes_reranker_field`**: `manage_knowledge.run(["status"], settings=...)` prints JSON with `reranker` and `reranker_note`.

- [ ] **Step 2: Run them and see them fail.** Then implement, then run `.venv/bin/pytest -q`. Expected: all pass.

- [ ] **Step 3: Check it end to end.**
  - Start a scratch server on port 8002 with the real `.env` and model files: `.venv/bin/uvicorn --app-dir backend app:app --port 8002`.
  - Check that the startup log shows `Library reranker: on: ms-marco-MiniLM-L-6-v2`.
  - As `demo2`, ask "What should I do if I miss a colchicine dose?". The answer comes back with library sources.
  - Stop the server. Do not touch ports 8000 or 8001.

- [ ] **Step 4: Commit.** `git commit -m "feat: the app and the knowledge CLI load the library reranker"`

---

### Task 5: Evaluation set, calibration, ship decision, README

**Files:**
- Create:
  - `eval/library_questions.json`
  - `scripts/eval_library.py`
- Modify:
  - `backend/config.py` (calibrated `rag_min_rerank` default, and the `rerank_enabled` default per the ship rule)
  - `README.md`
  - `backend/reranker.py`, only if `bge-reranker-base` wins (see Step 4)
- Test: `tests/test_eval_library.py` (file format and metrics only; offline)

**Interfaces:**
- Consumes: `retrieve(..., mode=)` (Task 3); `open_store` (Task 4).
- Produces:
  - `eval/library_questions.json` is a list of entries `{"id": str, "question": str, "expected": [source_id, ...], "split": "tune" | "test"}`. `expected: []` means the library cannot answer the question.
  - `scripts/eval_library.py`:
    - `metrics(results: list[list[str]], expected: list[list[str]]) -> dict`
      - Keys: `hit@1`, `hit@5`, `mrr`, `no_answer_false_hits`, `answerable`, `no_answer`.
      - A hit means any expected id appears in the top k.
      - A false hit is a no-answer question that returns any result.
    - `main(argv)`:
      - `--split tune|test|all` (default `all`).
      - `--sweep` prints `hit@5` and `no_answer_false_hits` on the tune split for `RAG_MIN_RERANK` values from -10 to 10 in steps of 0.5.
      - `--model NAME` picks the reranker.
      - It prints one table row per mode (`embedding`, `hybrid`, `rerank`) with the metrics, plus the median and worst reranker time in milliseconds (timed around `reranker.score` only).
      - It passes every RxCUI present in the index as `rxcuis`, so approved labels are eligible.

- [ ] **Step 1: Write and test `metrics` and the file format.**
  - In `tests/test_eval_library.py`:
    - `metrics([["a","b"],["c"],[]], [["b"],["x"],[]])` gives `hit@1 == 0`, `hit@5 == 0.5`, `mrr == 0.25` and `no_answer_false_hits == 0`.
    - Another case covers a returned result on a no-answer question.
    - `test_question_file_is_valid` checks that:
      - every id is unique;
      - every `expected` source id exists in `knowledge/manifest.json` or is a `dailymed-rxcui-*` id;
      - both splits have no-answer questions;
      - the file has 30 to 40 questions.
  - Then implement until the tests pass.

- [ ] **Step 2: Write the 30 to 40 questions.**
  - Coverage:
    - gout (`medlineplus-gout`, `niams-gout`);
    - kidney stones;
    - the other manifest scenarios;
    - drug names and doses;
    - test names;
    - paraphrases with no shared words;
    - at least 6 no-answer questions, for example a topic the library does not cover.
  - Phrase some like real patients, borrowing wording from the 15 gout questions in `git show dfd9d0f:retriever.py` (`evaluate_gout`).
  - Before labelling a question, read the source snapshot in `knowledge/sources/` to confirm the expected source really answers it.
  - Alternate the splits so each topic appears in both.

- [ ] **Step 3: Calibrate and measure.**
  - Run `.venv/bin/python scripts/eval_library.py --sweep`.
  - Set `RAG_MIN_RERANK`: choose the highest value whose tune-split `hit@5` equals the `hit@5` with no cut, and write it to the `rag_min_rerank` default in `config.py`.
  - Then run `--split test` and `--split all`. Never print the API key.

- [ ] **Step 4: Compare models (optional, by result).**
  - If `rerank` loses to `hybrid` on the test split:
    - Add `bge-reranker-base` to `RERANKERS` (`BAAI/bge-reranker-base`, revision `2cfc18c9415c912f9d8155881c133215df768a70`, files `onnx/model.onnx` and `tokenizer.json`), computing the sha256 values from a download.
    - Fetch it, and rerun Step 3 with `--model bge-reranker-base`.
    - Keep whichever model meets the ship rule within the 300 ms budget.
  - Otherwise skip this step and say so in the report.

- [ ] **Step 5: Apply the ship rule.**
  - Rerank stays on by default only if, on the **test** split, `rerank` is not worse than `embedding` on `hit@5` and `mrr`, and has no more `no_answer_false_hits`.
  - Otherwise set `rerank_enabled` default to `False`.

- [ ] **Step 6: Update the README.** Add a "Library search" section covering:
  - how a search works (two candidate lists, fusion, rerank, fallback);
  - `fetch-reranker`;
  - the three settings;
  - the results table for the three modes on the test split and on all questions;
  - the reranker latency;
  - the chosen `RAG_MIN_RERANK`;
  - the ship decision;
  - an honest note that the questions were written by the builder, that the tune/test split limits but does not remove overfitting, and that the doctor guidance memory is out of scope.

- [ ] **Step 7: Run the full suite and commit.**
  - Run `.venv/bin/pytest -q`. Expected: all pass.
  - Commit with `git commit -m "feat: library search evaluation set, calibrated rerank cut, results"`.
