# AfterCare - Hybrid Library Search (BM25 + Embeddings) With a Local Reranker

Date: 2026-11-14
Status: draft, awaiting team approval
Builds on: the education library (`backend/knowledge.py`), drug labels (`2026-11-13-drug-label-library-design.md`).

## 1. Intent

The team wants the AI clones' answers to be more accurate. An answer can only be as good as the passages the agent is given, so this change improves how education-library passages are found and ordered.

Today `KnowledgeStore.retrieve` (`knowledge.py`) ranks the library (49 documents, 822 chunks) by embedding cosine similarity alone (text-embedding-3-small). Embeddings match meaning but can miss exact terms: a drug name, a test name, a number with a unit. A teammate (renny475, branch `origin/final_project_rag`) built a separate BM25 retriever for discharge PDFs. The team wants AfterCare itself to use BM25, and to add a reranking step on top.

## 2. Decisions made with the team

- Rerank is added inside AfterCare, not to the teammate's standalone module. Goal: more accurate answers.
- **Approach A: a small cross-encoder reranker running on our own server** (CPU, no API cost, no new outside service).
- Rejected:
  - B, gpt-4o-mini as the reranker: one more model call per search, slower and costlier, scores vary between runs.
  - C, a hosted rerank API (Cohere, Jina): sends patient questions to another company and adds another key.
- **Scope: the education library only.** The doctor guidance memory (`guidance.py`) is unchanged: no BM25, no rerank, same 0.45 cosine floor. Reasons:
  - The dev database holds 4 guidance entries across three doctors, so there is nothing for a reranker to order.
  - Guidance is spoken as the doctor's own words. Two similar-looking questions can need opposite advice, and a clone must say it has no relevant advice when nothing is close. Loosening the floor works against both.
- **A clone evaluation (collect 20-30 cases per doctor; compare the clone with and without doctor memory against the doctor's own answers) is a separate, later project.**
- A retrieval evaluation set is built alongside this change, so "more accurate" is measured, not assumed (designer's call, offered to the team).

## 3. Goals

1. Library search finds candidates two ways (BM25 and embeddings), merges them, reranks them with a local cross-encoder, and returns the top results.
2. The reranker runs on the server CPU. No patient text leaves the server for this step.
3. Retrieval accuracy on the evaluation set is measured before and after; the ship rule is in section 10.
4. If the reranker is missing or fails, search still works, using the merged order. A patient's question never fails because of the reranker.

## 4. Non-goals and anti-goals

Non-goals:
- Any change to the doctor guidance memory.
- Changing how the discharge document reaches the agent (it is still given whole, not searched).
- Integrating the teammate's PDF module or its PDF parsing.
- Hosted rerank APIs or LLM-based reranking.
- Non-English questions (the default reranker is English-only).
- UI changes. Patients and doctors see the same screens.
- Changing the agent's rules, tools or tool outputs beyond the order and selection of passages.

Anti-goals (failure even if it works):
- **A passage the patient may not see is returned,** because BM25 or reranking bypassed a filter. The drug-label RxCUI filter runs **before** candidates are gathered, exactly as today.
- **Answers get slower in a way a patient notices.** Budget: reranking adds at most about 300 ms per library search on the dev laptop (Apple Silicon CPU).
- **The server cannot start, or tests need the network,** because of the model download.
- Accuracy drops on the evaluation set and nobody notices.

## 5. How a library search works after the change

1. **Filter** eligible rows exactly as today: manifest sources plus labels whose RxCUI is in the patient's set.
2. **Embedding candidates:** the top N by cosine similarity, as today.
3. **BM25 candidates:** the top N by BM25 over the same eligible rows.
4. **Merge** the two lists with reciprocal rank fusion (RRF) into one list without duplicates.
5. **Rerank** the candidates with the cross-encoder (query, passage) and keep the top `limit` (5, as today) whose reranker score is at or above `RAG_MIN_RERANK`.
6. If the reranker is unavailable or raises, skip step 5: use the RRF order and today's cosine floor (`RAG_MIN_SIMILARITY`, 0.25), and log a warning once.

N is about 20 per method; the exact number is tuned on the evaluation set. A BM25-only candidate (under the cosine floor) can be returned when the reranker scores it high enough; this is the point of adding BM25.

## 6. Reranker model and runtime

- Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (about 90 MB), run through ONNX so the server does not need PyTorch. Alternative to compare on the evaluation set: `BAAI/bge-reranker-base`. The builder picks the final model by the evaluation results and the latency budget.
- The model is downloaded once by an explicit command (for example `python backend/manage_knowledge.py fetch-reranker`) over HTTPS into `models/` (git-ignored), checked against a pinned SHA-256. The server never downloads at startup. If the files are missing it starts normally, uses the fallback, and reports the reranker as off.
- Settings: `RERANK_MODEL`, `RERANK_ENABLED`, `RAG_MIN_RERANK`.
- One reranker object is created in `create_app` and stored on `app.state.reranker`. `KnowledgeStore` takes `reranker=None`; `None` means the fallback, so existing callers and tests keep working. Tests pass a deterministic fake and never load a model.
- Inference runs in a worker thread (`asyncio.to_thread`) so it never blocks the event loop.
- `status()` gains a `reranker` field (`"on"`, or `"off"` with a reason), and the server logs one line at startup. This is not a user-visible change.
- Dependencies: `onnxruntime` and `tokenizers` are acceptable. Target: the existing Python 3.13 venv on macOS arm64; the builder confirms Linux x86-64 wheels exist too, for a later AWS deploy.

## 7. BM25

- Hand-written (no `rank_bm25`), in a small module of its own.
- Tokenizer: lowercase, split into words, keep numbers with their units (for example "0.6 mg" stays matchable), and a short English stopword list.
- IDF and average length come from the whole index, cached, and rebuilt when the index changes (add, rebuild, prune), so rankings do not depend on which labels a patient has. Scores are computed for all rows; the eligibility filter then keeps only allowed rows before the top N is taken.

## 8. Evaluation set

- A file under `eval/` with about 30 to 40 English questions. Each lists the library source id that should rank in the top results; a few list none (questions the library cannot answer, to test the cut).
- Coverage: gout (demo2), kidney stones (demo), the drug-label ingredients from the seed, exact-term questions (drug names, doses, test names), paraphrases, and questions with no answer.
- The teammate's 15 gout questions are about a discharge PDF, which AfterCare does not search, so they are used only as a source of real patient wording.
- The builder writes the questions and labels; the team reviews them.
- A script prints, for baseline (embedding only), hybrid without rerank, and hybrid with rerank: hit@1, hit@5, mean reciprocal rank, false hits on no-answer questions, and median and worst latency.
- The script runs on demand against the real index and embedding API (`.env`, `knowledge/index.sqlite3`). It is not part of the default pytest run. The README records the numbers.

## 9. Testing (pytest, offline)

In the existing style, with FakeLLM and a fake reranker:
- BM25 ranks an exact-term passage first when embeddings rank it low.
- RRF merges and removes duplicates.
- The reranker order replaces the merged order; the top `limit` is returned; results under `RAG_MIN_RERANK` are dropped.
- A label for an RxCUI outside the patient's set is never returned, even when it is the best BM25 or reranker match.
- A reranker that raises, or no reranker, falls back to the merged order with today's cosine floor; the agent turn succeeds.
- The server starts without model files.
- Guidance retrieval behaviour is unchanged (existing tests still pass).

## 10. Ship rule

Rerank is on by default only if, on the evaluation set, hybrid plus rerank is not worse than the embedding-only baseline on hit@5 and mean reciprocal rank, and returns no more false hits on no-answer questions. Otherwise the code ships with `RERANK_ENABLED` off, and the README says why.

## 11. Left to the builder

- Module layout (suggested: `backend/search.py` for BM25, RRF and the reranker wrapper).
- The ONNX runtime library and the fetch details.
- N, the RRF constant, tokenizer details and the calibrated `RAG_MIN_RERANK`, all from the evaluation set.
- The exact evaluation file format and script output.
