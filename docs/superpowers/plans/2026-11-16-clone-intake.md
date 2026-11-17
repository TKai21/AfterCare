# Clone multi-round intake - implementation plan

> For agentic workers: use superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** the clone takes a short history over up to 3 rounds (doctor's questions first, then clone-written follow-ups) and then answers the patient's original question, closes, or hands off.
**Spec:** docs/superpowers/specs/2026-11-15-clone-habits-design.md sections 4.2-4.4 and 4.7 (binding).
**Tech:** FastAPI + async SQLAlchemy (backend/), plain ES modules (frontend/), pytest with FakeLLM.

## Global constraints
- No em dash anywhere; use "-". Commit messages without an AI co-author line. Do not push.
- TDD: a failing test first for every behaviour change. Non-ASCII test strings as \uXXXX.
- New columns via db.ADDED_COLUMNS and sql/schema.sql.
- Fail closed: any unclear case goes to the doctor. The emergency backstop runs first on every message.
- Never write to aftercare-new.sqlite3 (live test server on port 8002).

## Review focus
1. A warning sign said in round 2 or 3 (not round 1) must still block answer/close.
2. A clone-written question that smuggles advice ("Have you tried taking 2 tablets?") or a drug the patient does not take.
3. The intake state surviving the 24 h timeout and a late reply (no duplicate cards).
4. Round 3 reply: clarify must be refused even if the model insists.
5. A new unrelated question mid-intake ends it and hands the symptom off.

### Task 1: structural close on reply turns (finish the interrupted work)
Uncommitted partial work in backend/agent.py and tests/test_agent.py (dedupe_terms, REPLY_JUDGE_PROMPT, _reply_judge). Rulings C1-C4 in .superpowers/sdd/2026-11-15-clone-habits/close-round-brief.md. Deliverable: reply-turn not_covered or validation fallback with a clean judge reply becomes a code-built close that passes validate and close_blocker; tests per C4; E2E check.

### Task 2: multi-round intake (spec 4.7)
- `intake` JSON column on care_team_questions (ADDED_COLUMNS + schema.sql): ordered list of {asked, reply}; asked/patient_reply keep the last round.
- PendingClarify carries all rounds and the round count; pending context lists every round; CLARIFY_MAX_ROUNDS (config, default 3): clarify allowed on a reply turn only while rounds < max.
- Clarify validation: per question, habit quote verbatim (cited) or clone-written (rules of 4.7, allowed drug names of 4.5, no repeat of an earlier question in the intake); at most 4 questions; at least one habit quote per round when the picker/agent context has a fitting ask habit is a prompt rule, not a validator rule.
- Prompt: habits first, own follow-ups allowed under the rules, answer the original question in the light of the replies when the evidence allows.
- Answer and close on any intake turn go through the 4.4 gate with warning terms checked across the original question and all replies; the judge (Task 1) reads all rounds.
- Route/inbox: settle_waiting appends rounds; handoff body and the doctor inbox show every round; the Ask page marks doctor questions vs clone follow-ups (payload per question: source "doctor" with habit id, or "clone").
- Tests for every Review Focus line; E2E demo check with real calls (budget in the brief).
