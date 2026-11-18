# Dictation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A microphone button that turns English speech into text in the patient and doctor text boxes.

**Architecture:** One backend endpoint sends an uploaded recording to the OpenAI transcription model and returns text, storing nothing. One shared front-end helper records with the existing voice code and appends the transcript to a textarea.

**Tech Stack:** FastAPI, OpenAI audio transcription API, MediaRecorder, plain ES modules, pytest with FakeLLM.

**Spec:** `docs/superpowers/specs/2026-11-18-dictation-design.md`

## Global Constraints

- `TRANSCRIBE_MODEL` default `gpt-4o-mini-transcribe`; `language="en"`.
- Upload limits are those of voice clips (`decode_upload`): at most 60 seconds and 5 MB.
- Nothing stored, nothing logged (no audio, no transcript text in logs).
- Empty transcript message: `I could not hear anything. Please try again.`
- The transcript is appended to the box and never sent automatically.
- No em dash. Never write to `aftercare-new.sqlite3` from tests. Commit messages without AI co-author lines.

## Review Focus

1. A transcript longer than the box limit is cut to the limit, not rejected - Task 1 test `test_transcribe_cuts_to_limit`.
2. A model failure must not leak the provider's error text to the client - Task 1 test `test_transcribe_model_failure_is_502_generic`.
3. Dictating while text is already typed keeps that text - Task 2 browser check.
4. Pressing the microphone during an interview must not hide "Not sure" or Send at 390 px - Task 2 screenshots.
5. A user who denies microphone permission sees a clear message and the box still works - Task 2 browser check.

---

### Task 1: Transcription endpoint

**Files:**
- Modify: `backend/llm.py`, `backend/config.py`, `backend/routes_voice.py`, the FakeLLM in `tests/conftest.py`
- Test: `tests/test_routes_voice.py` (or the existing voice route test file)

**Interfaces:**
- Produces: `LLMClient.transcribe(data: bytes, filename: str, content_type: str) -> str` (raises `LLMError`); `POST /api/voice/transcribe` -> `{"text": str}`; setting `transcribe_model`.

- [ ] **Step 1: Write the failing tests**: `test_transcribe_returns_text_and_stores_nothing` (no new rows in `voice_clips`, storage untouched), `test_transcribe_requires_login` (401), `test_transcribe_rejects_bad_upload` (422, same cases as clips), `test_transcribe_empty_is_422_with_message`, `test_transcribe_cuts_to_limit`, `test_transcribe_model_failure_is_502_generic`, `test_transcribe_without_model_is_503`, `test_transcribe_collapses_whitespace`.
- [ ] **Step 2: Run** them - FAIL.
- [ ] **Step 3: Implement.** `transcribe` posts multipart to `{base_url}/audio/transcriptions` with `model`, `language=en`, `response_format=json`, using the client's existing HTTP session, timeout and error mapping. The limit is the longer of the two box limits already defined in the routes (reuse the constant).
- [ ] **Step 4: Run** the voice tests and the full suite - PASS.
- [ ] **Step 5: Commit** `feat: transcription endpoint for dictation`.

### Task 2: Microphone button

**Files:**
- Modify: `frontend/voice.js`, `frontend/voice.css`, `frontend/ask.js`, `frontend/careteam.js`, `frontend/doctor.js`

**Interfaces:**
- Consumes: Task 1 endpoint.
- Produces: `attachDictation(button, textarea, {onBusy}) -> dispose` in `frontend/voice.js`.

- [ ] **Step 1: Implement** the helper and place the button in: the Ask composer (`#askForm`), the direct question box on the Ask page and the Care team page, and the doctor's reply box. States and placement per the spec.
- [ ] **Step 2: Verify in a browser** (Playwright, throwaway server on port 8003 with a DB copy, Chromium launched with `--use-fake-ui-for-media-stream --use-fake-device-for-media-stream --use-file-for-fake-audio-capture=<wav>`; make the wav with macOS `say -o x.aiff "No fever and no trouble breathing"` and `afconvert` to 16-bit WAV): dictate into the Ask composer with text already typed (text kept, transcript appended, not sent), dictate an interview reply, dictate a doctor reply; a context without microphone permission shows the error. Screenshots at 1100 and 390 px of idle, recording and transcribing states on the patient and doctor pages; no overlap or horizontal scroll.
- [ ] **Step 3: Commit** `feat: dictate into the question and reply boxes`.
