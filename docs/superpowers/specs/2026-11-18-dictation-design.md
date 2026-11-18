# Dictation (speech to text) - design

Date: 2026-11-18. Approved by the team in conversation (English only).

## Intent

Patients and doctors can speak instead of type: press a microphone, talk, and
the words appear in the text box to check and send. For doctors this matters
twice: a spoken reply is faster, and as text it is read into the clone's
habits (a voice clip is not).

## Goals, non-goals, anti-goals

- Goals: a microphone button on the patient Ask composer, the patient's
  direct question box (Ask page and Care team page), and the doctor's reply
  box; English speech becomes English text in that box.
- Non-goals: other languages, translation, live (streaming) captions, keeping
  the audio. Voice clips (recorded messages) stay as they are.
- Anti-goals: text sent without the speaker seeing it; audio or transcripts
  stored or logged on the server.

## Approaches

- **Chosen: server-side transcription with the OpenAI audio API.** Works in
  every browser with MediaRecorder, handles medical words, about $0.003 per
  minute, and the key stays on the server.
- Browser Web Speech API: free, but Chrome-only, unreliable, and sends audio
  to a third party we do not configure.
- Languages: English only (chosen). Transcribing other languages as spoken
  would bypass the emergency backstop, which matches English terms.

## Design

- `POST /api/voice/transcribe` (any logged-in user, multipart `file`): the
  same upload checks as voice clips (`decode_upload`: at most 60 seconds, 5
  MB, the accepted audio types). The server sends the audio to the
  transcription model with `language="en"` and returns `{"text": "..."}`
  (whitespace collapsed, at most the box's character limit). Nothing is
  stored; the audio and the text are not logged. Errors: 503 when no model is
  configured, 502 when the model fails, 422 for a bad upload, 422 with "I
  could not hear anything. Please try again." for an empty transcript.
- Setting `TRANSCRIBE_MODEL` (default `gpt-4o-mini-transcribe`), called
  through a new `LLMClient.transcribe(data: bytes, filename: str,
  content_type: str) -> str` so tests fake it like `chat`.
- Front end: one shared `attachDictation(button, textarea, {onBusy})` in
  `frontend/voice.js`, reusing the existing recording code. States: idle
  (microphone icon, label "Dictate"), recording (stop icon, elapsed time,
  stops itself at 60 seconds), transcribing (spinner, box and Send disabled),
  error (inline message under the box, same style as existing notices). The
  transcript is **appended** to whatever is in the box (with a space), the
  box is focused, and nothing is sent: the person presses Send.
- Placement: inside the composer next to Send; during an interview the order
  is microphone, "Not sure", Send, and it must fit at 390 px without wrapping
  the placeholder away. Keyboard and screen-reader labels as for the existing
  buttons. The button is hidden when the browser has no MediaRecorder.

## Left to the builder

Icon drawing (match the existing inline icons), exact CSS, and the wording of
the microphone-permission error.

## Calls made by Claude

Append rather than replace; no audio kept; 60-second limit reused from voice
clips.
