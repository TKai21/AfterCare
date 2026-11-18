# Voice: dictation, and voice clips that already exist

Since 2026-11-18 dictation is the only way to use the voice: the microphone button
turns English speech into text in the patient's question boxes (Ask composer, direct
question on Ask and Care team) and the doctor's answer box. The person reads the text
and presses Send; nothing is sent automatically. Sending recorded voice clips was
removed, because a clip is not transcribed, not checked by the emergency backstop and
not learned by the clone. Clips sent earlier still play.

## Dictation

- `POST /api/voice/transcribe` (any logged-in user, multipart `file`) decodes the
  recording with FFmpeg (at most 60 seconds and 5 MB; WebM/Opus, Ogg, MP4 and WAV),
  sends it to the transcription model with `language=en` and returns `{"text": "..."}`.
- Nothing is stored or logged: not the audio, not the text.
- Model: `TRANSCRIBE_MODEL` (default `gpt-4o-mini-transcribe`), through the same
  `LLM_API_BASE` and `LLM_API_KEY` as chat. 503 without a configured model, 502 when
  the model fails (a generic message), 422 for a bad upload or when nothing was heard.
- The transcript is cut to the longest box limit and appended to what is already typed.
- Microphone capture needs HTTPS or localhost and browser permission. The button is
  hidden when the browser has no MediaRecorder.
- Install FFmpeg on the backend host (`sudo apt install -y ffmpeg`).

## Existing clips

- Questions with a clip show "Load ... (Ns)" and a normal audio player.
  `GET /api/voice/clips/{id}` needs the patient's or the assigned doctor's token
  (404 for anyone else, 401 without a login), is `Cache-Control: no-store, private`,
  and the browser fetches it with an Authorization header, never a token in the URL.
- Files live under `private/voice/` (or `VOICE_STORAGE_PATH`), gitignored and not served
  as frontend files. The `voice_clips` table is unchanged.
- An old clip-only question ("Voice question") that is still open is answered in text
  by the doctor. Such an answer closes it but is never saved as AI guidance.

## Tests

```bash
python -m pytest tests/test_voice_storage.py tests/test_routes_voice.py tests/test_voice_frontend.py -q
node tests/voice_frontend_checks.mjs
```
