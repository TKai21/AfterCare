# Voice questions and replies

This milestone adds original voice clips to the existing Care team workflow.
There is no dictation, transcription, speech synthesis, or reuse of recorded
doctor answers by the AI. The original text features remain available.

## Setup

Install FFmpeg on the backend host. On Ubuntu/WSL:

```bash
sudo apt update
sudo apt install -y ffmpeg
ffmpeg -version
```

No new Python package is needed beyond `backend/requirements.txt`.
Use your existing `.env` and normal application startup command:

```bash
uvicorn --app-dir backend app:app --host 0.0.0.0 --port 8000
```

The existing startup creates the new `voice_clips` table automatically. It does
not drop or alter an existing table. Do NOT delete your database for this change.
`sql/schema.sql` also describes the new table for manual MySQL setups.

Files default to `private/voice/` relative to the project root, and that directory
is gitignored and not mounted as public frontend files. An optional
`VOICE_STORAGE_PATH` can point to a private persistent directory instead. The app
refuses a storage path under `frontend/`. Keep custom paths out of Git as well.
Database metadata and recordings must be backed up together if persistence is needed.

## Patient workflow

1. Log in and open Care team. No confirmed document or language model is needed
   to send a direct voice question.
2. Choose the intended doctor.
3. Under the existing form, select Record voice clip and grant microphone access.
4. Stop recording, listen to the preview, and either Discard or Send voice clip.
5. The question appears in the selected doctor's existing queue.
6. After the doctor replies, revisit or refresh Care team to see the answer.
7. Select Load doctor reply, then use the normal audio controls to play/pause/seek.

The current queue refresh behavior is reused; this feature does not add live push
notifications or automatic polling.

## Doctor workflow

1. Open `/doctor` in a separate browser profile/private window so the patient and
   doctor logins do not overwrite each other's browser token.
2. Load and play the patient's question.
3. Record a voice reply, preview it, and send it.
4. The question moves to Answered. The patient can then load the reply.

Voice questions accept voice replies. A doctor can also choose a voice reply for
an existing text question, but it will not create AI guidance. Existing typed
replies to text questions keep their previous behavior.

Each question supports one question clip and one answer clip. The existing
answered-state/atomic-update logic plus a unique database constraint prevents
duplicate answers. Resolving a question keeps its clips available to its participants.

## Recording limits and access

- Recording is explicit and automatically stops after 60 seconds.
- The server accepts at most 5 MiB and checks duration by decoding actual audio,
  not by trusting a client-supplied duration. A 250 ms codec/frame tolerance is used.
- WebM/Opus, Ogg, MP4 and WAV inputs are decoded locally using FFmpeg, normalized
  to mono 16 kHz WAV, and stored without the original container metadata.
- FFmpeg accepts input/output pipes only, with no file/network URL protocols.
- Playback requires the current patient's or assigned doctor's login token.
  Another patient/doctor receives 404; unauthenticated requests receive 401.
- The browser fetches audio with an Authorization header, then uses a temporary
  blob URL for its audio player. Tokens never appear in playback URLs.
- Responses are marked `Cache-Control: no-store, private`.
- Cancelling a draft discards it locally. Stopping, discarding, hiding the page or
  removing a recording UI releases the microphone. Sent clips are retained.
- There is no automatic recording-retention purge or delete-recording UI yet.
  The current application does not delete individual care-team questions; dev
  reseeding/deleting database rows can leave inaccessible orphan files. Retention
  and administrative cleanup are future work, not silently implemented here.

Microphone capture needs HTTPS or localhost and browser permission. An HTTP LAN
IP/cloud IP generally will not provide microphone access. Keep a supported text
workflow available on unsupported browsers. Audio clips are not analyzed for
emergencies; the recording UI explicitly identifies the inbox as non-urgent.

## Database compatibility

The existing non-null question text and answer-state logic are preserved.
Voice questions use the display label `Voice question` and source `patient_voice`;
voice answers use the display label `Voice reply`. These labels are NOT transcripts.
The real message content is the private audio attachment. The new table stores
question ID, sender, question/answer kind, storage key, duration and byte size.

Recorded replies do not create DoctorGuidance rows or request chat/embedding
services. The text-answer endpoint rejects voice questions so their placeholder
labels cannot be accidentally saved as reusable AI advice.

## Tests

From the repository root, with the project's Python environment active:

```bash
python -m pytest tests/test_voice_storage.py tests/test_routes_voice.py tests/test_voice_frontend.py -q
python -m pytest -q
```

The storage suite can also run without pytest or backend web/database packages:

```bash
PYTHONPATH=backend python -m unittest discover -s tests -p test_voice_storage.py -v
node tests/voice_frontend_checks.mjs
```

API integration checks cover both roles, unauthorized playback, queue/exported
metadata, no-model operation, no AI-guidance creation, duplicate/concurrent
answers, invalid/oversized/overlong audio and resolving a question.

### Verification performed while building this patch

- 10 real local audio/storage tests passed, including WAV, WebM, Ogg and MP4.
- 9 frontend lifecycle checks passed using browser-device/DOM test doubles.
- Python files parsed and frontend JavaScript syntax was checked.
- The full FastAPI/SQLAlchemy API suite could NOT run in the build environment:
  required backend packages were absent and the package server was unreachable.
- Real microphone/browser interaction and MySQL have NOT been run here.

Run the API tests on WSL before pushing, then perform the patient/doctor browser
walkthrough using fictional recordings and separate browser sessions. Test denied
microphone permission, Discard, navigation during recording, and both participants'
playback. Check that My guidance gains no entry after a voice reply.
