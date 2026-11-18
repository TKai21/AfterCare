"""Voice API integration checks: ownership, persistence, status and no AI reuse."""
import asyncio
import io
from pathlib import Path
import shutil
import wave

import pytest
import pytest_asyncio
from sqlalchemy import func, select

from conftest import auth_headers, doctor_headers
from models import CareTeamQuestion, DoctorGuidance, User, VoiceClip
from voice_storage import MAX_UPLOAD_BYTES, VoiceStorage


def recording(seconds=0.4):
    data = io.BytesIO()
    with wave.open(data, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(16000)
        clip.writeframes(b"\x00\x00" * round(seconds * 16000))
    return data.getvalue()


def upload(data=None, content_type="audio/wav"):
    return {"file": ("clip.wav", recording() if data is None else data, content_type)}


@pytest_asyncio.fixture
async def voice_users(client, tmp_path):
    client.app.state.voice_storage = VoiceStorage(tmp_path / "private-voice")
    patient = await auth_headers(client, "alice")
    doctor = await doctor_headers(client)
    doctor_id = (await client.get("/api/doctors", headers=patient)).json()["doctors"][0]["id"]
    return client, patient, doctor, doctor_id


@pytest.fixture(autouse=True)
def require_audio_decoder():
    if shutil.which("ffmpeg") is None:
        pytest.skip("Install FFmpeg before running voice API integration tests")


async def legacy_question(ctx):
    """A clip-only question as sent before sending clips was removed."""
    client, patient, _, doctor_id = ctx
    from voice_storage import normalize_audio
    async with client.app.state.session_factory() as session:
        user = await session.scalar(select(User).where(User.username == "alice"))
        question = CareTeamQuestion(user_id=user.id, question="Voice question",
                                    reason="Voice question from the patient", source="patient_voice",
                                    doctor_id=doctor_id, status="open")
        session.add(question)
        await session.flush()
        audio = normalize_audio(recording(), "audio/wav")
        key = client.app.state.voice_storage.save(audio)
        session.add(VoiceClip(question_id=question.id, sender_id=user.id, kind="question",
                              storage_key=key, duration_ms=audio.duration_ms, byte_size=len(audio.data)))
        await session.commit()
        return question.id


async def question_json(ctx, headers):
    client = ctx[0]
    return (await client.get("/api/care-team", headers=headers)).json()["questions"][0]


async def test_existing_clips_are_playable_by_their_patient_and_doctor(voice_users):
    client, patient, doctor, _ = voice_users
    await legacy_question(voice_users)
    question = await question_json(voice_users, patient)
    url = question["voice"]["question"]["url"]
    for headers in (patient, doctor):
        response = await client.get(url, headers=headers)
        assert response.status_code == 200
        assert response.content[:4] == b"RIFF"
        assert "no-store" in response.headers["cache-control"]


async def test_other_patients_and_doctors_cannot_play_clips(voice_users):
    client, patient, _, _ = voice_users
    await legacy_question(voice_users)
    url = (await question_json(voice_users, patient))["voice"]["question"]["url"]
    other_patient = await auth_headers(client, "bob")
    other_doctor = await doctor_headers(client, "dr.other", "Other Doctor")
    assert (await client.get(url)).status_code == 401
    for headers in (other_patient, other_doctor):
        assert (await client.get(url, headers=headers)).status_code == 404


async def test_missing_stored_recording_returns_404(voice_users):
    client, patient, _, _ = voice_users
    await legacy_question(voice_users)
    url = (await question_json(voice_users, patient))["voice"]["question"]["url"]
    for path in client.app.state.voice_storage.directory.glob("*.wav"):
        path.unlink()
    assert (await client.get(url, headers=patient)).status_code == 404


async def test_sending_clips_is_gone(voice_users):
    client, patient, doctor, doctor_id = voice_users
    qid = await legacy_question(voice_users)
    sent = await client.post("/api/voice/questions", headers=patient, data={"doctor_id": doctor_id}, files=upload())
    assert sent.status_code in (404, 405)
    reply = await client.post(f"/api/voice/questions/{qid}/reply", headers=doctor, files=upload())
    assert reply.status_code in (404, 405)


async def test_old_clip_question_can_be_answered_in_text_and_teaches_nothing(voice_users):
    client, patient, doctor, _ = voice_users
    qid = await legacy_question(voice_users)
    result = await client.post(f"/api/doctor/questions/{qid}/answer", headers=doctor, json={"answer": "Yes, with food."})
    assert result.status_code == 200, result.text
    assert result.json()["question"]["answer"] == "Yes, with food."
    assert result.json()["guidance"] is None
    async with client.app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(DoctorGuidance)) == 0
    assert set((await question_json(voice_users, patient))["voice"]) == {"question"}
    general = await client.post(f"/api/doctor/questions/{qid}/answer", headers=doctor,
                                json={"answer": "x", "general": True})
    assert general.status_code in (409, 422)


# ---- Dictation: speech to text, nothing stored ----

async def clip_count(client):
    async with client.app.state.session_factory() as session:
        return await session.scalar(select(func.count()).select_from(VoiceClip))


async def test_transcribe_returns_text_and_stores_nothing(voice_users, fake_llm):
    client, patient, _, _ = voice_users
    fake_llm.transcripts.append("No fever and no trouble breathing.")
    result = await client.post("/api/voice/transcribe", headers=patient, files=upload())
    assert result.status_code == 200, result.text
    assert result.json() == {"text": "No fever and no trouble breathing."}
    assert await clip_count(client) == 0
    assert not list(client.app.state.voice_storage.directory.glob("*.wav"))
    data, filename, content_type = fake_llm.transcribe_calls[0]
    assert data[:4] == b"RIFF" and content_type == "audio/wav"


async def test_transcribe_works_for_doctors(voice_users, fake_llm):
    client, _, doctor, _ = voice_users
    fake_llm.transcripts.append("Take it with food.")
    result = await client.post("/api/voice/transcribe", headers=doctor, files=upload())
    assert result.status_code == 200 and result.json() == {"text": "Take it with food."}


async def test_transcribe_requires_login(voice_users, fake_llm):
    client, *_ = voice_users
    result = await client.post("/api/voice/transcribe", files=upload())
    assert result.status_code == 401
    assert fake_llm.transcribe_calls == []


@pytest.mark.parametrize("data,mime,expected", [
    (b"", "audio/wav", 400),
    (b"<html>not audio</html>", "audio/webm", 400),
    (recording(), "text/plain", 415),
    (b"x" * (MAX_UPLOAD_BYTES + 1), "audio/wav", 413),
    (recording(61), "audio/wav", 413),
])
async def test_transcribe_rejects_bad_upload(voice_users, fake_llm, data, mime, expected):
    client, patient, _, _ = voice_users
    result = await client.post("/api/voice/transcribe", headers=patient, files=upload(data, mime))
    assert result.status_code == expected
    assert fake_llm.transcribe_calls == []


async def test_transcribe_missing_file_is_422(voice_users):
    client, patient, _, _ = voice_users
    assert (await client.post("/api/voice/transcribe", headers=patient)).status_code == 422


@pytest.mark.parametrize("silence", ["", "   \n\t "])
async def test_transcribe_empty_is_422_with_message(voice_users, fake_llm, silence):
    client, patient, _, _ = voice_users
    fake_llm.transcripts.append(silence)
    result = await client.post("/api/voice/transcribe", headers=patient, files=upload())
    assert result.status_code == 422
    assert result.json()["detail"] == "I could not hear anything. Please try again."


async def test_transcribe_cuts_to_limit(voice_users, fake_llm):
    from routes_care_team import MAX_QUESTION_CHARS
    from routes_doctor import MAX_TEXT_CHARS
    MAX_TRANSCRIPT_CHARS = max(MAX_QUESTION_CHARS, MAX_TEXT_CHARS)
    client, patient, _, _ = voice_users
    fake_llm.transcripts.append("word " * 3000)
    result = await client.post("/api/voice/transcribe", headers=patient, files=upload())
    assert result.status_code == 200
    text = result.json()["text"]
    assert 0 < len(text) <= MAX_TRANSCRIPT_CHARS
    assert text == text.strip()


async def test_transcribe_model_failure_is_502_generic(voice_users, fake_llm):
    from llm import LLMError
    client, patient, _, _ = voice_users
    fake_llm.transcripts.append(LLMError("401 sk-secret-key invalid for https://api.openai.com"))
    result = await client.post("/api/voice/transcribe", headers=patient, files=upload())
    assert result.status_code == 502
    assert "sk-secret" not in result.text and "openai" not in result.text.lower()


async def test_transcribe_without_model_is_503(voice_users):
    client, patient, _, _ = voice_users
    client.app.state.llm = None
    result = await client.post("/api/voice/transcribe", headers=patient, files=upload())
    assert result.status_code == 503


async def test_transcribe_collapses_whitespace(voice_users, fake_llm):
    client, patient, _, _ = voice_users
    fake_llm.transcripts.append("  No   fever\n\nand\tno  cough. ")
    result = await client.post("/api/voice/transcribe", headers=patient, files=upload())
    assert result.json() == {"text": "No fever and no cough."}


async def test_llm_client_transcribe_posts_multipart_to_the_audio_api():
    import httpx
    from config import Settings
    from llm import LLMClient, LLMError
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.content
        return httpx.Response(200, json={"text": "hello"})

    settings = Settings(llm_api_base="https://api.example/v1", llm_api_key="k")
    client = LLMClient(settings, transport=httpx.MockTransport(handler))
    assert await client.transcribe(b"RIFFdata", "dictation.wav", "audio/wav") == "hello"
    assert seen["url"] == "https://api.example/v1/audio/transcriptions"
    assert seen["auth"] == "Bearer k"
    for part in (b"gpt-4o-mini-transcribe", b'name="language"', b"\r\nen\r\n", b"json", b"RIFFdata"):
        assert part in seen["body"]
    bad = LLMClient(settings, transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    with pytest.raises(LLMError):
        await bad.transcribe(b"x", "dictation.wav", "audio/wav")
