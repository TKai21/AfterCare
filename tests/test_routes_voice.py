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


async def new_question(ctx):
    client, patient, _, doctor_id = ctx
    result = await client.post("/api/voice/questions", headers=patient,
                               data={"doctor_id": doctor_id}, files=upload())
    assert result.status_code == 201, result.text
    return result.json()["question"]


async def test_patient_question_and_doctor_reply_are_playable(voice_users):
    client, patient, doctor, _ = voice_users
    question = await new_question(voice_users)
    question_url = question["voice"]["question"]["url"]
    for headers in (patient, doctor):
        response = await client.get(question_url, headers=headers)
        assert response.status_code == 200
        assert response.content[:4] == b"RIFF"
        assert response.headers["content-type"].startswith("audio/wav")
        assert "no-store" in response.headers["cache-control"]
    result = await client.post(f"/api/voice/questions/{question['id']}/reply",
                               headers=doctor, files=upload())
    assert result.status_code == 200, result.text
    reply = result.json()["question"]
    assert reply["status"] == "answered"
    assert reply["answer"] == "Voice reply"
    assert reply["voice"]["answer"]["duration_ms"] > 0
    assert (await client.get(reply["voice"]["answer"]["url"], headers=patient)).status_code == 200
    listed = (await client.get("/api/care-team", headers=patient)).json()["questions"][0]
    assert set(listed["voice"]) == {"question", "answer"}
    queue = (await client.get("/api/doctor/questions", headers=doctor)).json()
    assert queue["open"] == []
    assert queue["answered"][0]["voice"]["answer"]["id"] == reply["voice"]["answer"]["id"]


async def test_voice_never_calls_model_or_creates_guidance(voice_users):
    client, _, doctor, _ = voice_users

    class ForbiddenModel:
        async def chat(self, *args, **kwargs):
            raise AssertionError("Voice clips must not call a chat model")

        async def embed(self, *args, **kwargs):
            raise AssertionError("Voice clips must not call embeddings")

    client.app.state.llm = ForbiddenModel()
    question = await new_question(voice_users)
    result = await client.post(f"/api/voice/questions/{question['id']}/reply", headers=doctor, files=upload())
    assert result.status_code == 200
    async with client.app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(DoctorGuidance)) == 0


async def test_other_patients_and_doctors_cannot_play_or_reply(voice_users):
    client, _, _, _ = voice_users
    question = await new_question(voice_users)
    other_patient = await auth_headers(client, "bob")
    other_doctor = await doctor_headers(client, "dr.other", "Other Doctor")
    url = question["voice"]["question"]["url"]
    assert (await client.get(url)).status_code == 401
    for headers in (other_patient, other_doctor):
        assert (await client.get(url, headers=headers)).status_code == 404
    denied = await client.post(f"/api/voice/questions/{question['id']}/reply", headers=other_doctor, files=upload())
    assert denied.status_code == 404
    assert (await client.get("/api/care-team", headers=other_patient)).json()["questions"] == []


async def test_role_checks_apply_to_uploads(voice_users):
    client, patient, doctor, doctor_id = voice_users
    assert (await client.post("/api/voice/questions", data={"doctor_id": doctor_id}, files=upload())).status_code == 401
    assert (await client.post("/api/voice/questions", headers=doctor, data={"doctor_id": doctor_id}, files=upload())).status_code == 403
    question = await new_question(voice_users)
    assert (await client.post(f"/api/voice/questions/{question['id']}/reply", headers=patient, files=upload())).status_code == 403


async def test_no_model_configuration_is_needed(voice_users):
    client, _, doctor, _ = voice_users
    client.app.state.llm = None
    question = await new_question(voice_users)
    response = await client.post(f"/api/voice/questions/{question['id']}/reply", headers=doctor, files=upload())
    assert response.status_code == 200


async def test_voice_question_cannot_be_saved_as_text_guidance(voice_users):
    client, _, doctor, _ = voice_users
    question = await new_question(voice_users)
    result = await client.post(f"/api/doctor/questions/{question['id']}/answer", headers=doctor,
                               json={"answer": "A typed answer", "general": False})
    assert result.status_code == 409
    async with client.app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(DoctorGuidance)) == 0


async def test_duplicate_reply_does_not_create_an_extra_file(voice_users):
    client, _, doctor, _ = voice_users
    question = await new_question(voice_users)
    url = f"/api/voice/questions/{question['id']}/reply"
    assert (await client.post(url, headers=doctor, files=upload())).status_code == 200
    assert (await client.post(url, headers=doctor, files=upload())).status_code == 409
    assert len(list(client.app.state.voice_storage.directory.glob("*.wav"))) == 2


async def test_resolving_keeps_voice_attachments(voice_users):
    client, patient, doctor, _ = voice_users
    question = await new_question(voice_users)
    await client.post(f"/api/voice/questions/{question['id']}/reply", headers=doctor, files=upload())
    result = await client.post(f"/api/care-team/{question['id']}/resolve", headers=patient)
    assert result.json()["status"] == "resolved"
    assert set(result.json()["voice"]) == {"question", "answer"}


async def test_voice_reply_to_existing_text_question_skips_guidance(voice_users):
    client, patient, doctor, doctor_id = voice_users
    asked = await client.post("/api/care-team", headers=patient,
                               json={"doctor_id": doctor_id, "question": "Can I ask a question?"})
    qid = asked.json()["question"]["id"]
    response = await client.post(f"/api/voice/questions/{qid}/reply", headers=doctor, files=upload())
    assert response.status_code == 200
    assert set(response.json()["question"]["voice"]) == {"answer"}
    async with client.app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(DoctorGuidance)) == 0


@pytest.mark.parametrize("data,mime,expected", [
    (b"", "audio/wav", 400),
    (b"<html>not audio</html>", "audio/webm", 400),
    (recording(), "text/plain", 415),
    (b"x" * (MAX_UPLOAD_BYTES + 1), "audio/wav", 413),
    (recording(61), "audio/wav", 413),
])
async def test_invalid_uploads_create_no_question_or_file(voice_users, data, mime, expected):
    client, patient, _, doctor_id = voice_users
    result = await client.post("/api/voice/questions", headers=patient, data={"doctor_id": doctor_id}, files=upload(data, mime))
    assert result.status_code == expected
    assert (await client.get("/api/care-team", headers=patient)).json()["questions"] == []
    assert not list(client.app.state.voice_storage.directory.glob("*.wav"))


async def test_missing_file_and_invalid_doctor_are_rejected(voice_users):
    client, patient, _, _ = voice_users
    assert (await client.post("/api/voice/questions", headers=patient, data={"doctor_id": 999999}, files=upload())).status_code == 422
    assert (await client.post("/api/voice/questions", headers=patient, data={"doctor_id": 999999})).status_code == 422


async def test_missing_stored_recording_returns_404(voice_users):
    client, patient, _, _ = voice_users
    question = await new_question(voice_users)
    for path in client.app.state.voice_storage.directory.glob("*.wav"):
        path.unlink()
    assert (await client.get(question["voice"]["question"]["url"], headers=patient)).status_code == 404


async def test_concurrent_voice_replies_only_one_wins(file_client, tmp_path):
    client = file_client
    client.app.state.voice_storage = VoiceStorage(tmp_path / "voice-concurrent")
    patient = await auth_headers(client)
    doctor = await doctor_headers(client)
    doctor_id = (await client.get("/api/doctors", headers=patient)).json()["doctors"][0]["id"]
    question = await new_question((client, patient, doctor, doctor_id))
    url = f"/api/voice/questions/{question['id']}/reply"
    replies = await asyncio.gather(*[client.post(url, headers=doctor, files=upload()) for _ in range(2)])
    assert sorted(r.status_code for r in replies) == [200, 409]
    assert len(list(client.app.state.voice_storage.directory.glob("*.wav"))) == 2
    async with client.app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(VoiceClip)) == 2


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
