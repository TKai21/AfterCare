"""Dictation (speech to text) and playback of voice clips sent before clips were removed."""
import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm import LLMError
from auth import current_user
from db import get_session
from models import CareTeamQuestion, User, VoiceClip
from voice_storage import MAX_UPLOAD_BYTES, VoiceError, normalize_audio

router = APIRouter()
EMPTY_TRANSCRIPT = "I could not hear anything. Please try again."


async def decode_upload(request: Request, file: UploadFile):
    try:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    # Per-process admission limit; decoding never blocks FastAPI's event loop.
    if not hasattr(request.app.state, "voice_decoder_slots"):
        request.app.state.voice_decoder_slots = asyncio.Semaphore(2)
    try:
        async with request.app.state.voice_decoder_slots:
            return await asyncio.to_thread(normalize_audio, data, file.content_type or "")
    except VoiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/api/voice/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    """Dictation: audio in, text out. Nothing is stored or logged."""
    from routes_care_team import MAX_QUESTION_CHARS
    from routes_doctor import MAX_TEXT_CHARS

    llm = request.app.state.llm
    if llm is None:
        raise HTTPException(status_code=503, detail="Dictation is not available right now.")
    await session.commit()  # no DB read transaction during decoding or the model call
    audio = await decode_upload(request, file)
    try:
        text = await llm.transcribe(audio.data, "dictation.wav", "audio/wav")
    except LLMError as exc:
        raise HTTPException(status_code=502, detail="Dictation failed. Please try again.") from exc
    text = " ".join(text.split())[:max(MAX_QUESTION_CHARS, MAX_TEXT_CHARS)].rstrip()
    if not text:
        raise HTTPException(status_code=422, detail=EMPTY_TRANSCRIPT)
    return {"text": text}


@router.get("/api/voice/clips/{clip_id}")
async def play_clip(
    clip_id: int,
    request: Request,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    row = (await session.execute(
        select(VoiceClip, CareTeamQuestion)
        .join(CareTeamQuestion, CareTeamQuestion.id == VoiceClip.question_id)
        .where(VoiceClip.id == clip_id)
    )).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Voice clip not found")
    clip, question = row
    allowed = ((user.role == "patient" and question.user_id == user.id)
               or (user.role == "doctor" and question.doctor_id == user.id))
    if not allowed:
        raise HTTPException(status_code=404, detail="Voice clip not found")
    try:
        path = request.app.state.voice_storage.path(clip.storage_key)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Voice clip not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Voice clip not found")
    return FileResponse(path, media_type="audio/wav", headers={
        "Cache-Control": "no-store, private", "X-Content-Type-Options": "nosniff",
        "Content-Disposition": 'inline; filename="voice-clip.wav"',
    })
