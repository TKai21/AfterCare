"""Patient/doctor voice clips in the existing care-team queue; dictation transcription."""
import asyncio
from datetime import datetime, timezone
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import case, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from llm import LLMError
from auth import current_doctor, current_patient, current_user
from db import get_session
from doctors import doctor_labels, require_doctor
from models import CareTeamQuestion, User, VoiceClip
from voice import VOICE_ANSWER, VOICE_QUESTION, voice_views
from voice_storage import MAX_UPLOAD_BYTES, VoiceError, normalize_audio

router = APIRouter()
EMPTY_TRANSCRIPT = "I could not hear anything. Please try again."
log = logging.getLogger(__name__)


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


async def rendered_question(session, question):
    from routes_care_team import question_view
    names = await doctor_labels(session, {question.doctor_id, question.answered_by})
    voices = await voice_views(session, [question.id])
    return question_view(question, names, voices)


def discard(storage, key):
    try:
        storage.remove(key)
    except OSError:
        log.exception("Could not remove an uncommitted voice clip")


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


@router.post("/api/voice/questions", status_code=201)
async def send_question(
    request: Request,
    doctor_id: int = Form(...),
    file: UploadFile = File(...),
    user: User = Depends(current_patient),
    session: AsyncSession = Depends(get_session),
):
    from routes_documents import active_document

    await require_doctor(session, doctor_id)
    user_id = user.id
    await session.commit()  # no DB read transaction during audio decoding
    audio = await decode_upload(request, file)
    try:
        document = await active_document(session, user)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        document = None
    question = CareTeamQuestion(
        user_id=user_id, document_id=document.id if document else None,
        question=VOICE_QUESTION, reason="Voice question from the patient",
        source="patient_voice", doctor_id=doctor_id, status="open",
    )
    storage = request.app.state.voice_storage
    key = await asyncio.to_thread(storage.save, audio)
    try:
        session.add(question)
        await session.flush()
        session.add(VoiceClip(question_id=question.id, sender_id=user_id, kind="question",
                              storage_key=key, duration_ms=audio.duration_ms,
                              byte_size=len(audio.data)))
        await session.commit()
    except BaseException:
        await session.rollback()
        discard(storage, key)
        raise
    await session.refresh(question)
    return {"question": await rendered_question(session, question)}


@router.post("/api/voice/questions/{question_id}/reply")
async def send_reply(
    question_id: int,
    request: Request,
    file: UploadFile = File(...),
    doctor: User = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    question = await session.get(CareTeamQuestion, question_id)
    if question is None or question.doctor_id not in (None, doctor.id):
        raise HTTPException(status_code=404, detail="Question not found")
    if question.answer is not None:
        raise HTTPException(status_code=409, detail="This question already has an answer.")
    doctor_id = doctor.id
    await session.commit()
    audio = await decode_upload(request, file)
    storage = request.app.state.voice_storage
    key = await asyncio.to_thread(storage.save, audio)
    try:
        claimed = await session.execute(
            update(CareTeamQuestion)
            .where(CareTeamQuestion.id == question_id,
                   CareTeamQuestion.answer.is_(None),
                   or_(CareTeamQuestion.doctor_id == doctor_id,
                       CareTeamQuestion.doctor_id.is_(None)))
            .values(answer=VOICE_ANSWER, answered_by=doctor_id,
                    doctor_id=func.coalesce(CareTeamQuestion.doctor_id, doctor_id),
                    answered_at=datetime.now(timezone.utc).replace(tzinfo=None),
                    status=case((CareTeamQuestion.status == "resolved", "resolved"), else_="answered"))
            .execution_options(synchronize_session=False)
        )
        if claimed.rowcount != 1:
            raise HTTPException(status_code=409, detail="This question already has an answer.")
        session.add(VoiceClip(question_id=question_id, sender_id=doctor_id, kind="answer",
                              storage_key=key, duration_ms=audio.duration_ms,
                              byte_size=len(audio.data)))
        # No DoctorGuidance row, embedding call, transcription or model request.
        await session.commit()
    except BaseException as exc:
        await session.rollback()
        discard(storage, key)
        if isinstance(exc, IntegrityError):
            raise HTTPException(status_code=409, detail="This question already has an answer.") from exc
        raise
    await session.refresh(question)
    return {"question": await rendered_question(session, question)}


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
