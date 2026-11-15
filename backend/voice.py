"""Presentation metadata for attachments; deliberately contains no AI hooks."""
from sqlalchemy import select

from models import VoiceClip

VOICE_QUESTION = "Voice question"
VOICE_ANSWER = "Voice reply"


async def voice_views(session, question_ids) -> dict:
    ids = set(question_ids)
    if not ids:
        return {}
    clips = await session.scalars(select(VoiceClip).where(VoiceClip.question_id.in_(ids)))
    out = {}
    for clip in clips:
        out.setdefault(clip.question_id, {})[clip.kind] = {
            "id": clip.id, "url": f"/api/voice/clips/{clip.id}",
            "duration_ms": clip.duration_ms, "byte_size": clip.byte_size,
            "created_at": clip.created_at.isoformat() if clip.created_at else None,
        }
    return out
