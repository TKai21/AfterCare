"""Bounded audio validation and private, normalized WAV storage.

No speech recognition, text generation or third-party audio API is used.
FFmpeg decodes local bytes only; it cannot open file/network URLs in uploads.
"""
from dataclasses import dataclass
import io
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid
import wave

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_SECONDS = 60
SAMPLE_RATE = 16000
ALLOWED_TYPES = {"audio/webm", "audio/ogg", "audio/mp4", "audio/x-m4a",
                 "audio/wav", "audio/x-wav", "audio/wave"}
KEY = re.compile(r"[a-f0-9]{32}\.wav\Z")
log = logging.getLogger(__name__)


class VoiceError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class NormalizedAudio:
    data: bytes
    duration_ms: int


def normalize_audio(data: bytes, content_type: str) -> NormalizedAudio:
    if not data:
        raise VoiceError("The recording is empty. Record a new clip.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise VoiceError("Voice clips must be smaller than 5 MB.", 413)
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime not in ALLOWED_TYPES:
        raise VoiceError("This recording format is not supported. Try another browser.", 415)
    # Refuse playlists/text before a media decoder sees them.
    recognizable = (data.startswith((b"\x1aE\xdf\xa3", b"OggS"))
                    or (data.startswith(b"RIFF") and data[8:12] == b"WAVE")
                    or data[4:8] == b"ftyp")
    if not recognizable:
        raise VoiceError("The uploaded file is not a supported audio recording.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log.error("Voice uploads require FFmpeg on the backend host")
        raise VoiceError("Voice messages are temporarily unavailable. Please try again later.", 503)
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
               "-protocol_whitelist", "pipe", "-threads", "1", "-i", "pipe:0",
               "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-t", str(MAX_SECONDS + 1), "-f", "s16le", "pipe:1"]
    try:
        result = subprocess.run(command, input=data, capture_output=True, timeout=20, check=False)
    except subprocess.TimeoutExpired as exc:
        raise VoiceError("The recording took too long to process. Try a shorter clip.") from exc
    except OSError as exc:
        raise VoiceError("The audio processor is unavailable.", 503) from exc
    if result.returncode != 0 or not result.stdout:
        raise VoiceError("The recording could not be read. Record it again.")
    seconds = len(result.stdout) / (2 * SAMPLE_RATE)
    # A small codec/frame tolerance accommodates the browser's stop timer.
    if seconds > MAX_SECONDS + 0.25:
        raise VoiceError("Recordings may be at most 60 seconds long.", 413)
    if seconds < 0.1:
        raise VoiceError("The recording is too short. Record a new clip.")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(result.stdout)
    return NormalizedAudio(output.getvalue(), round(seconds * 1000))


class VoiceStorage:
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()

    def path(self, key: str) -> Path:
        if not KEY.fullmatch(key):
            raise ValueError("Invalid voice storage key.")
        path = self.directory / key
        if path.is_symlink():
            raise ValueError("Invalid voice storage path.")
        return path

    def save(self, audio: NormalizedAudio) -> str:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = uuid.uuid4().hex + ".wav"
        path = self.path(key)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(audio.data)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return key

    def remove(self, key: str):
        self.path(key).unlink(missing_ok=True)
