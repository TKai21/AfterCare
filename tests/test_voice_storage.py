"""Real local audio decoding/storage checks; also runnable with unittest."""
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from voice_storage import MAX_UPLOAD_BYTES, VoiceError, VoiceStorage, normalize_audio


def sample_wav(seconds=0.4):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * round(seconds * 16000))
    return output.getvalue()


class VoiceStorageTests(unittest.TestCase):
    def test_empty_upload_is_rejected(self):
        with self.assertRaises(VoiceError):
            normalize_audio(b"", "audio/wav")

    def test_oversized_upload_is_rejected(self):
        with self.assertRaises(VoiceError) as raised:
            normalize_audio(b"x" * (MAX_UPLOAD_BYTES + 1), "audio/wav")
        self.assertEqual(raised.exception.status_code, 413)

    def test_html_and_playlist_uploads_are_rejected(self):
        for payload in (b"<html>not audio</html>", b"#EXTM3U\nfile:///etc/passwd"):
            with self.assertRaises(VoiceError):
                normalize_audio(payload, "audio/webm")

    def test_unsupported_mime_is_rejected(self):
        with self.assertRaises(VoiceError) as raised:
            normalize_audio(sample_wav(), "text/plain")
        self.assertEqual(raised.exception.status_code, 415)

    def test_missing_ffmpeg_returns_service_unavailable(self):
        with patch("voice_storage.shutil.which", return_value=None):
            with self.assertRaises(VoiceError) as raised:
                normalize_audio(sample_wav(), "audio/wav")
        self.assertEqual(raised.exception.status_code, 503)

    def test_keys_cannot_escape_private_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = VoiceStorage(Path(directory))
            for key in ("../secret.wav", "/tmp/secret.wav", "bad.html"):
                with self.assertRaises(ValueError):
                    storage.path(key)

    @unittest.skipUnless(shutil.which("ffmpeg"), "Install FFmpeg for decoder tests")
    def test_normalized_wav_and_private_storage_roundtrip(self):
        audio = normalize_audio(sample_wav(), "audio/wav")
        self.assertAlmostEqual(audio.duration_ms, 400, delta=5)
        with wave.open(io.BytesIO(audio.data), "rb") as decoded:
            self.assertEqual(decoded.getnchannels(), 1)
            self.assertEqual(decoded.getframerate(), 16000)
        with tempfile.TemporaryDirectory() as directory:
            storage = VoiceStorage(Path(directory) / "private")
            key = storage.save(audio)
            path = storage.path(key)
            self.assertEqual(path.read_bytes(), audio.data)
            self.assertEqual(path.stat().st_mode & 0o077, 0)
            storage.remove(key)
            self.assertFalse(path.exists())

    @unittest.skipUnless(shutil.which("ffmpeg"), "Install FFmpeg for decoder tests")
    def test_long_recording_is_rejected_from_decoded_duration(self):
        with self.assertRaises(VoiceError) as raised:
            normalize_audio(sample_wav(61), "audio/wav")
        self.assertEqual(raised.exception.status_code, 413)

    @unittest.skipUnless(shutil.which("ffmpeg"), "Install FFmpeg for decoder tests")
    def test_very_short_recording_is_rejected(self):
        with self.assertRaises(VoiceError):
            normalize_audio(sample_wav(0.02), "audio/wav")

    @unittest.skipUnless(shutil.which("ffmpeg"), "Install FFmpeg for decoder tests")
    def test_browser_recording_containers_decode(self):
        formats = [
            (["-c:a", "libopus", "-f", "webm"], "audio/webm;codecs=opus"),
            (["-c:a", "libopus", "-f", "ogg"], "audio/ogg"),
            (["-c:a", "aac", "-movflags", "frag_keyframe+empty_moov", "-f", "mp4"], "audio/mp4"),
        ]
        for options, mime in formats:
            with self.subTest(mime=mime):
                result = subprocess.run(
                    [shutil.which("ffmpeg"), "-v", "error", "-f", "wav", "-i", "pipe:0", *options, "pipe:1"],
                    input=sample_wav(), capture_output=True, timeout=15, check=True,
                )
                audio = normalize_audio(result.stdout, mime)
                self.assertGreater(audio.duration_ms, 300)
                self.assertLess(audio.duration_ms, 600)


if __name__ == "__main__":
    unittest.main()
