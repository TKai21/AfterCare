from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_voice_recorder_and_authenticated_playback_lifecycle():
    version = subprocess.run([shutil.which("node"), "-p", "process.versions.node.split('.')[0]"], capture_output=True, text=True, check=True)
    if int(version.stdout.strip()) < 18:
        pytest.skip("Node.js 18+ is needed for the browser-API test doubles")
    script = Path(__file__).with_name("voice_frontend_checks.mjs")
    result = subprocess.run([shutil.which("node"), str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
