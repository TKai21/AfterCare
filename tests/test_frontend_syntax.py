"""Every frontend script parses, including inline <script type="module"> blocks.

A syntax error in a module (for example a top-level `return`) stops the whole
page from running with no server-side trace, so this check runs `node --check`
on each script. Skipped when Node.js is not installed."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
NODE = shutil.which("node")
INLINE_MODULE = re.compile(r'<script type="module">(.*?)</script>', re.DOTALL)

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def scripts():
    for path in sorted(FRONTEND.glob("*.js")):
        yield path.name, path.read_text(encoding="utf-8")
    for path in sorted(FRONTEND.glob("*.html")):
        for index, body in enumerate(INLINE_MODULE.findall(path.read_text(encoding="utf-8"))):
            yield f"{path.name} inline module {index}", body


@pytest.mark.parametrize("name,source", list(scripts()), ids=lambda v: v if isinstance(v, str) and len(v) < 60 else "")
def test_script_parses(name, source, tmp_path):
    module = tmp_path / "check.mjs"
    module.write_text(source, encoding="utf-8")
    result = subprocess.run([NODE, "--check", str(module)], capture_output=True, text=True)
    assert result.returncode == 0, f"{name}: {result.stderr}"
