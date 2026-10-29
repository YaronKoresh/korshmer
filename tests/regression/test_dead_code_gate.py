import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("vulture")

SRC = Path(__file__).resolve().parents[2] / "src" / "korshmer"


def test_vulture_dead_code_gate_is_clean():
    result = subprocess.run(
        [sys.executable, "-m", "vulture", str(SRC), "--min-confidence", "80"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
