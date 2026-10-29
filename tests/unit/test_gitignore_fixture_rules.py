from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pre_commit_does_not_sort_gitignore():
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "file-contents-sorter" not in config


def test_wav_fixtures_are_not_ignored():
    result = subprocess.run(
        ["git", "check-ignore", "-q", "tests/fixtures/new_fixture.wav"],
        cwd=ROOT,
        capture_output=True,
    )
    assert result.returncode == 1
