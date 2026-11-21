import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_all.py"


@pytest.fixture
def build_all():
    spec = importlib.util.spec_from_file_location("build_all_failures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exit_code(build_all, monkeypatch, *flags):
    monkeypatch.setattr(sys, "argv", ["build_all.py", *flags])
    try:
        build_all.main()
    except SystemExit as stopped:
        return stopped.code
    return 0


def test_a_failed_executable_build_ends_with_an_error_code(
    build_all, monkeypatch
):
    monkeypatch.setattr(build_all, "build_pyinstaller", lambda: False)
    assert exit_code(build_all, monkeypatch, "--pyinstaller") == 1


def test_a_failed_wheel_build_ends_with_an_error_code(build_all, monkeypatch):
    monkeypatch.setattr(build_all, "build_wheel", lambda: False)
    assert exit_code(build_all, monkeypatch, "--wheel") == 1


def test_a_successful_build_ends_normally_as_a_safeguard(
    build_all, monkeypatch
):
    monkeypatch.setattr(build_all, "build_wheel", lambda: True)
    monkeypatch.setattr(build_all, "build_pyinstaller", lambda: True)
    assert exit_code(build_all, monkeypatch, "--wheel", "--pyinstaller") == 0
