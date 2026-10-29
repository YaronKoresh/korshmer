import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_all.py"


@pytest.fixture
def build_all():
    spec = importlib.util.spec_from_file_location("build_all_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def calls(build_all, monkeypatch):
    recorded = []

    def fake_run(cmd, *args, **kwargs):
        recorded.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build_all.subprocess, "run", fake_run)
    return recorded


@pytest.fixture
def exe(tmp_path):
    path = tmp_path / "korshmer"
    path.write_bytes(b"x")
    return path


def test_executable_path_has_exe_suffix_only_on_windows(build_all):
    assert build_all.executable_path("app", "Windows").name == "app.exe"
    assert build_all.executable_path("app", "Linux").name == "app"
    assert build_all.executable_path("app", "Darwin").name == "app"


def test_macos_signs_with_identity_and_verifies(build_all, calls, exe):
    env = {"KORSHMER_MACOS_IDENTITY": "Developer ID Application: Test"}
    assert build_all.sign_executable(exe, "Darwin", env) is True
    assert calls[0][0] == "codesign"
    assert "Developer ID Application: Test" in calls[0]
    assert calls[0][-1] == str(exe)
    assert calls[1][:2] == ["codesign", "--verify"]


def test_macos_without_identity_fails_before_running_anything(
    build_all, calls, exe
):
    assert build_all.sign_executable(exe, "Darwin", {}) is False
    assert calls == []


def test_macos_sign_failure_skips_verify(build_all, monkeypatch, exe):
    seen = []

    def fake_run(cmd, *args, **kwargs):
        seen.append(cmd)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(build_all.subprocess, "run", fake_run)
    env = {"KORSHMER_MACOS_IDENTITY": "id"}
    assert build_all.sign_executable(exe, "Darwin", env) is False
    assert len(seen) == 1


def test_windows_signs_with_certificate_and_verifies(
    build_all, calls, exe, monkeypatch
):
    monkeypatch.setattr(build_all, "find_signtool", lambda: "signtool.exe")
    env = {
        "KORSHMER_WIN_CERT_FILE": "cert.pfx",
        "KORSHMER_WIN_CERT_PASSWORD": "secret",
        "KORSHMER_TIMESTAMP_URL": "http://ts.example",
    }
    assert build_all.sign_executable(exe, "Windows", env) is True
    sign_cmd = calls[0]
    assert sign_cmd[:2] == ["signtool.exe", "sign"]
    assert sign_cmd[sign_cmd.index("/f") + 1] == "cert.pfx"
    assert sign_cmd[sign_cmd.index("/tr") + 1] == "http://ts.example"
    assert sign_cmd[-1] == str(exe)
    assert calls[1][:3] == ["signtool.exe", "verify", "/pa"]


def test_windows_uses_default_timestamp_server(
    build_all, calls, exe, monkeypatch
):
    monkeypatch.setattr(build_all, "find_signtool", lambda: "signtool.exe")
    env = {
        "KORSHMER_WIN_CERT_FILE": "cert.pfx",
        "KORSHMER_WIN_CERT_PASSWORD": "secret",
    }
    build_all.sign_executable(exe, "Windows", env)
    sign_cmd = calls[0]
    url = sign_cmd[sign_cmd.index("/tr") + 1]
    assert url == build_all.DEFAULT_TIMESTAMP_URL


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"KORSHMER_WIN_CERT_FILE": "cert.pfx"},
        {"KORSHMER_WIN_CERT_PASSWORD": "secret"},
    ],
)
def test_windows_without_certificate_settings_fails(
    build_all, calls, exe, monkeypatch, env
):
    monkeypatch.setattr(build_all, "find_signtool", lambda: "signtool.exe")
    assert build_all.sign_executable(exe, "Windows", env) is False
    assert calls == []


def test_windows_without_signtool_fails(build_all, calls, exe, monkeypatch):
    monkeypatch.setattr(build_all, "find_signtool", lambda: None)
    env = {
        "KORSHMER_WIN_CERT_FILE": "cert.pfx",
        "KORSHMER_WIN_CERT_PASSWORD": "secret",
    }
    assert build_all.sign_executable(exe, "Windows", env) is False
    assert calls == []


def test_missing_executable_fails(build_all, calls, tmp_path):
    missing = tmp_path / "nothing"
    assert build_all.sign_executable(missing, "Darwin", {}) is False
    assert calls == []


def test_linux_has_nothing_to_sign(build_all, calls, exe):
    assert build_all.sign_executable(exe, "Linux", {}) is True
    assert calls == []


def test_sign_flag_requires_pyinstaller(build_all, monkeypatch):
    monkeypatch.setattr(build_all.sys, "argv", ["build_all.py", "--sign"])
    with pytest.raises(SystemExit) as info:
        build_all.main()
    assert info.value.code == 2


def test_failed_signing_exits_with_error(build_all, monkeypatch):
    monkeypatch.setattr(
        build_all.sys, "argv", ["build_all.py", "--pyinstaller", "--sign"]
    )
    monkeypatch.setattr(build_all, "build_pyinstaller", lambda: True)
    monkeypatch.setattr(build_all, "sign_executable", lambda path: False)
    with pytest.raises(SystemExit) as info:
        build_all.main()
    assert info.value.code == 1


def test_successful_signing_does_not_exit(build_all, monkeypatch):
    monkeypatch.setattr(
        build_all.sys, "argv", ["build_all.py", "--pyinstaller", "--sign"]
    )
    monkeypatch.setattr(build_all, "build_pyinstaller", lambda: True)
    signed = []
    monkeypatch.setattr(
        build_all, "sign_executable", lambda path: signed.append(path) or True
    )
    build_all.main()
    assert signed == [build_all.executable_path()]
