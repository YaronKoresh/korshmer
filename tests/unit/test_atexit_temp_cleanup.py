import os
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[2] / "src")


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )


def test_main_entry_registers_exit_cleanup_without_audio_processing(tmp_path):
    (tmp_path / "temp_preview.wav").write_bytes(b"x")
    (tmp_path / "temp_spectrum.png").write_bytes(b"x")
    r = _run(
        f"""
        import sys
        sys.path.insert(0, {SRC!r})
        import korshmer.__main__ as mm
        import korshmer.audio.io as io_mod
        io_mod.DATA_DIR = io_mod.TEMP_DIR = {str(tmp_path)!r}
        mm.register_exit_cleanup()
        mm.register_exit_cleanup()
        """
    )
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "temp_preview.wav").exists()
    assert not (tmp_path / "temp_spectrum.png").exists()


def test_register_is_idempotent(monkeypatch):
    import atexit

    from korshmer.audio import io as m

    calls = []
    monkeypatch.setattr(atexit, "register", lambda f: calls.append(f))
    monkeypatch.setattr(m, "_cleanup_registered", False)
    m.register_temp_cleanup()
    m.register_temp_cleanup()
    assert calls == [m.cleanup_temp_files]
