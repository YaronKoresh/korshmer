import logging
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from korshmer.utils import crash_reporter

SRC = str(Path(__file__).resolve().parents[2] / "src")


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        timeout=120,
    )


def _reports(path):
    return sorted(Path(path).glob("crash-*.txt"))


def _raise_and_capture(exc):
    try:
        raise exc
    except BaseException:
        return sys.exc_info()


@pytest.fixture
def installed(tmp_path):
    crash_reporter.uninstall_crash_handler()
    crash_reporter.install_crash_handler(tmp_path, enable_faulthandler=False)
    yield tmp_path
    crash_reporter.uninstall_crash_handler()


def test_write_crash_report_contains_trace_and_environment(tmp_path):
    info = _raise_and_capture(ValueError("bad sample rate"))
    path = crash_reporter.write_crash_report(
        *info, thread_name="AudioWorker", reports_dir=tmp_path
    )
    assert path is not None and path.exists()
    text = path.read_text(encoding="utf-8")
    assert "KORSHMER crash report" in text
    assert "Thread: AudioWorker" in text
    assert "Exception: ValueError: bad sample rate" in text
    assert "Traceback (most recent call last)" in text
    assert "_raise_and_capture" in text
    assert f"Version: {crash_reporter.__version__}" in text
    assert f"Process ID: {os.getpid()}" in text
    assert not list(tmp_path.glob("*.tmp"))


def test_write_crash_report_logs_critical_with_trace(tmp_path, caplog):
    info = _raise_and_capture(RuntimeError("engine stopped"))
    with caplog.at_level(logging.CRITICAL, logger=crash_reporter.__name__):
        path = crash_reporter.write_crash_report(*info, reports_dir=tmp_path)
    record = next(r for r in caplog.records if r.levelno == logging.CRITICAL)
    assert str(path) in record.getMessage()
    assert record.exc_info[1] is info[1]


def test_report_names_are_unique_within_one_second(tmp_path):
    info = _raise_and_capture(KeyError("x"))
    paths = {
        crash_reporter.write_crash_report(*info, reports_dir=tmp_path)
        for _ in range(5)
    }
    assert len(paths) == 5
    assert len(_reports(tmp_path)) == 5


def test_old_reports_are_pruned(tmp_path, monkeypatch):
    monkeypatch.setattr(crash_reporter, "MAX_CRASH_REPORTS", 3)
    info = _raise_and_capture(KeyError("x"))
    written = [
        crash_reporter.write_crash_report(*info, reports_dir=tmp_path)
        for _ in range(6)
    ]
    remaining = _reports(tmp_path)
    assert len(remaining) == 3
    assert remaining == sorted(written[-3:])


def test_unwritable_directory_returns_none(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    info = _raise_and_capture(OSError("disk"))
    assert (
        crash_reporter.write_crash_report(*info, reports_dir=blocker / "sub")
        is None
    )


def test_sys_excepthook_writes_report_and_calls_previous(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: seen.append(a[0]))
    crash_reporter.uninstall_crash_handler()
    crash_reporter.install_crash_handler(tmp_path, enable_faulthandler=False)
    try:
        sys.excepthook(*_raise_and_capture(ZeroDivisionError("div")))
    finally:
        crash_reporter.uninstall_crash_handler()
    assert seen == [ZeroDivisionError]
    reports = _reports(tmp_path)
    assert len(reports) == 1
    assert "ZeroDivisionError: div" in reports[0].read_text(encoding="utf-8")


def test_keyboard_interrupt_does_not_write_report(installed, monkeypatch):
    monkeypatch.setitem(crash_reporter._state, "sys_hook", lambda *a: None)
    sys.excepthook(*_raise_and_capture(KeyboardInterrupt()))
    assert _reports(installed) == []


def test_thread_exception_writes_report_with_thread_name(
    installed, monkeypatch
):
    monkeypatch.setitem(crash_reporter._state, "thread_hook", lambda a: None)

    t = threading.Thread(target=lambda: None, name="StemWorker")
    crash_reporter._threading_excepthook(
        threading.ExceptHookArgs(
            (*_raise_and_capture(RuntimeError("worker failed")), t)
        )
    )
    reports = _reports(installed)
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "Thread: StemWorker" in text
    assert "RuntimeError: worker failed" in text


def test_install_is_idempotent_and_uninstall_restores(tmp_path):
    crash_reporter.uninstall_crash_handler()
    before_sys, before_thread = sys.excepthook, threading.excepthook
    first = crash_reporter.install_crash_handler(
        tmp_path, enable_faulthandler=False
    )
    second = crash_reporter.install_crash_handler(
        tmp_path / "other", enable_faulthandler=False
    )
    assert first == second == tmp_path
    assert crash_reporter.is_installed()
    assert crash_reporter._state["sys_hook"] is before_sys
    crash_reporter.uninstall_crash_handler()
    assert sys.excepthook is before_sys
    assert threading.excepthook is before_thread
    assert not crash_reporter.is_installed()


def test_uncaught_errors_in_real_process_write_reports(tmp_path):
    r = _run(
        f"""
        import sys, threading
        sys.path.insert(0, {SRC!r})
        from korshmer.utils import crash_reporter
        crash_reporter.install_crash_handler({str(tmp_path)!r})
        def worker():
            raise ValueError("thread boom")
        t = threading.Thread(target=worker, name="Worker-7")
        t.start()
        t.join()
        raise RuntimeError("main boom")
        """
    )
    assert r.returncode == 1
    assert "RuntimeError: main boom" in r.stderr
    texts = [p.read_text(encoding="utf-8") for p in _reports(tmp_path)]
    assert len(texts) == 2
    assert any("Thread: Worker-7" in t and "thread boom" in t for t in texts)
    assert any("Thread: MainThread" in t and "main boom" in t for t in texts)
    assert (tmp_path / crash_reporter.FATAL_ERRORS_FILE).exists()


def test_qt_slot_and_qthread_errors_write_reports(tmp_path):
    pytest.importorskip("PySide6")
    r = _run(
        f"""
        import sys
        sys.path.insert(0, {SRC!r})
        from PySide6.QtCore import QThread, QTimer
        from PySide6.QtWidgets import QApplication
        from korshmer.utils import crash_reporter
        crash_reporter.install_crash_handler(
            {str(tmp_path)!r}, enable_faulthandler=False
        )
        app = QApplication([])
        class Job(QThread):
            def run(self):
                raise ValueError("qthread boom")
        job = Job()
        job.start()
        job.wait()
        def slot():
            raise KeyError("slot boom")
        QTimer.singleShot(0, slot)
        QTimer.singleShot(100, app.quit)
        app.exec()
        """
    )
    assert r.returncode == 0, r.stderr
    texts = [p.read_text(encoding="utf-8") for p in _reports(tmp_path)]
    assert any("qthread boom" in t for t in texts)
    assert any("slot boom" in t for t in texts)


def test_fatal_native_error_is_dumped(tmp_path):
    r = _run(
        f"""
        import faulthandler, sys
        sys.path.insert(0, {SRC!r})
        from korshmer.utils import crash_reporter
        crash_reporter.install_crash_handler({str(tmp_path)!r})
        faulthandler._sigsegv()
        """
    )
    assert r.returncode != 0
    text = (tmp_path / crash_reporter.FATAL_ERRORS_FILE).read_text(
        encoding="utf-8"
    )
    assert "Segmentation fault" in text
    assert "KORSHMER" in text


def test_main_entry_installs_crash_handler(tmp_path):
    r = _run(
        f"""
        import sys
        sys.path.insert(0, {SRC!r})
        import korshmer.__main__ as mm
        from korshmer.utils import crash_reporter
        assert not crash_reporter.is_installed()
        crash_reporter.CRASH_REPORTS_DIR = {str(tmp_path)!r}
        class Stop(Exception):
            pass
        def stop():
            raise Stop()
        mm.register_exit_cleanup = stop
        try:
            mm.main()
        except Stop:
            pass
        assert crash_reporter.is_installed()
        assert sys.excepthook is crash_reporter._sys_excepthook
        """
    )
    assert r.returncode == 0, r.stderr
