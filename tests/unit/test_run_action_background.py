import sys
import threading
import time

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

if not QApplication.instance():
    QApplication(sys.argv)

from korshmer.mixins.actions.base import BaseActionsMixin

MAIN = threading.current_thread()


class Host(BaseActionsMixin):
    parent = None

    def __init__(self):
        self.done = []


@pytest.fixture
def dialogs(monkeypatch):
    calls = {"info": [], "critical": []}
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *a, **k: calls["info"].append(
            (threading.current_thread(), a[2])
        ),
    )
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda *a, **k: calls["critical"].append(
            (threading.current_thread(), a[2])
        ),
    )
    return calls


def pump(cond, timeout=5):
    end = time.time() + timeout
    while time.time() < end and not cond():
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()
    return cond()


def test_action_runs_off_ui_thread_and_ui_stays_responsive(dialogs):
    host = Host()
    ran_in = []
    release = threading.Event()

    def slow():
        ran_in.append(threading.current_thread())
        release.wait(5)
        return ["/tmp/a.wav", "/tmp/b.wav"]

    returned = host._run_action(
        slow, success_msg="Done", on_done=host.done.append
    )
    assert returned is None
    assert pump(lambda: ran_in)
    assert ran_in[0] is not MAIN
    assert dialogs["info"] == []
    release.set()
    assert pump(lambda: host.done)
    assert host.done == [["/tmp/a.wav", "/tmp/b.wav"]]
    assert dialogs["info"][0][0] is MAIN


def test_error_is_shown_in_critical_dialog_on_ui_thread(dialogs):
    host = Host()

    def boom():
        raise RuntimeError("bad input")

    host._run_action(boom, success_msg="Done", on_done=host.done.append)
    assert pump(lambda: dialogs["critical"])
    thread, text = dialogs["critical"][0]
    assert thread is MAIN
    assert "bad input" in text
    assert host.done == []


def test_none_result_means_canceled_no_dialog(dialogs):
    host = Host()
    host._run_action(lambda: None, success_msg="Done", on_done=host.done.append)
    assert pump(lambda: getattr(host, "_action_worker", 1) is None)
    assert dialogs["info"] == [] and dialogs["critical"] == []
    assert host.done == []


def test_second_action_rejected_while_first_runs(dialogs):
    host = Host()
    release = threading.Event()
    second = []
    host._run_action(
        lambda: release.wait(5) and "/tmp/x.mid", success_msg="First"
    )
    host._run_action(lambda: second.append(1), success_msg="Second")
    release.set()
    assert pump(lambda: dialogs["info"])
    assert second == []
