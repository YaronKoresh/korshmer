import time

import pytest

from korshmer.network.youtube_live import (
    BitrateAdapter,
    YouTubeLiveStreamer,
    build_command,
    valid_stream_key,
)


class FakeStdin:
    def __init__(self, proc):
        self.proc = proc

    def write(self, data):
        if self.proc.fail:
            raise BrokenPipeError()
        self.proc.written.append(data)

    def flush(self):
        pass

    def close(self):
        pass


class FakeProc:
    def __init__(self, cmd):
        self.cmd = cmd
        self.written = []
        self.fail = False
        self.stdin = FakeStdin(self)
        self.terminated = False

    def poll(self):
        return 1 if self.fail else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def wait_for(cond, t=3.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_adapter_drops_after_failures_and_rises_after_successes():
    a = BitrateAdapter((64, 128, 192), failures_to_drop=2, successes_to_raise=3)
    assert a.bitrate == 192
    assert not a.record_failure()
    assert a.record_failure() and a.bitrate == 128
    a.record_failure()
    a.record_failure()
    assert a.bitrate == 64
    a.record_failure()
    assert not a.record_failure() and a.bitrate == 64
    assert not a.record_success() and not a.record_success()
    assert a.record_success() and a.bitrate == 128


def test_adapter_failure_resets_success_run():
    a = BitrateAdapter(
        (64, 128), start=64, failures_to_drop=5, successes_to_raise=2
    )
    a.record_success()
    a.record_failure()
    assert not a.record_success() and a.bitrate == 64


def test_adapter_validation():
    with pytest.raises(ValueError):
        BitrateAdapter(())
    with pytest.raises(ValueError):
        BitrateAdapter((64, 128), start=100)


def test_stream_key_validation():
    assert valid_stream_key("abcd-1234-efgh")
    assert not valid_stream_key("")
    assert not valid_stream_key("a b")
    assert not valid_stream_key("a/../b")


def test_streamer_sends_pcm_and_adapts_after_send_failures():
    procs, status = [], []
    broken = []

    def popen(cmd, **kw):
        p = FakeProc(cmd)
        p.fail = bool(broken)
        procs.append(p)
        return p

    adapter = BitrateAdapter(
        (64, 128), failures_to_drop=2, successes_to_raise=2
    )
    s = YouTubeLiveStreamer(
        "key-123",
        adapter=adapter,
        popen=popen,
        ffmpeg="ffmpeg",
        on_status=status.append,
    )
    assert s.start()
    assert procs[0].cmd[-1].endswith("/key-123") and "128k" in procs[0].cmd
    s.push(b"\x00\x01")
    assert wait_for(lambda: procs[0].written == [b"\x00\x01"])
    procs[0].fail = True
    broken.append(1)
    s.push(b"a")
    s.push(b"b")
    assert wait_for(lambda: s.bitrate == 64 and len(procs) >= 2)
    assert "64k" in procs[-1].cmd
    assert all("key-123" not in t for t in status)
    s.stop()
    assert not s.running


def test_missing_ffmpeg_returns_false(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    status = []
    s = YouTubeLiveStreamer("key", on_status=status.append)
    assert s.start() is False
    assert "ffmpeg" in status[0]


def test_push_never_blocks_and_drops_oldest():
    s = YouTubeLiveStreamer(
        "key", popen=lambda c, **k: FakeProc(c), ffmpeg="ffmpeg", max_queue=2
    )
    s._running = True
    assert s.push(b"1") and s.push(b"2")
    assert s.push(b"3") is False
    assert s.dropped_blocks == 1


def test_build_command_has_no_shell_and_key_last():
    cmd = build_command("ffmpeg", "k", 128, 48000, 2)
    assert cmd[0] == "ffmpeg" and cmd[-1].endswith("/k") and "-b:a" in cmd
