import pytest

from korshmer.network.rtmp_server import (
    LocalRtmpServer,
    RtmpDestination,
    valid_rtmp_url,
)
from korshmer.network.youtube_live import build_command
from tests.unit.test_youtube_live import FakeProc, wait_for


def test_build_command_custom_destination_ends_with_that_url():
    cmd = build_command(
        "ffmpeg", "", 128, 48000, 2, ingest_url="rtmp://example.com/live/abc"
    )
    assert cmd[-1] == "rtmp://example.com/live/abc"
    cmd = build_command(
        "ffmpeg", "k1", 128, 48000, 2, ingest_url="rtmps://h/app"
    )
    assert cmd[-1] == "rtmps://h/app/k1"
    assert "-listen" not in cmd


def test_build_command_listen_option_before_output():
    cmd = build_command(
        "ffmpeg", "s", 128, 48000, 2, "rtmp://127.0.0.1:1935/live", True
    )
    assert cmd.index("-listen") < len(cmd) - 2
    assert cmd[-1] == "rtmp://127.0.0.1:1935/live/s"


@pytest.mark.parametrize(
    "url",
    [
        "rtmp://example.com/live",
        "rtmps://example.com:443/app/key",
        "rtmp://127.0.0.1:1935/live",
    ],
)
def test_valid_urls(url):
    assert valid_rtmp_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "http://example.com/live",
        "file:///etc/passwd",
        "-rtmp://x",
        "rtmp://",
        "rtmp://user:pw@example.com/live",
        "rtmp://example.com/live key",
        "rtmp://example.com/li\nve",
        "rtmp://example.com:99999/live",
        "rtmp://example.com/live?x=1",
        "rtmp://example.com/" + "a" * 3000,
        None,
    ],
)
def test_invalid_urls_rejected(url):
    assert not valid_rtmp_url(url)


def test_destination_rejects_bad_url():
    with pytest.raises(ValueError):
        RtmpDestination("http://example.com/live")


def test_destination_sends_pcm_to_custom_url_and_hides_key():
    procs, status = [], []

    def popen(cmd, **kw):
        procs.append(FakeProc(cmd))
        return procs[-1]

    d = RtmpDestination(
        "rtmp://example.com/live/",
        "secret-key",
        popen=popen,
        ffmpeg="ffmpeg",
        on_status=status.append,
    )
    assert d.start()
    assert procs[0].cmd[-1] == "rtmp://example.com/live/secret-key"
    d.push(b"\x00\x01")
    assert wait_for(lambda: procs[0].written == [b"\x00\x01"])
    d.stop()
    assert status and all("secret-key" not in t for t in status)
    assert all(t.startswith("RTMP") for t in status)


def test_destination_without_key_uses_url_as_is():
    procs = []
    d = RtmpDestination(
        "rtmp://example.com/live/abc",
        popen=lambda c, **k: procs.append(FakeProc(c)) or procs[-1],
        ffmpeg="ffmpeg",
    )
    assert d.start()
    assert procs[0].cmd[-1] == "rtmp://example.com/live/abc"
    d.stop()


def test_local_server_defaults_to_loopback_and_listens():
    procs, status = [], []
    s = LocalRtmpServer(
        port=19350,
        popen=lambda c, **k: procs.append(FakeProc(c)) or procs[-1],
        ffmpeg="ffmpeg",
        on_status=status.append,
    )
    assert s.host == "127.0.0.1"
    assert s.url == "rtmp://127.0.0.1:19350/live/korshmer"
    assert s.start()
    cmd = procs[0].cmd
    assert "-listen" in cmd and cmd[-1] == s.url
    s.push(b"ab")
    assert wait_for(lambda: procs[0].written == [b"ab"])
    s.stop()
    assert not s.running
    assert any(t.startswith("RTMP server") for t in status)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"port": 0},
        {"port": 70000},
        {"port": "1935"},
        {"port": True},
        {"host": "-evil"},
        {"host": "a b"},
        {"app": "a/b"},
        {"stream": ""},
    ],
)
def test_local_server_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        LocalRtmpServer(**kwargs)


def test_missing_ffmpeg_reports_status(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    status = []
    s = LocalRtmpServer(on_status=status.append)
    assert s.start() is False
    assert "ffmpeg" in status[0]
