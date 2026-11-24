import logging
from urllib.parse import urlsplit

import pytest

from korshmer.network import multi_streamer, youtube_live
from korshmer.network.multi_streamer import make_platform_streamer
from korshmer.network.youtube_live import YouTubeLiveStreamer, build_command

KEY = "abcd-efgh-ijkl-mnop"


class Proc:
    def __init__(self, cmd):
        self.cmd = cmd
        self.stdin = self

    def write(self, data):
        pass

    def flush(self):
        pass

    def close(self):
        pass

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def started_command(make):
    started = []

    def popen(cmd, **options):
        started.append(cmd)
        return Proc(cmd)

    streamer = make(popen)
    assert streamer.start()
    streamer.stop()
    return started[0]


def test_the_youtube_command_uses_the_encrypted_address():
    assert build_command("ffmpeg", KEY, 128, 48000, 2)[-1] == (
        f"rtmps://a.rtmps.youtube.com:443/live2/{KEY}"
    )


def test_a_youtube_streamer_uses_the_encrypted_address_by_default():
    cmd = started_command(
        lambda popen: YouTubeLiveStreamer(KEY, popen=popen, ffmpeg="ffmpeg")
    )
    address = urlsplit(cmd[-1])
    assert address.scheme == "rtmps"
    assert address.hostname == "a.rtmps.youtube.com"
    assert address.port == 443


@pytest.mark.parametrize("platform", ["youtube", "twitch", "facebook"])
def test_every_platform_sends_its_stream_key_encrypted(platform):
    cmd = started_command(
        lambda popen: make_platform_streamer(
            platform, KEY, popen=popen, ffmpeg="ffmpeg"
        )
    )
    address = urlsplit(cmd[-1])
    assert address.scheme == "rtmps"
    assert address.path.endswith("/" + KEY)


def test_both_streaming_modules_share_one_address_table():
    assert multi_streamer.PLATFORM_INGEST is youtube_live.PLATFORM_INGEST
    assert youtube_live.INGEST_URL == youtube_live.PLATFORM_INGEST["youtube"][1]


@pytest.mark.parametrize("platform", ["youtube", "twitch", "facebook"])
def test_starting_a_stream_logs_the_command_without_the_key(platform, caplog):
    with caplog.at_level(logging.DEBUG):
        started_command(
            lambda popen: make_platform_streamer(
                platform, KEY, popen=popen, ffmpeg="ffmpeg"
            )
        )
    messages = [record.getMessage() for record in caplog.records]
    assert any("rtmps://" in m and "****" in m for m in messages)
    assert all(KEY not in m for m in messages)


def test_a_failed_start_never_writes_the_key_to_the_log_as_a_safeguard(caplog):
    def popen(cmd, **options):
        raise OSError(f"cannot run {' '.join(cmd)}")

    streamer = YouTubeLiveStreamer(KEY, popen=popen, ffmpeg="ffmpeg")
    with caplog.at_level(logging.DEBUG):
        assert streamer.start() is False
    assert all(KEY not in r.getMessage() for r in caplog.records)
