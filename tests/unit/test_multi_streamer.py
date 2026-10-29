import pytest

from korshmer.network.multi_streamer import (
    PLATFORM_INGEST,
    MultiStreamer,
    make_platform_streamer,
)
from tests.unit.test_youtube_live import FakeProc, wait_for


def make(multi, name, platform, key, procs, fail_start=False):
    def popen(cmd, **kw):
        if fail_start:
            raise OSError("no connection")
        p = FakeProc(cmd)
        procs.setdefault(name, []).append(p)
        return p

    return multi.add_platform(name, platform, key, popen=popen, ffmpeg="ffmpeg")


def test_three_targets_receive_the_same_bytes():
    procs, multi = {}, MultiStreamer()
    for n, plat in (("yt", "youtube"), ("tw", "twitch"), ("fb", "facebook")):
        make(multi, n, plat, f"key-{n}", procs)
    assert multi.start() == {"yt": True, "tw": True, "fb": True}
    assert multi.push(b"\x01\x02") == 3
    assert wait_for(
        lambda: all(procs[n][0].written == [b"\x01\x02"] for n in procs)
    )
    assert procs["yt"][0].cmd[-1] == "rtmp://a.rtmp.youtube.com/live2/key-yt"
    assert procs["tw"][0].cmd[-1] == "rtmp://live.twitch.tv/app/key-tw"
    assert procs["fb"][0].cmd[-1].startswith("rtmps://")
    multi.stop()
    assert not multi.running


def test_failing_target_does_not_stop_the_others():
    procs, multi = {}, MultiStreamer()
    make(multi, "yt", "youtube", "key-a", procs)
    make(multi, "tw", "twitch", "key-b", procs, fail_start=True)
    make(multi, "fb", "facebook", "key-c", procs)
    assert multi.start() == {"yt": True, "tw": False, "fb": True}
    multi.push(b"x")
    assert wait_for(
        lambda: (
            procs["yt"][0].written == [b"x"]
            and procs["fb"][0].written == [b"x"]
        )
    )
    assert "could not start" in multi.statuses["tw"]
    assert "streaming" in multi.statuses["yt"]
    multi.stop()


def test_target_exception_is_isolated_and_reported():
    class Bad:
        running = True

        def start(self):
            raise RuntimeError("boom")

        def stop(self):
            raise RuntimeError("boom")

        def push(self, pcm):
            raise RuntimeError("boom")

    got = []

    class Good:
        running = True

        def start(self):
            return True

        def stop(self):
            pass

        def push(self, pcm):
            got.append(pcm)
            return True

    multi = MultiStreamer()
    multi.add_target("bad", Bad())
    multi.add_target("good", Good())
    assert multi.start() == {"bad": False, "good": True}
    assert multi.push(b"z") == 1 and got == [b"z"]
    multi.stop()
    assert (
        "RuntimeError" in multi.statuses["bad"]
        or "stopped" in multi.statuses["bad"]
    )


def test_status_is_reported_per_target_without_key():
    procs, seen = {}, []
    multi = MultiStreamer(on_status=lambda n, t: seen.append((n, t)))
    make(multi, "yt", "youtube", "secret-yt", procs)
    make(multi, "tw", "twitch", "secret-tw", procs)
    multi.start()
    multi.stop()
    assert {n for n, _ in seen} == {"yt", "tw"}
    assert all("secret" not in t for _, t in seen)


def test_validation():
    with pytest.raises(ValueError):
        make_platform_streamer("myspace", "key")
    with pytest.raises(ValueError):
        make_platform_streamer("twitch", "bad key!")
    with pytest.raises(ValueError):
        make_platform_streamer("twitch", "")
    multi = MultiStreamer()
    with pytest.raises(ValueError):
        multi.add_target("", object())
    with pytest.raises(ValueError):
        multi.add_target("x", object())
    multi.add_platform("a", "twitch", "k", popen=lambda c, **k: FakeProc(c))
    with pytest.raises(ValueError):
        multi.add_platform("a", "twitch", "k")
    assert multi.remove_target("a") and not multi.remove_target("a")


def test_facebook_uses_rtmps_and_all_urls_are_known():
    assert PLATFORM_INGEST["facebook"][1].startswith("rtmps://")
    assert set(PLATFORM_INGEST) == {"youtube", "twitch", "facebook"}


def test_health_counts_bitrate_drops_after_failed_writes():
    procs, multi = [], MultiStreamer()

    def popen(cmd, **kw):
        p = FakeProc(cmd)
        p.fail = bool(procs)
        procs.append(p)
        return p

    s = multi.add_platform(
        "yt", "youtube", "key-a", popen=popen, ffmpeg="ffmpeg"
    )
    assert multi.start()["yt"]
    assert s.health().bitrate_drops == 0
    procs[0].fail = True
    for _ in range(3):
        multi.push(b"x")
    assert wait_for(lambda: s.health().bitrate_drops == 1)
    assert s.health().disconnects >= 1
    multi.stop()


def test_congested_queue_rates_poor_and_reports_once():
    seen, multi = [], MultiStreamer(on_status=lambda n, t: seen.append(t))
    s = multi.add_platform(
        "yt",
        "youtube",
        "key-a",
        popen=lambda c, **k: FakeProc(c),
        ffmpeg="ffmpeg",
        max_queue=10,
    )
    s._running = True
    for _ in range(10):
        s.push(b"x")
    assert s.health().queue_fill == 1.0
    assert s.health().quality == "poor"
    s.push(b"x")
    assert s.health().dropped_blocks == 1
    assert [t for t in seen if "quality: poor" in t] == [
        "yt connection quality: poor"
    ]
    assert "key-a" not in "".join(seen)


def test_health_has_a_key_per_target():
    procs, multi = {}, MultiStreamer()
    make(multi, "yt", "youtube", "key-a", procs)
    make(multi, "tw", "twitch", "key-b", procs)
    multi.add_target(
        "other",
        type(
            "T",
            (),
            {
                "start": lambda s: True,
                "stop": lambda s: None,
                "push": lambda s, b: True,
            },
        )(),
    )
    assert set(multi.health) == {"yt", "tw"}
    assert multi.health["yt"].quality == "good"
