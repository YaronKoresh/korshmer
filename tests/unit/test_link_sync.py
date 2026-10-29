import pytest

from korshmer.network.link_sync import LinkSync, link_available


class FakeLink:
    def __init__(self, bpm, loop):
        self.tempo = bpm
        self.enabled = False
        self.num_peers = 0
        self.callback = None
        self.set_calls = []

    def set_tempo_callback(self, cb):
        self.callback = cb

    def remote_change(self, bpm):
        self.tempo = bpm
        self.callback(bpm)


def _make():
    received = []
    holder = {}

    def factory(bpm, loop):
        holder["link"] = FakeLink(bpm, loop)
        return holder["link"]

    sync = LinkSync(received.append, link_factory=factory)
    return sync, received, holder


class TestLinkSyncWithFake:
    def test_start_enables_link(self):
        sync, _, holder = _make()
        try:
            assert sync.start(120)
            assert sync.enabled
            assert holder["link"].enabled is True
        finally:
            sync.stop()
        assert not sync.enabled

    def test_remote_tempo_reaches_callback(self):
        sync, received, holder = _make()
        try:
            sync.start(120)
            holder["link"].remote_change(140.0)
            assert received == [140.0]
        finally:
            sync.stop()

    def test_remote_tempo_is_clamped(self):
        sync, received, holder = _make()
        try:
            sync.start(120)
            holder["link"].remote_change(500.0)
            assert received == [300.0]
        finally:
            sync.stop()

    def test_invalid_remote_tempo_is_ignored(self):
        sync, received, holder = _make()
        try:
            sync.start(120)
            holder["link"].remote_change(float("nan"))
            holder["link"].callback("abc")
            assert received == []
        finally:
            sync.stop()

    def test_local_tempo_is_published(self):
        sync, _, holder = _make()
        try:
            sync.start(120)
            assert sync.publish_tempo(150)
            assert holder["link"].tempo == 150
        finally:
            sync.stop()

    def test_decimal_echo_of_the_same_tempo_is_not_published(self):
        sync, received, holder = _make()
        try:
            sync.start(120)
            holder["link"].remote_change(140.4)
            assert not sync.publish_tempo(140.4)
            assert not sync.publish_tempo(140.401)
            assert holder["link"].tempo == 140.4
        finally:
            sync.stop()

    def test_a_different_decimal_tempo_is_published(self):
        sync, received, holder = _make()
        try:
            sync.start(120)
            holder["link"].remote_change(140.4)
            assert sync.publish_tempo(140.0)
            assert holder["link"].tempo == 140.0
            assert sync.publish_tempo(140.37)
            assert holder["link"].tempo == pytest.approx(140.37)
        finally:
            sync.stop()

    def test_publish_without_start_is_noop(self):
        sync, _, _ = _make()
        assert not sync.publish_tempo(130)

    def test_factory_failure_returns_false(self):
        def boom(bpm, loop):
            raise RuntimeError("no link")

        sync = LinkSync(lambda b: None, link_factory=boom)
        assert not sync.start(120)
        assert not sync.enabled


@pytest.mark.skipif(not link_available(), reason="aalink not installed")
def test_real_link_session_tempo_roundtrip():
    sync = LinkSync(lambda b: None)
    try:
        assert sync.start(120)
        assert sync.publish_tempo(150)
        assert sync._link.tempo == pytest.approx(150, abs=0.01)
    finally:
        sync.stop()
