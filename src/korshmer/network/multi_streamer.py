from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from korshmer.network.delay_buffer import DelayBuffer
from korshmer.network.youtube_live import (
    BitrateAdapter,
    YouTubeLiveStreamer,
    valid_stream_key,
)

_logger = logging.getLogger(__name__)


PLATFORM_INGEST = {
    "youtube": ("YouTube Live", "rtmp://a.rtmp.youtube.com/live2"),
    "twitch": ("Twitch", "rtmp://live.twitch.tv/app"),
    "facebook": ("Facebook Live", "rtmps://live-api-s.facebook.com:443/rtmp"),
}


def make_platform_streamer(
    platform: str,
    stream_key: str,
    sample_rate: int = 48000,
    channels: int = 2,
    on_status: Callable[[str], None] | None = None,
    **kwargs,
) -> YouTubeLiveStreamer:
    try:
        label, url = PLATFORM_INGEST[platform.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"unknown platform: {platform!r}") from None
    if not valid_stream_key(stream_key or ""):
        raise ValueError(f"invalid {label} stream key")
    kwargs.setdefault("adapter", BitrateAdapter())
    return YouTubeLiveStreamer(
        stream_key,
        sample_rate=sample_rate,
        channels=channels,
        on_status=on_status,
        ingest_url=url,
        label=label,
        **kwargs,
    )


class MultiStreamer:
    def __init__(self, on_status: Callable[[str, str], None] | None = None):
        self._on_status = on_status
        self._targets: dict[str, object] = {}
        self._status: dict[str, str] = {}
        self._lock = threading.Lock()
        self._delay = DelayBuffer()
        self._ducker = None
        self._mixer = None

    @property
    def delay(self) -> float:
        return self._delay.delay

    def set_delay(self, seconds: float) -> None:
        self._delay.set_delay(seconds)

    def silence_last(self, seconds: float) -> int:
        return self._delay.silence_last(seconds)

    @property
    def ducker(self):
        return self._ducker

    def set_ducking(self, ducker) -> None:
        self._ducker = ducker

    def set_mixer(self, mixer) -> None:
        self._mixer = mixer

    def push_sources(
        self, sources: dict[str, bytes], voice: bytes | None = None
    ) -> int:
        mixer = self._mixer
        if mixer is not None:
            block = mixer.mix(sources)
        else:
            block = next(iter(sources.values()), b"")
        return self.push(block, voice) if block else 0

    def add_target(self, name: str, target: object) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("a target needs a name")
        for attr in ("start", "stop", "push"):
            if not callable(getattr(target, attr, None)):
                raise ValueError(f"target needs a {attr}() method")
        with self._lock:
            if name in self._targets:
                raise ValueError(f"target {name!r} already exists")
            self._targets[name] = target
            self._status[name] = "idle"

    def add_platform(
        self,
        name: str,
        platform: str,
        stream_key: str,
        sample_rate: int = 48000,
        channels: int = 2,
        **kwargs,
    ) -> YouTubeLiveStreamer:
        streamer = make_platform_streamer(
            platform,
            stream_key,
            sample_rate,
            channels,
            on_status=lambda text, n=name: self._report(n, text),
            on_quality=lambda q, n=name: self._report(
                n, f"{n} connection quality: {q}"
            ),
            **kwargs,
        )
        self.add_target(name, streamer)
        return streamer

    def remove_target(self, name: str) -> bool:
        with self._lock:
            target = self._targets.pop(name, None)
            self._status.pop(name, None)
        if target is None:
            return False
        self._safe(name, target.stop)
        return True

    @property
    def names(self) -> list[str]:
        with self._lock:
            return list(self._targets)

    @property
    def statuses(self) -> dict[str, str]:
        with self._lock:
            return dict(self._status)

    @property
    def health(self) -> dict[str, object]:
        result = {}
        for name, target in self._snapshot().items():
            getter = getattr(target, "health", None)
            if callable(getter):
                try:
                    result[name] = getter()
                except Exception:
                    _logger.debug("health of %s failed", name, exc_info=True)
        return result

    @property
    def running(self) -> bool:
        return any(
            getattr(t, "running", False) for t in self._snapshot().values()
        )

    def start(self) -> dict[str, bool]:
        results = {}
        for name, target in self._snapshot().items():
            ok = bool(self._safe(name, target.start, default=False))
            results[name] = ok
            if not ok:
                self._report(name, f"{name} could not start.")
        return results

    def stop(self) -> None:
        self._delay.clear()
        for name, target in self._snapshot().items():
            self._safe(name, target.stop)
            self._report(name, "stopped")

    def push(self, pcm: bytes, voice: bytes | None = None) -> int:
        ducker = self._ducker
        if ducker is not None:
            pcm = ducker.process(pcm, voice)
        if self._delay.delay <= 0 and not self._delay.buffered_bytes:
            return self._deliver(pcm)
        accepted = len(self._snapshot())
        for block in self._delay.push(pcm):
            accepted = self._deliver(block)
        return accepted

    def _deliver(self, pcm: bytes) -> int:
        accepted = 0
        for name, target in self._snapshot().items():
            if self._safe(name, target.push, pcm, default=False):
                accepted += 1
        return accepted

    def _snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._targets)

    def _report(self, name: str, text: str) -> None:
        with self._lock:
            if name in self._status:
                self._status[name] = text
        if self._on_status is not None:
            try:
                self._on_status(name, text)
            except Exception:
                _logger.debug("status callback failed", exc_info=True)

    def _safe(self, name, func, *args, default=None):
        try:
            return func(*args)
        except Exception as e:
            _logger.debug("target %s failed: %s", name, type(e).__name__)
            self._report(name, f"{name} failed ({type(e).__name__}).")
            return default
