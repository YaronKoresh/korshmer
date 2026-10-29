from __future__ import annotations

import logging
import math
import numbers
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

_logger = logging.getLogger(__name__)

INGEST_URL = "rtmp://a.rtmp.youtube.com/live2"


DEFAULT_LADDER = (64, 96, 128, 192)

MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000
MAX_QUEUE_BLOCKS = 100000

_KEY_RE = re.compile(r"^[A-Za-z0-9_\-]+(?:-[A-Za-z0-9_\-]+)*$")


def valid_stream_key(key: str) -> bool:
    return bool(key) and len(key) <= 128 and bool(_KEY_RE.match(key))


def _whole_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(value) and value == int(value)


def check_audio_format(sample_rate, channels) -> None:
    if (
        not _whole_number(sample_rate)
        or not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE
    ):
        raise ValueError(
            f"sample_rate must be a whole number from {MIN_SAMPLE_RATE} to "
            f"{MAX_SAMPLE_RATE}, got {sample_rate!r}"
        )
    if not _whole_number(channels) or channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2, got {channels!r}")


def check_queue_size(max_queue) -> None:
    if not _whole_number(max_queue) or not 1 <= max_queue <= MAX_QUEUE_BLOCKS:
        raise ValueError(
            f"max_queue must be a whole number from 1 to {MAX_QUEUE_BLOCKS}, "
            f"got {max_queue!r}"
        )


def valid_ingest_url(url) -> bool:
    if not isinstance(url, str) or not url or url != url.strip():
        return False
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in url):
        return False
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    return (
        parts.scheme in ("rtmp", "rtmps")
        and bool(parts.hostname)
        and not parts.query
        and not parts.fragment
        and (port is None or port > 0)
    )


class BitrateAdapter:
    def __init__(
        self,
        ladder: Sequence[int] = DEFAULT_LADDER,
        start: int | None = None,
        failures_to_drop: int = 3,
        successes_to_raise: int = 10,
    ):
        if not ladder or any(not _whole_number(b) or b <= 0 for b in ladder):
            raise ValueError("ladder needs positive whole bitrates")
        for name, value in (
            ("failures_to_drop", failures_to_drop),
            ("successes_to_raise", successes_to_raise),
        ):
            if not _whole_number(value):
                raise ValueError(f"{name} must be a whole number")
        self.ladder = tuple(sorted({int(b) for b in ladder}))
        if start is None:
            self._index = len(self.ladder) - 1
        else:
            if start not in self.ladder:
                raise ValueError("start must be one of the ladder bitrates")
            self._index = self.ladder.index(start)
        self.failures_to_drop = max(1, failures_to_drop)
        self.successes_to_raise = max(1, successes_to_raise)
        self._failures = 0
        self._successes = 0

    @property
    def bitrate(self) -> int:
        return self.ladder[self._index]

    def record_failure(self) -> bool:
        self._successes = 0
        self._failures += 1
        if self._failures >= self.failures_to_drop:
            self._failures = 0
            if self._index > 0:
                self._index -= 1
                return True
        return False

    def record_success(self) -> bool:
        self._failures = 0
        self._successes += 1
        if self._successes >= self.successes_to_raise:
            self._successes = 0
            if self._index < len(self.ladder) - 1:
                self._index += 1
                return True
        return False


def build_command(
    ffmpeg: str,
    stream_key: str,
    bitrate: int,
    sample_rate: int,
    channels: int,
    ingest_url: str = INGEST_URL,
    listen: bool = False,
) -> list[str]:
    destination = f"{ingest_url}/{stream_key}" if stream_key else ingest_url
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(int(sample_rate)),
        "-ac",
        str(int(channels)),
        "-i",
        "pipe:0",
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        f"{int(bitrate)}k",
        "-ar",
        "44100" if sample_rate < 44100 else str(int(sample_rate)),
        *(["-listen", "1"] if listen else []),
        "-f",
        "flv",
        destination,
    ]


QUALITY_GOOD, QUALITY_FAIR, QUALITY_POOR = "good", "fair", "poor"


HEALTH_WINDOW = 60.0


@dataclass(frozen=True)
class StreamHealth:
    bitrate: int = 0
    bitrate_drops: int = 0
    bitrate_raises: int = 0
    disconnects: int = 0
    dropped_blocks: int = 0
    queue_fill: float = 0.0
    quality: str = QUALITY_GOOD


def rate_quality(
    queue_fill: float, recent_disconnects: int, congested_at: float = 0.8
) -> str:
    if queue_fill >= congested_at or recent_disconnects >= 3:
        return QUALITY_POOR
    if queue_fill >= congested_at / 2 or recent_disconnects >= 1:
        return QUALITY_FAIR
    return QUALITY_GOOD


class YouTubeLiveStreamer:
    def __init__(
        self,
        stream_key: str,
        sample_rate: int = 48000,
        channels: int = 2,
        adapter: BitrateAdapter | None = None,
        on_status: Callable[[str], None] | None = None,
        popen: Callable[..., object] | None = None,
        ffmpeg: str | None = None,
        on_quality: Callable[[str], None] | None = None,
        max_queue: int = 64,
        congestion_ratio: float = 0.8,
        ingest_url: str = INGEST_URL,
        label: str = "YouTube Live",
        listen: bool = False,
        allow_empty_key: bool = False,
    ):
        if not (allow_empty_key and not stream_key) and not valid_stream_key(
            stream_key or ""
        ):
            raise ValueError("invalid stream key")
        check_audio_format(sample_rate, channels)
        check_queue_size(max_queue)
        if (
            isinstance(congestion_ratio, bool)
            or not isinstance(congestion_ratio, (int, float))
            or not math.isfinite(congestion_ratio)
            or not 0 < congestion_ratio <= 1
        ):
            raise ValueError(
                "congestion_ratio must be above 0 and at most 1, "
                f"got {congestion_ratio!r}"
            )
        if not valid_ingest_url(ingest_url):
            raise ValueError(
                f"ingest_url must be an rtmp:// or rtmps:// address, "
                f"got {ingest_url!r}"
            )
        self._key = stream_key or ""
        self._ingest_url = ingest_url
        self._label = label
        self._listen = bool(listen)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.adapter = adapter or BitrateAdapter()
        self._on_status = on_status
        self._on_quality = on_quality
        self._health_lock = threading.Lock()
        self._bitrate_drops = 0
        self._bitrate_raises = 0
        self._disconnects: deque = deque()
        self._quality = QUALITY_GOOD
        self._popen = popen or subprocess.Popen
        self._ffmpeg = ffmpeg
        self._queue: queue.Queue = queue.Queue(maxsize=int(max_queue))
        self._congested_at = max(1, int(max_queue * congestion_ratio))
        self._proc = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.dropped_blocks = 0
        self._congested_ratio = congestion_ratio

    def health(self) -> StreamHealth:
        with self._health_lock:
            return self._health_locked()

    def _health_locked(self) -> StreamHealth:
        now = time.monotonic()
        while self._disconnects and now - self._disconnects[0] > HEALTH_WINDOW:
            self._disconnects.popleft()
        fill = self._queue.qsize() / self._queue.maxsize
        return StreamHealth(
            bitrate=self.bitrate,
            bitrate_drops=self._bitrate_drops,
            bitrate_raises=self._bitrate_raises,
            disconnects=len(self._disconnects),
            dropped_blocks=self.dropped_blocks,
            queue_fill=fill,
            quality=rate_quality(
                fill, len(self._disconnects), self._congested_ratio
            ),
        )

    def _update_quality(self) -> None:
        with self._health_lock:
            quality = self._health_locked().quality
            if quality == self._quality:
                return
            self._quality = quality
        if self._on_quality is not None:
            try:
                self._on_quality(quality)
            except Exception:
                _logger.debug("quality callback failed", exc_info=True)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def bitrate(self) -> int:
        return self.adapter.bitrate

    def _status(self, text: str) -> None:
        _logger.info(text)
        if self._on_status is not None:
            try:
                self._on_status(text)
            except Exception:
                _logger.debug("status callback failed", exc_info=True)

    def _spawn(self) -> bool:
        ffmpeg = self._ffmpeg or shutil.which("ffmpeg")
        if not ffmpeg:
            self._status(f"{self._label} needs ffmpeg, which was not found.")
            return False
        try:
            self._proc = self._popen(
                build_command(
                    ffmpeg,
                    self._key,
                    self.bitrate,
                    self.sample_rate,
                    self.channels,
                    self._ingest_url,
                    self._listen,
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self._proc = None
            self._status(f"{self._label} could not start ({type(e).__name__}).")
            return False
        return True

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        for action in (lambda: proc.stdin.close(), proc.terminate):
            try:
                action()
            except Exception:
                pass
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def start(self) -> bool:
        if self._running:
            return True
        if not self._spawn():
            return False
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name=f"korshmer-{self._label.split()[0].lower()}",
            daemon=True,
        )
        self._thread.start()
        self._status(f"{self._label} streaming at {self.bitrate} kbit/s.")
        return True

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(5.0)
        self._thread = None
        self._kill()

    def push(self, pcm: bytes) -> bool:
        if not self._running:
            return False
        try:
            self._queue.put_nowait(bytes(pcm))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            with self._health_lock:
                self.dropped_blocks += 1
            try:
                self._queue.put_nowait(bytes(pcm))
            except queue.Full:
                pass
            self._update_quality()
            return False
        self._update_quality()
        return True

    def _change_bitrate(self) -> bool:
        self._kill()
        if not self._spawn():
            return False
        self._status(f"{self._label} bitrate now {self.bitrate} kbit/s.")
        return True

    def _run(self) -> None:
        try:
            while self._running:
                block = self._queue.get()
                if block is None or not self._running:
                    break
                congested = self._queue.qsize() >= self._congested_at
                try:
                    self._proc.stdin.write(block)
                    self._proc.stdin.flush()
                    failed = congested
                except Exception:
                    failed = True
                    if self._proc is not None and self._proc.poll() is not None:
                        with self._health_lock:
                            self._disconnects.append(time.monotonic())
                        self._status(
                            f"{self._label} connection lost, retrying."
                        )
                        if not self._spawn():
                            break
                changed = (
                    self.adapter.record_failure()
                    if failed
                    else self.adapter.record_success()
                )
                if changed:
                    with self._health_lock:
                        if failed:
                            self._bitrate_drops += 1
                        else:
                            self._bitrate_raises += 1
                self._update_quality()
                if changed and not self._change_bitrate():
                    break
        except Exception as e:
            _logger.debug("YouTube sender ended: %s", type(e).__name__)
        finally:
            if self._running:
                self._running = False
                self._status(f"{self._label} stopped: the stream ended.")
            self._kill()
