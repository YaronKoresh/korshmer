from __future__ import annotations

import logging
import queue
import re
import threading
from collections.abc import Callable

from korshmer.network.youtube_live import (
    check_audio_format,
    check_queue_size,
)

_logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9 _\-.]{1,64}$")


def valid_source_name(name: str) -> bool:
    return (
        isinstance(name, str)
        and bool(_NAME_RE.match(name))
        and bool(name.strip())
    )


class CyndilibSender:
    def __init__(self, name: str, sample_rate: int, channels: int):
        from cyndilib.audio_frame import AudioSendFrame
        from cyndilib.sender import Sender

        self._frame = AudioSendFrame()
        self._frame.sample_rate = sample_rate
        self._frame.num_channels = channels
        self._frame.set_max_num_samples(sample_rate)
        self._sender = Sender(name)
        self._sender.set_audio_frame(self._frame)
        self._sender.open()

    def write(self, planar) -> None:
        self._sender.write_audio(planar)

    def close(self) -> None:
        self._sender.close()


class NdiOutput:
    def __init__(
        self,
        source_name: str = "KORSHMER",
        sample_rate: int = 48000,
        channels: int = 2,
        on_status: Callable[[str], None] | None = None,
        sender_factory: Callable[[str, int, int], object] | None = None,
        max_queue: int = 64,
    ):
        if not valid_source_name(source_name):
            raise ValueError("invalid NDI source name")
        check_audio_format(sample_rate, channels)
        check_queue_size(max_queue)
        self.source_name = source_name
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._on_status = on_status
        self._factory = sender_factory or CyndilibSender
        self._queue: queue.Queue = queue.Queue(maxsize=int(max_queue))
        self._sender = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.dropped_blocks = 0

    @property
    def running(self) -> bool:
        return self._running

    def _status(self, text: str) -> None:
        _logger.info(text)
        if self._on_status is not None:
            try:
                self._on_status(text)
            except Exception:
                _logger.debug("status callback failed", exc_info=True)

    def start(self) -> bool:
        if self._running:
            return True
        try:
            self._sender = self._factory(
                self.source_name, self.sample_rate, self.channels
            )
        except ImportError:
            self._sender = None
            self._status(
                "NDI output is not available: install the 'cyndilib' package."
            )
            return False
        except Exception as e:
            self._sender = None
            self._status(f"NDI output could not start ({type(e).__name__}).")
            return False
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="korshmer-ndi", daemon=True
        )
        self._thread.start()
        self._status(f"NDI source '{self.source_name}' is on the network.")
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
        self._close()

    def _close(self) -> None:
        sender, self._sender = self._sender, None
        if sender is None:
            return
        try:
            sender.close()
        except Exception:
            _logger.debug("NDI close failed", exc_info=True)

    def push(self, pcm: bytes) -> bool:
        if not self._running:
            return False
        try:
            data = bytes(pcm)
        except Exception:
            return False
        try:
            self._queue.put_nowait(data)
            return True
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self.dropped_blocks += 1
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                pass
            return False

    def _planar(self, block: bytes):
        import numpy as np

        frame_bytes = 2 * self.channels
        usable = len(block) - len(block) % frame_bytes
        if usable <= 0:
            return None
        samples = np.frombuffer(block[:usable], dtype="<i2").astype(np.float32)
        samples /= 32768.0
        return np.ascontiguousarray(samples.reshape(-1, self.channels).T)

    def _run(self) -> None:
        try:
            while self._running:
                block = self._queue.get()
                if block is None or not self._running:
                    break
                planar = self._planar(block)
                if planar is None:
                    continue
                self._sender.write(planar)
        except Exception as e:
            _logger.debug("NDI sender ended: %s", type(e).__name__)
        finally:
            if self._running:
                self._running = False
                self._status("NDI output stopped: sending failed.")
            self._close()
