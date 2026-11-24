import logging
import threading
from collections import deque
from collections.abc import Callable

import numpy as np


class AudioBlockQueue:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._items: deque[tuple[int, int, np.ndarray]] = deque()
        self.generation = 0

    def __len__(self) -> int:
        return len(self._items)

    def is_full(self) -> bool:
        return len(self._items) >= self._capacity

    def put(self, generation: int, start: int, block: np.ndarray) -> None:
        self._items.append((generation, start, block))

    def take(self, start: int, frames: int) -> np.ndarray | None:
        while True:
            try:
                generation, head_start, block = self._items[0]
            except IndexError:
                return None
            if generation != self.generation or head_start < start:
                self._pop()
                continue
            if head_start == start and len(block) == frames:
                self._pop()
                return block
            return None

    def take_from(self, start: int) -> np.ndarray | None:
        while True:
            try:
                generation, head_start, block = self._items[0]
            except IndexError:
                return None
            if generation != self.generation:
                self._pop()
                continue
            if head_start == start:
                self._pop()
                return block
            return None

    def clear(self) -> None:
        self.generation += 1
        self._items.clear()

    def _pop(self) -> None:
        try:
            self._items.popleft()
        except IndexError:
            pass


class AudioPrefetcher:
    POLL_SECONDS = 0.002

    def __init__(
        self,
        read_block: Callable[[int, int], np.ndarray],
        block_frames: int,
        capacity: int = 8,
    ):
        if block_frames < 1:
            raise ValueError("block_frames must be at least 1")
        self._read_block = read_block
        self._block_frames = block_frames
        self._queue = AudioBlockQueue(capacity)
        self._reach = capacity * block_frames
        self._target = 0
        self._expected: int | None = None
        self._floor = 0
        self._loop: tuple[int, int] | None = None
        self._carry: np.ndarray | None = None
        self._carry_start = 0
        self._priming = True
        self._primed = threading.Event()
        self.error: str | None = None
        self._end_frame = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._logger = logging.getLogger(__name__)
        self.hits = 0
        self.misses = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def priming(self) -> bool:
        return self._priming

    def wait_until_primed(self, timeout: float) -> bool:
        return self._primed.wait(timeout)

    def start(self, start_frame: int, end_frame: int) -> None:
        self.stop()
        self._end_frame = end_frame
        self.error = None
        self.seek(start_frame)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            args=(self._stop_event,),
            name="korshmer-audio-prefetch",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        self._thread = None
        if thread is not None:
            self._stop_event.set()
            thread.join(timeout=2.0)
        self._queue.clear()

    def seek(self, start_frame: int) -> None:
        self._target = start_frame
        self._expected = start_frame
        self._floor = start_frame
        self._carry = None
        self._priming = True
        self._primed.clear()
        self._queue.clear()

    def set_loop(self, start_frame: int, end_frame: int, position: int) -> None:
        self._loop = (start_frame, end_frame)
        self.seek(position)

    def clear_loop(self, position: int) -> None:
        self._loop = None
        self.seek(position)

    def take(self, start_frame: int, frames: int) -> np.ndarray | None:
        block = self._queue.take(start_frame, frames)
        if block is not None:
            self.hits += 1
            self._consumed(start_frame + frames)
            return block
        self.misses += 1
        if (
            self._thread is not None
            and self._expected != start_frame
            and not self._on_its_way(start_frame)
        ):
            self.seek(start_frame)
        self._expected = start_frame + frames
        return None

    def read_into(
        self, target: np.ndarray, start_frame: int, gain: float = 1.0
    ) -> int:
        frames = len(target)
        filled = 0
        while filled < frames:
            position = start_frame + filled
            block = self._carry
            first = self._carry_start
            if block is None or not first <= position < first + len(block):
                block = self._next_block(position)
                if block is None:
                    break
                self._carry = block
                self._carry_start = first = position
            offset = position - first
            count = min(frames - filled, len(block) - offset)
            np.multiply(
                block[offset : offset + count],
                gain,
                out=target[filled : filled + count],
            )
            filled += count
        return filled

    def _next_block(self, position: int) -> np.ndarray | None:
        block = self._queue.take_from(position)
        if block is None:
            self.misses += 1
            if self._thread is not None and not self._on_its_way(position):
                self.seek(position)
            return None
        self.hits += 1
        self._consumed(position + len(block))
        return block

    def _on_its_way(self, position: int) -> bool:
        ahead = position - self._floor
        return 0 <= ahead <= self._reach and ahead % self._block_frames == 0

    def _consumed(self, end_frame: int) -> None:
        loop = self._loop
        if loop is not None and end_frame == loop[1]:
            end_frame = loop[0]
        self._floor = end_frame
        self._expected = end_frame
        self._priming = False

    def _run(self, stop_event: threading.Event) -> None:
        generation = self._queue.generation
        cursor = self._target
        while not stop_event.is_set():
            current = self._queue.generation
            if current != generation:
                generation = current
                cursor = self._target
            loop = self._loop
            if loop is not None and cursor >= loop[1]:
                cursor = loop[0]
            if cursor >= self._end_frame:
                stop_event.wait(self.POLL_SECONDS * 50)
                continue
            if self._queue.is_full():
                stop_event.wait(self.POLL_SECONDS)
                continue
            stop = min(cursor + self._block_frames, self._end_frame)
            if loop is not None:
                stop = min(stop, loop[1])
            try:
                block = self._read_block(cursor, stop)
            except Exception as e:
                self._logger.error(f"Audio prefetch failed at {cursor}: {e}")
                self.error = str(e) or type(e).__name__
                return
            if stop_event.is_set():
                return
            self._queue.put(generation, cursor, block)
            if self._queue.generation == generation:
                self._primed.set()
            cursor = stop
