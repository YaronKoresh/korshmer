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
        self._items: deque = deque()
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
        self._target = 0
        self._expected = None
        self._end_frame = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._logger = logging.getLogger(__name__)
        self.hits = 0
        self.misses = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, start_frame: int, end_frame: int) -> None:
        self.stop()
        self._target = start_frame
        self._end_frame = end_frame
        self._expected = start_frame
        self._queue.clear()
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
        self._queue.clear()

    def take(self, start_frame: int, frames: int) -> np.ndarray | None:
        block = self._queue.take(start_frame, frames)
        if block is not None:
            self.hits += 1
            self._expected = start_frame + frames
            return block
        self.misses += 1
        if self._thread is not None and self._expected != start_frame:
            self.seek(start_frame)
        self._expected = start_frame + frames
        return None

    def _run(self, stop_event: threading.Event) -> None:
        generation = self._queue.generation
        cursor = self._target
        while not stop_event.is_set():
            current = self._queue.generation
            if current != generation:
                generation = current
                cursor = self._target
            if cursor >= self._end_frame:
                stop_event.wait(self.POLL_SECONDS * 50)
                continue
            if self._queue.is_full():
                stop_event.wait(self.POLL_SECONDS)
                continue
            stop = min(cursor + self._block_frames, self._end_frame)
            try:
                block = self._read_block(cursor, stop)
            except Exception as e:
                self._logger.error(f"Audio prefetch failed at {cursor}: {e}")
                return
            if stop_event.is_set():
                return
            self._queue.put(generation, cursor, block)
            cursor = stop
