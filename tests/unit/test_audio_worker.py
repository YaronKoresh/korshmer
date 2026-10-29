import threading

import numpy as np

from korshmer.workers.audio_worker import AudioBlockQueue, AudioPrefetcher


def _block(start, frames=4):
    return np.full((frames, 1), float(start), dtype="float32")


class TestAudioBlockQueue:
    def test_take_returns_block_with_matching_start_and_length(self):
        queue = AudioBlockQueue(capacity=4)
        queue.put(queue.generation, 0, _block(0))
        queue.put(queue.generation, 4, _block(4))

        assert np.array_equal(queue.take(0, 4), _block(0))
        assert np.array_equal(queue.take(4, 4), _block(4))

    def test_take_misses_on_wrong_start_or_length(self):
        queue = AudioBlockQueue(capacity=4)
        queue.put(queue.generation, 8, _block(8))

        assert queue.take(0, 4) is None
        assert queue.take(8, 2) is None
        assert queue.take(8, 4) is not None

    def test_clear_drops_blocks_from_old_generation(self):
        queue = AudioBlockQueue(capacity=4)
        old = queue.generation
        queue.put(old, 0, _block(0))
        queue.clear()
        queue.put(old, 4, _block(4))
        queue.put(queue.generation, 0, _block(0))

        assert queue.take(0, 4) is not None
        assert queue.take(4, 4) is None

    def test_is_full_respects_capacity(self):
        queue = AudioBlockQueue(capacity=2)
        queue.put(queue.generation, 0, _block(0))
        assert not queue.is_full()
        queue.put(queue.generation, 4, _block(4))
        assert queue.is_full()


class TestAudioPrefetcher:
    def test_produces_blocks_in_order_up_to_end(self):
        reads = []

        def read_block(start, stop):
            reads.append((start, stop))
            return _block(start, stop - start)

        prefetch = AudioPrefetcher(read_block, block_frames=4, capacity=8)
        prefetch.start(0, 10)

        collected = []
        for start in (0, 4, 8):
            frames = 4 if start < 8 else 2
            block = None
            for _ in range(2000):
                block = prefetch.take(start, frames)
                if block is not None:
                    break
                threading.Event().wait(0.001)
            collected.append(block)
        prefetch.stop()

        assert [len(b) for b in collected] == [4, 4, 2]
        assert reads == [(0, 4), (4, 8), (8, 10)]
        assert prefetch.hits == 3

    def test_seek_moves_producer_to_new_position(self):
        def read_block(start, stop):
            return _block(start, stop - start)

        prefetch = AudioPrefetcher(read_block, block_frames=4, capacity=2)
        prefetch.start(0, 100)
        prefetch.seek(40)

        block = None
        for _ in range(2000):
            block = prefetch.take(40, 4)
            if block is not None:
                break
            threading.Event().wait(0.001)
        prefetch.stop()

        assert block is not None
        assert np.array_equal(block, _block(40))

    def test_stop_joins_producer_thread(self):
        prefetch = AudioPrefetcher(
            lambda start, stop: _block(start, stop - start),
            block_frames=4,
        )
        prefetch.start(0, 1000)
        assert prefetch.running
        prefetch.stop()
        assert not prefetch.running
