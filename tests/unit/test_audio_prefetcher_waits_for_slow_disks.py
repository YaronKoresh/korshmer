import threading

import numpy as np

from korshmer.workers.audio_worker import AudioPrefetcher

BLOCK = 4


def _block(start, stop):
    return np.arange(start, stop, dtype="float32").reshape(-1, 1)


class SlowDisk:
    def __init__(self):
        self.reads = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail_from = None

    def read(self, start, stop):
        self.reads.append((start, stop))
        self.entered.set()
        self.release.wait(10)
        if self.fail_from is not None and start >= self.fail_from:
            raise OSError("the drive went away")
        return _block(start, stop)


def _fetcher(disk, capacity=8):
    return AudioPrefetcher(disk.read, block_frames=BLOCK, capacity=capacity)


def _read_when_ready(prefetch, frames, start, gain=1.0):
    out = np.zeros((frames, 1), dtype="float32")
    done = 0
    for _ in range(2000):
        done += prefetch.read_into(out[done:], start + done, gain)
        if done == frames:
            break
        threading.Event().wait(0.001)
    return out, done


def test_asking_again_for_a_block_still_being_read_does_not_restart_the_read():
    disk = SlowDisk()
    prefetch = _fetcher(disk)
    prefetch.start(0, 100)
    try:
        assert disk.entered.wait(10)
        misses = [prefetch.take(0, BLOCK) for _ in range(5)]
        disk.release.set()
        block = None
        for _ in range(2000):
            block = prefetch.take(0, BLOCK)
            if block is not None:
                break
            threading.Event().wait(0.001)
    finally:
        disk.release.set()
        prefetch.stop()

    assert misses == [None] * 5
    assert np.array_equal(block, _block(0, BLOCK))
    assert disk.reads.count((0, BLOCK)) == 1


def test_a_waiting_listener_gets_every_frame_once_the_disk_answers():
    disk = SlowDisk()
    prefetch = _fetcher(disk)
    prefetch.start(0, 10)
    try:
        assert disk.entered.wait(10)
        out = np.zeros((3, 1), dtype="float32")
        waited = [prefetch.read_into(out, 0) for _ in range(5)]
        disk.release.set()
        heard, done = _read_when_ready(prefetch, 10, 0)
    finally:
        disk.release.set()
        prefetch.stop()

    assert waited == [0] * 5
    assert done == 10
    assert np.array_equal(heard, _block(0, 10))
    assert disk.reads == [(0, 4), (4, 8), (8, 10)]


def test_reading_scales_the_sound_by_the_volume():
    disk = SlowDisk()
    disk.release.set()
    prefetch = _fetcher(disk)
    prefetch.start(0, 8)
    try:
        heard, _ = _read_when_ready(prefetch, 8, 0, gain=0.5)
    finally:
        prefetch.stop()

    assert np.allclose(heard, _block(0, 8) * 0.5)


def test_the_first_wait_after_a_start_or_a_seek_is_not_a_dropout():
    disk = SlowDisk()
    prefetch = _fetcher(disk)
    prefetch.start(0, 100)
    try:
        starting = prefetch.priming
        disk.release.set()
        _read_when_ready(prefetch, BLOCK, 0)
        flowing = prefetch.priming
        prefetch.seek(40)
        seeking = prefetch.priming
    finally:
        prefetch.stop()

    assert (starting, flowing, seeking) == (True, False, True)


def test_a_looped_region_is_fetched_round_and_round_without_a_gap():
    disk = SlowDisk()
    prefetch = _fetcher(disk)
    prefetch.start(0, 100)
    assert disk.entered.wait(10)
    prefetch.set_loop(10, 16, 10)
    disk.release.set()
    try:
        first, _ = _read_when_ready(prefetch, 6, 10)
        second, _ = _read_when_ready(prefetch, 6, 10)
        third, _ = _read_when_ready(prefetch, 6, 10)
    finally:
        prefetch.stop()

    assert np.array_equal(first, _block(10, 16))
    assert np.array_equal(second, _block(10, 16))
    assert np.array_equal(third, _block(10, 16))
    assert disk.reads[1:5] == [(10, 14), (14, 16), (10, 14), (14, 16)]


def test_leaving_the_loop_carries_on_past_its_end():
    disk = SlowDisk()
    disk.release.set()
    prefetch = _fetcher(disk)
    prefetch.start(0, 100)
    prefetch.set_loop(10, 16, 10)
    try:
        _read_when_ready(prefetch, 6, 10)
        prefetch.clear_loop(10)
        heard, _ = _read_when_ready(prefetch, 12, 10)
    finally:
        prefetch.stop()

    assert np.array_equal(heard, _block(10, 22))


def test_a_disk_that_fails_is_reported_instead_of_silently_ending():
    disk = SlowDisk()
    disk.fail_from = 0
    prefetch = _fetcher(disk)
    prefetch.start(0, 100)
    try:
        assert disk.entered.wait(10)
        healthy = prefetch.failed
        disk.release.set()
        for _ in range(2000):
            if prefetch.failed:
                break
            threading.Event().wait(0.001)
    finally:
        prefetch.stop()

    assert not healthy
    assert prefetch.failed
    assert "the drive went away" in prefetch.error
