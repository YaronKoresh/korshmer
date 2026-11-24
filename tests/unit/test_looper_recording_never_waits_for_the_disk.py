import os
import tempfile
import threading

import numpy as np
import pytest

from korshmer.core.looper import Looper


class Disk:
    def __init__(self, monkeypatch, folder):
        self.folder = folder
        self.save_threads = []
        self.folder_threads = []
        self.saved = threading.Semaphore(0)
        self.saving_allowed = threading.Event()
        self.saving_allowed.set()
        self.loading = threading.Event()
        self.loading_allowed = threading.Event()
        self.loading_allowed.set()
        self._save = np.save
        self._load = np.load
        monkeypatch.setattr(np, "save", self.save)
        monkeypatch.setattr(np, "load", self.load)
        monkeypatch.setattr(tempfile, "mkdtemp", self.make_folder)

    def make_folder(self, prefix=""):
        self.folder_threads.append(threading.get_ident())
        os.makedirs(self.folder, exist_ok=True)
        return str(self.folder)

    def save(self, path, array):
        self.save_threads.append(threading.get_ident())
        self.saving_allowed.wait(10)
        self._save(path, array)
        self.saved.release()

    def load(self, path, **kwargs):
        self.loading.set()
        self.loading_allowed.wait(10)
        return self._load(path, **kwargs)

    def wait_for_saves(self, count):
        return all(self.saved.acquire(timeout=10) for _ in range(count))

    def files(self):
        return sorted(os.listdir(self.folder))


@pytest.fixture
def disk(monkeypatch, tmp_path):
    return Disk(monkeypatch, tmp_path / "layers")


@pytest.fixture
def looper():
    lp = Looper(bpm=240, bars=1, sample_rate=1000)
    yield lp
    lp.close()


def _on_another_thread(action):
    idents = []

    def run():
        idents.append(threading.get_ident())
        action()

    worker = threading.Thread(target=run)
    worker.start()
    return worker, idents


def _take(looper, value):
    return np.full((looper.loop_frames, 1), value, np.float32)


def _play(looper):
    out = np.zeros((looper.loop_frames, 1), dtype=np.float32)
    looper.mix_out(out)
    return out


def _record(looper, value):
    looper.arm_record()
    if looper.armed:
        _play(looper)
    looper.feed_input(_take(looper, value))


def test_a_finished_layer_is_saved_away_from_the_audio_thread(disk, looper):
    looper.arm_record()
    take = _take(looper, 0.25)

    audio, audio_idents = _on_another_thread(lambda: looper.feed_input(take))
    audio.join(10)

    assert disk.wait_for_saves(1)
    assert audio_idents[0] not in disk.save_threads
    assert audio_idents[0] not in disk.folder_threads


def test_a_new_layer_is_heard_while_the_disk_is_still_busy(disk, looper):
    disk.saving_allowed.clear()
    looper.arm_record()
    take = _take(looper, 0.25)

    audio, _ = _on_another_thread(lambda: looper.feed_input(take))
    audio.join(2)
    finished_in_time = not audio.is_alive()
    heard = _play(looper)
    disk.saving_allowed.set()

    assert finished_in_time
    assert np.allclose(heard, 0.25)


def test_playback_keeps_running_while_an_undo_waits_for_a_slow_disk(
    disk, looper
):
    _record(looper, 0.25)
    _record(looper, 0.5)
    assert disk.wait_for_saves(2)
    disk.loading_allowed.clear()
    undo, _ = _on_another_thread(looper.undo_layer)
    assert disk.loading.wait(10)

    def overdub_and_play():
        looper.arm_record()
        _play(looper)

    audio, _ = _on_another_thread(overdub_and_play)
    audio.join(2)
    played_in_time = not audio.is_alive()
    disk.loading_allowed.set()
    undo.join(10)
    audio.join(10)

    assert played_in_time


class TestBehaviourThatMustStayAsItWas:
    def test_layers_end_up_on_disk_after_recording(self, disk, looper):
        _record(looper, 0.25)
        _record(looper, 0.5)

        assert disk.wait_for_saves(2)
        saved = [np.load(disk.folder / name) for name in disk.files()]
        assert sorted(float(layer[0, 0]) for layer in saved) == [0.25, 0.5]

    def test_undo_takes_back_a_layer_that_is_still_being_saved(
        self, disk, looper
    ):
        _record(looper, 0.25)
        assert disk.wait_for_saves(1)
        disk.saving_allowed.clear()
        _record(looper, 0.5)

        undone = looper.undo_layer()
        heard = _play(looper)
        disk.saving_allowed.set()

        assert undone and looper.layer_count == 1
        assert np.allclose(heard, 0.25)

    def test_a_layer_recorded_after_an_undo_is_not_mixed_up_with_the_old_one(
        self, disk, looper
    ):
        _record(looper, 0.25)
        disk.saving_allowed.clear()
        _record(looper, 0.5)
        looper.undo_layer()
        _record(looper, 0.125)
        disk.saving_allowed.set()
        _record(looper, 0.0)
        looper.undo_layer()

        assert np.allclose(_play(looper), 0.375)

    def test_closing_the_looper_leaves_no_layer_files_behind(
        self, disk, looper
    ):
        _record(looper, 0.25)
        disk.saving_allowed.clear()
        _record(looper, 0.5)
        looper.undo_layer()
        disk.saving_allowed.set()

        looper.close()

        assert not os.path.exists(disk.folder)
        assert not looper.playing
