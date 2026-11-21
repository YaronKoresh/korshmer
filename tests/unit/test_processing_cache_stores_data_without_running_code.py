import os
import pickle

import numpy as np
import pytest

from korshmer.processing.cache import ProcessingCache


@pytest.fixture
def cache(tmp_path):
    return ProcessingCache(tmp_path / "cache")


class RunsOnLoad:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.mkdir, (self.marker,))


def test_a_planted_cache_file_cannot_run_code_when_read(cache, tmp_path):
    marker = tmp_path / "ran"
    key = cache._make_key("a")
    cache.set(key, [1])
    for path in cache.cache_dir.glob("??/*"):
        path.write_bytes(pickle.dumps(RunsOnLoad(str(marker))))
    assert cache.get(key) is None
    assert not marker.exists()


def test_entries_are_not_written_in_the_pickle_format(cache):
    cache.set(cache._make_key("a"), {"x": [1, 2, 3]})
    for path in cache.cache_dir.glob("??/*"):
        assert not path.read_bytes().startswith(b"\x80")


def test_sound_data_comes_back_unchanged_as_a_safeguard(cache):
    key = cache._make_key("audio")
    stereo = np.linspace(-1, 1, 2000, dtype=np.float32).reshape(2, 1000)
    assert cache.set(key, stereo) is True
    restored = cache.get(key)
    assert restored.dtype == np.float32
    assert np.array_equal(restored, stereo)


def test_mixed_results_keep_their_shape_and_types_as_a_safeguard(cache):
    key = cache._make_key("mixed")
    value = {
        "tempo": 123.4,
        "beats": np.array([0.5, 1.0, 1.5]),
        "pair": (np.arange(3), 44100),
        "names": ["kick", "snare"],
        "level": np.float32(0.25),
        "nothing": None,
        "ok": True,
        "array": 1,
        "tuple": [2],
    }
    assert cache.set(key, value) is True
    restored = cache.get(key)
    assert restored["tempo"] == 123.4
    assert np.array_equal(restored["beats"], value["beats"])
    assert isinstance(restored["pair"], tuple)
    assert np.array_equal(restored["pair"][0], np.arange(3))
    assert restored["pair"][1] == 44100
    assert restored["names"] == ["kick", "snare"]
    assert restored["level"] == np.float32(0.25)
    assert restored["nothing"] is None and restored["ok"] is True
    assert restored["array"] == 1 and restored["tuple"] == [2]


def test_values_that_are_not_plain_data_are_not_stored(cache):
    key = cache._make_key("object")
    assert cache.set(key, np.array([object()], dtype=object)) is False
    assert cache.set(key, {1: "number keys"}) is False
    assert cache.set(key, {"when": RunsOnLoad("x")}) is False
    assert list(cache.cache_dir.glob("??/*")) == []


def test_files_left_by_the_old_format_are_removed_on_start(tmp_path):
    folder = tmp_path / "cache" / "ab"
    folder.mkdir(parents=True)
    old = folder / ("ab" + "0" * 62 + ".pkl")
    old.write_bytes(pickle.dumps([1, 2, 3]))
    cache = ProcessingCache(tmp_path / "cache")
    assert cache.get(old.stem) is None
    assert not old.exists()
    assert cache.stats()["entry_count"] == 0
