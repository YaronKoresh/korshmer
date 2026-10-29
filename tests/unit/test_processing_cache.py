import os
import pickle
import threading
import time

import pytest

import korshmer.processing.cache as cache_module
from korshmer.processing.cache import ProcessingCache, cached


@pytest.fixture
def cache(tmp_path):
    return ProcessingCache(tmp_path / "cache")


def _entry_path(cache, key):
    return cache.cache_dir / key[:2] / f"{key}.pkl"


def test_round_trip(cache):
    key = cache._make_key("a")
    assert cache.set(key, {"x": [1, 2, 3]}) is True
    assert cache.get(key) == {"x": [1, 2, 3]}


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        pickle.dumps(list(range(1000)))[:30],
        b"not a pickle",
        b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x08nomodule"
        b"\x94\x8c\x03Foo\x94\x93\x94.",
    ],
)
def test_unreadable_entry_is_a_miss_and_is_removed(cache, payload):
    key = cache._make_key("a")
    cache.set(key, [1])
    path = _entry_path(cache, key)
    path.write_bytes(payload)
    assert cache.get(key) is None
    assert not path.exists()
    assert cache.stats()["entry_count"] == 0


def test_invalidate_removes_file_not_in_index(cache, tmp_path):
    key = cache._make_key("a")
    cache.set(key, [1])
    fresh = ProcessingCache(cache.cache_dir)
    fresh.invalidate(key)
    assert not _entry_path(cache, key).exists()


def test_cached_separates_positional_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cache_module, "_default_cache", ProcessingCache(tmp_path / "cache")
    )
    source = tmp_path / "in.wav"
    source.write_bytes(b"x" * 100)
    calls = []

    @cached("1")
    def run(path, gain, **kwargs):
        calls.append(gain)
        return gain * 10

    assert run(str(source), 1) == 10
    assert run(str(source), 2) == 20
    assert run(str(source), 2) == 20
    assert run(str(source), gain=3) == 30
    assert calls == [1, 2, 3]


def test_entries_survive_a_restart(tmp_path):
    cache_dir = tmp_path / "cache"
    first = ProcessingCache(cache_dir)
    key = first._make_key("a")
    first.set(key, [1, 2, 3])

    second = ProcessingCache(cache_dir)
    assert second.stats()["entry_count"] == 1
    assert second.get(key) == [1, 2, 3]


def test_expired_files_are_removed_on_start(tmp_path):
    cache_dir = tmp_path / "cache"
    first = ProcessingCache(cache_dir, ttl_seconds=60)
    key = first._make_key("a")
    first.set(key, [1])
    path = _entry_path(first, key)
    old = time.time() - 3600
    os.utime(path, (old, old))

    second = ProcessingCache(cache_dir, ttl_seconds=60)
    assert second.get(key) is None
    assert not path.exists()


def test_size_limit_counts_files_from_earlier_runs(tmp_path):
    cache_dir = tmp_path / "cache"
    first = ProcessingCache(cache_dir)
    for i in range(3):
        first.set(first._make_key(i), "x" * 600_000)

    second = ProcessingCache(cache_dir, max_size_mb=1)
    assert second.stats()["total_size_mb"] <= 1
    assert len(list(cache_dir.glob("??/*.pkl"))) == 1


def test_set_of_unpicklable_value_returns_false_and_leaves_no_file(cache):
    key = cache._make_key("lock")
    assert cache.set(key, threading.Lock()) is False
    assert cache.set(key, lambda: 1) is False
    assert list(cache.cache_dir.glob("??/*")) == []
    assert cache.stats()["entry_count"] == 0


def test_failed_set_keeps_the_previous_entry(cache):
    key = cache._make_key("keep")
    assert cache.set(key, {"good": 1}) is True
    assert cache.set(key, lambda: 1) is False
    assert cache.get(key) == {"good": 1}
    assert cache.stats()["entry_count"] == 1


def test_failed_write_leaves_no_temp_file(cache, monkeypatch):
    key = cache._make_key("replace")

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(cache_module.os, "replace", broken_replace)
    assert cache.set(key, [1, 2]) is False
    assert list(cache.cache_dir.glob("??/*")) == []
    assert cache.get(key) is None
