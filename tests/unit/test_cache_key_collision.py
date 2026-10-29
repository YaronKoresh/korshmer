from __future__ import annotations

import os
import tempfile

from korshmer.processing.cache import ProcessingCache


def test_same_prefix_different_content_produces_different_keys():
    cache = ProcessingCache.__new__(ProcessingCache)

    prefix = b"A" * 65536
    with tempfile.TemporaryDirectory() as td:
        f1 = os.path.join(td, "a.wav")
        f2 = os.path.join(td, "b.wav")
        with open(f1, "wb") as f:
            f.write(prefix + b"TAIL_ONE")
        with open(f2, "wb") as f:
            f.write(prefix + b"TAIL_TWO")

        k1 = cache.make_file_key(f1, "v1", {})
        k2 = cache.make_file_key(f2, "v1", {})

    assert k1 != k2, "Files with same 64KB prefix must not share a cache key"


def test_same_file_same_key():
    cache = ProcessingCache.__new__(ProcessingCache)

    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "a.wav")
        with open(f, "wb") as fh:
            fh.write(b"hello")

        k1 = cache.make_file_key(f, "v1", {"x": 1})
        k2 = cache.make_file_key(f, "v1", {"x": 1})

    assert k1 == k2
