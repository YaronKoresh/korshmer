import pytest

from korshmer.utils import worker_count
from korshmer.utils.worker_count import compute_worker_count


@pytest.mark.parametrize(
    "kind,cores,expected",
    [
        ("io", 1, 2),
        ("io", 4, 4),
        ("io", 16, 16),
        ("compute", 1, 1),
        ("compute", 4, 3),
        ("synth", 2, 2),
        ("synth", 4, 4),
        ("synth", 16, 4),
    ],
)
def test_counts(kind, cores, expected):
    assert compute_worker_count(kind, cores) == expected


def test_unknown_kind():
    with pytest.raises(ValueError):
        compute_worker_count("gpu", 4)


def test_default_uses_detected_cores(monkeypatch):
    monkeypatch.setattr(worker_count.os, "cpu_count", lambda: None)
    assert compute_worker_count("io") == 2
    assert compute_worker_count("compute") == 1


def test_four_core_machine_matches_every_pool(monkeypatch):
    monkeypatch.setattr(worker_count.os, "cpu_count", lambda: 4)
    from korshmer.processing.queue import ProcessingQueue

    queue = ProcessingQueue()
    try:
        assert queue.max_workers == compute_worker_count("compute") == 3
    finally:
        queue._executor.shutdown(wait=False)


def test_no_independent_formulas_left():
    import pathlib
    import re

    root = pathlib.Path(worker_count.__file__).parents[1]
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if p != pathlib.Path(worker_count.__file__)
        and re.search(
            r"os\.cpu_count\(\)|max_workers=\d", p.read_text(encoding="utf-8")
        )
    ]
    assert offenders == []
