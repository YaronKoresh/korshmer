import pytest

from korshmer.utils.perf_metrics import PerfMetrics

NAN = float("nan")


@pytest.mark.parametrize("window", [0, -1, 2.5, "5", None, True, NAN])
def test_window_must_be_a_positive_whole_number(window):
    with pytest.raises(ValueError, match="window"):
        PerfMetrics(window)


def test_valid_window_is_kept():
    metrics = PerfMetrics(4)
    for i in range(10):
        metrics.record("a", 0.001 * i)
    assert len(metrics._stats["a"].recent) == 4


@pytest.mark.parametrize("seconds", [NAN, float("inf"), "1", None, True])
def test_record_rejects_a_duration_that_is_not_a_finite_number(seconds):
    metrics = PerfMetrics()
    with pytest.raises(ValueError, match="seconds"):
        metrics.record("a", seconds)
    assert "a" not in metrics._stats


def test_record_clamps_a_negative_duration_to_zero():
    metrics = PerfMetrics()
    metrics.record("a", -1.0)
    assert metrics._stats["a"].as_dict()["max_ms"] == 0.0


@pytest.mark.parametrize("name", [None, 5, "", ["a"]])
def test_record_needs_a_text_name(name):
    metrics = PerfMetrics()
    with pytest.raises(ValueError, match="name"):
        metrics.record(name, 0.1)


def test_a_rejected_value_does_not_poison_later_statistics():
    metrics = PerfMetrics()
    with pytest.raises(ValueError):
        metrics.record("a", NAN)
    metrics.record("a", 1.0)
    stats = metrics._stats["a"].as_dict()
    assert stats["count"] == 1 and stats["total_ms"] == 1000.0
