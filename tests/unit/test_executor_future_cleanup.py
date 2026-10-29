from __future__ import annotations

from korshmer.ui.utils import executor


def test_completed_futures_are_pruned():
    old_futures = executor._active_futures[:]
    try:
        executor._active_futures.clear()

        for i in range(5):
            f = executor.run_async(lambda x=i: x * 2)
            f.result(timeout=5)

        assert len(executor._active_futures) <= 1, (
            "Completed futures should be pruned on each submit"
        )
    finally:
        executor._active_futures[:] = old_futures
