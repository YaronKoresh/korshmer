from __future__ import annotations

import inspect
import unittest.mock as mock


def test_run_async_prunes_done_futures():
    import korshmer.workers.generic as mod

    old_executor = mod._executor
    old_shutdown = mod._executor_shutdown
    old_futures = list(mod._active_futures)

    try:
        mod._executor_shutdown = False
        mod._active_futures.clear()

        fake_executor = mock.MagicMock()
        pending_future = mock.MagicMock()
        pending_future.done.return_value = False
        done_future = mock.MagicMock()
        done_future.done.return_value = True

        mod._active_futures.extend([done_future, pending_future])

        new_future = mock.MagicMock()
        fake_executor.submit.return_value = new_future
        mod._executor = fake_executor

        result = mod.run_async(lambda: None)

        assert result is new_future
        assert done_future not in mod._active_futures
        assert pending_future in mod._active_futures
        assert new_future in mod._active_futures
        assert len(mod._active_futures) == 2
    finally:
        mod._executor = old_executor
        mod._executor_shutdown = old_shutdown
        mod._active_futures[:] = old_futures


def test_run_async_source_has_cleanup():
    import korshmer.workers.generic as mod

    source = inspect.getsource(mod.run_async)
    assert "_active_futures[:] =" in source or "_active_futures[:] =" in source


def test_session_controller_argument_order():
    source = inspect.getsource(
        __import__(
            "korshmer.app.session", fromlist=["SessionController"]
        ).SessionController.start_conversion
    )
    idx_file = source.index("file_paths")
    idx_params = source.index("params", idx_file)
    idx_func = source.index("process_track_func", idx_params)
    assert idx_file < idx_params < idx_func
