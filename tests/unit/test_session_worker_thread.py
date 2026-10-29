from __future__ import annotations

from unittest.mock import MagicMock, patch

from korshmer.app.session import SessionController


def _make_container():
    return MagicMock()


def test_start_conversion_passes_args_in_correct_order():
    ctrl = SessionController(_make_container())
    file_paths = ["/tmp/a.wav", "/tmp/b.wav"]
    params = {
        "octave": "Original",
        "root": "Auto",
        "scale": "Auto (Detect)",
        "grid": "1/8",
        "quantize": 100,
        "swing": 0,
        "vel_mode": "Fixed",
        "vel_val": 100,
        "stems": [],
        "export": "Separate",
        "gm_drums": True,
        "cc_gen": True,
        "wav32": True,
    }
    process_func = MagicMock()

    with patch("korshmer.app.session.WorkerThread") as MockWorker:
        instance = MagicMock()
        MockWorker.return_value = instance
        with patch("korshmer.app.session._WORKER_AVAILABLE", True):
            ctrl.start_conversion(file_paths, params, process_func)

    MockWorker.assert_called_once_with(file_paths, params, process_func)


def test_start_conversion_returns_none_when_worker_unavailable():
    ctrl = SessionController(_make_container())
    with patch("korshmer.app.session._WORKER_AVAILABLE", False):
        result = ctrl.start_conversion(["/tmp/a.wav"], {}, MagicMock())
    assert result is None
    assert ctrl._current_worker is None
