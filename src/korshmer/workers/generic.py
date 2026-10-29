import concurrent.futures
import logging
import os
import threading

from korshmer.constants import current_dir
from korshmer.utils.worker_count import compute_worker_count

_executor_lock = threading.RLock()
_executor = None
_executor_shutdown = False
_active_futures = []


def get_executor():
    global _executor
    with _executor_lock:
        if _executor is None:
            max_workers = compute_worker_count("io")
            _executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="korshmer"
            )
    return _executor


def run_async(func, *args, cancel_token=None, **kwargs):
    if cancel_token is not None:
        original = func

        def func(*a, **kw):
            if cancel_token.is_set():
                return None
            return original(*a, **kw)

    with _executor_lock:
        if _executor_shutdown:
            raise RuntimeError("Cannot submit tasks during shutdown")
        _active_futures[:] = [f for f in _active_futures if not f.done()]
        future = get_executor().submit(func, *args, **kwargs)
        _active_futures.append(future)
    return future


def wait_all(*futures, timeout=None, raise_on_error=False):
    results = []
    first_exception = None
    for f in futures:
        try:
            res = f.result(timeout=timeout)
            if res is not None:
                results.append(res)
        except concurrent.futures.CancelledError:
            continue
        except Exception as e:
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.exception("Error in async task")
            else:
                logging.error(f"Error in async task: {e}")
            if first_exception is None:
                first_exception = e
    if raise_on_error and first_exception:
        raise first_exception
    return results


try:
    from PySide6.QtCore import Signal

    from korshmer.workers.base import TaskWorker

    class WorkerThread(TaskWorker):
        progress_update = Signal(int, str)
        process_finished = Signal(list)
        process_error = Signal(str)

        def __init__(self, file_paths, params, process_track_func=None):
            super().__init__()
            self.file_paths = file_paths
            self.params = params
            self.process_track_func = process_track_func
            self.cancel_token = threading.Event()
            self._futures = []

        def _process_single_file(self, idx, audio_path, total_files):
            if not self.is_running or self.cancel_token.is_set():
                return None
            file_name = os.path.basename(audio_path)

            def callback(val, msg):
                if not self.is_running:
                    return
                current_file_percent = int(val * 100)
                status_msg = f"[{idx + 1}/{total_files}] {file_name}: {msg}"
                self.progress_update.emit(current_file_percent, status_msg)

            def check_stop():
                return not self.is_running or self.cancel_token.is_set()

            try:
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"File not found: {audio_path}")

                if self.process_track_func is None:
                    raise RuntimeError(
                        "process_track function not provided to WorkerThread"
                    )

                option_kwargs = {
                    key: self.params[key]
                    for key in ("music_options", "separation_options")
                    if self.params.get(key) is not None
                }
                return self.process_track_func(
                    audio_path,
                    self.params["octave"],
                    self.params["root"],
                    self.params["scale"],
                    self.params["grid"],
                    self.params["quantize"],
                    self.params["swing"],
                    self.params["vel_mode"],
                    self.params["vel_val"],
                    self.params["stems"],
                    self.params["export"],
                    self.params["gm_drums"],
                    self.params["cc_gen"],
                    self.params["wav32"],
                    progress_callback=callback,
                    check_stop_callback=check_stop,
                    **option_kwargs,
                )
            except Exception as e:
                error_msg = f"Error processing {file_name}: {str(e)}"
                self.process_error.emit(error_msg)
                return None

        def execute(self):
            total_generated = []
            total_files = len(self.file_paths)

            futures = [
                run_async(
                    self._process_single_file,
                    i,
                    path,
                    total_files,
                    cancel_token=self.cancel_token,
                )
                for i, path in enumerate(self.file_paths)
            ]
            self._futures = futures
            if self.cancel_token.is_set():
                for f in futures:
                    f.cancel()

            results = wait_all(*futures)

            for res in results:
                if res:
                    if isinstance(res, list):
                        total_generated.extend(res)
                    elif isinstance(res, str):
                        total_generated.append(res)

            self.process_finished.emit(total_generated)
            return total_generated

        def stop(self):
            super().stop()
            self.cancel_token.set()
            for f in list(self._futures):
                f.cancel()
            self.progress_update.emit(0, "Stopping process... please wait")

except ImportError:

    class WorkerThread:
        def __init__(self, file_paths, params, process_track_func=None):
            self.file_paths = file_paths
            self.params = params
            self.is_running = True
            self.process_track_func = process_track_func
            self.cancel_token = threading.Event()

        def stop(self):
            self.is_running = False
            self.cancel_token.set()


PRESETS_FILE = os.path.abspath(os.path.join(current_dir, "..", "presets.json"))
