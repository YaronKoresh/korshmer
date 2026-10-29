import atexit
import concurrent.futures
import logging
import sys
import threading

from korshmer.utils.tracing import propagate
from korshmer.utils.worker_count import compute_worker_count

_executor_lock = threading.Lock()
_executors: dict[str, concurrent.futures.ThreadPoolExecutor] = {}
_executor_shutdown = False
_active_futures: dict[str, list] = {}


def _module_name_of(frame) -> str:
    return frame.f_globals.get("__name__", "__main__")


def _get_executor_unlocked(module: str):
    global _executor_shutdown
    executor = _executors.get(module)
    if executor is None:
        _executor_shutdown = False
        max_workers = compute_worker_count("io")
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=f"korshmer-{module}"
        )
        _executors[module] = executor
    return executor


def get_executor(module: str | None = None):
    if module is None:
        module = _module_name_of(sys._getframe(1))
    with _executor_lock:
        return _get_executor_unlocked(module)


def run_async(func, *args, **kwargs):
    module = _module_name_of(sys._getframe(1))
    with _executor_lock:
        if _executor_shutdown:
            raise RuntimeError("Cannot submit tasks during shutdown")
        future = _get_executor_unlocked(module).submit(
            propagate(func), *args, **kwargs
        )
        _active_futures.setdefault(module, []).append(future)
    return future


def wait_all(*futures, timeout=None, raise_on_error=False):
    results = []
    first_exception = None
    for f in futures:
        try:
            res = f.result(timeout=timeout)
            if res is not None:
                results.append(res)
        except (TimeoutError, concurrent.futures.TimeoutError):
            raise
        except Exception as e:
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.exception("Error in async task")
            else:
                logging.error(f"Error in async task: {e}")
            if first_exception is None:
                first_exception = e

    cleanup_completed_futures()

    if raise_on_error and first_exception:
        raise first_exception
    return results


def cleanup_completed_futures():
    with _executor_lock:
        cleaned_count = 0

        for module in list(_active_futures):
            new_active_futures = []
            for future in _active_futures[module]:
                if future.cancelled():
                    cleaned_count += 1
                elif future.done():
                    cleaned_count += 1

                    try:
                        future.result(timeout=0)
                    except Exception as e:
                        logging.error(f"Exception in completed future: {e}")
                else:
                    new_active_futures.append(future)

            if new_active_futures:
                _active_futures[module] = new_active_futures
            else:
                del _active_futures[module]

        if cleaned_count > 0:
            logging.debug(
                f"Cleaned up {cleaned_count} completed/cancelled futures"
            )

        return cleaned_count


def shutdown_executor():
    global _executor_shutdown
    with _executor_lock:
        _executor_shutdown = True
        futures_to_cancel = [
            f for futures in _active_futures.values() for f in futures
        ]
        _active_futures.clear()
        executors = list(_executors.values())
        _executors.clear()
    if executors:
        logging.debug("Shutting down thread pool executors")
    for future in futures_to_cancel:
        if not future.done():
            future.cancel()
    for executor in executors:
        try:
            executor.shutdown(wait=True)
        except Exception as e:
            logging.error(f"Error during executor shutdown: {e}")


atexit.register(shutdown_executor)
