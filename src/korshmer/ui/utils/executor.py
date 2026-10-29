import atexit
import concurrent.futures
import logging
import threading

from korshmer.utils.tracing import propagate
from korshmer.utils.worker_count import compute_worker_count

_executor_lock = threading.Lock()
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


def run_async(func, *args, **kwargs):
    executor = get_executor()
    with _executor_lock:
        if _executor_shutdown:
            raise RuntimeError("Cannot submit tasks during shutdown")
        _active_futures[:] = [f for f in _active_futures if not f.done()]
        future = executor.submit(propagate(func), *args, **kwargs)
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


def shutdown_executor():
    global _executor, _executor_shutdown
    with _executor_lock:
        _executor_shutdown = True
        futures_to_cancel = list(_active_futures)
        _active_futures.clear()

    if _executor is not None:
        logging.debug("Shutting down thread pool executor")
        try:
            for future in futures_to_cancel:
                if not future.done():
                    future.cancel()
            _executor.shutdown(wait=True)
        except Exception as e:
            logging.error(f"Error during executor shutdown: {e}")
        finally:
            _executor = None


atexit.register(shutdown_executor)
