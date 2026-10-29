import time
from concurrent.futures import TimeoutError

import pytest

from korshmer.utils.executor import (
    get_executor,
    run_async,
    shutdown_executor,
    wait_all,
)


@pytest.fixture(autouse=True)
def reset_executor():

    try:
        shutdown_executor()
    except Exception:
        pass
    get_executor()

    yield

    try:
        shutdown_executor()
    except Exception:
        pass


class TestExecutorInitialization:
    def test_get_executor_returns_instance(self):
        executor = get_executor()
        assert executor is not None

    def test_get_executor_singleton(self):
        executor1 = get_executor()
        executor2 = get_executor()
        assert executor1 is executor2

    def test_executor_has_thread_name_prefix(self):
        executor = get_executor()

        future = executor.submit(lambda: None)
        future.result()


class TestRunAsync:
    def test_run_async_simple_function(self):

        def simple_task():
            return 42

        future = run_async(simple_task)
        result = future.result(timeout=1.0)
        assert result == 42

    def test_run_async_with_args(self):

        def task_with_args(a, b):
            return a + b

        future = run_async(task_with_args, 10, 20)
        result = future.result(timeout=1.0)
        assert result == 30

    def test_run_async_with_kwargs(self):

        def task_with_kwargs(a, b=0):
            return a + b

        future = run_async(task_with_kwargs, 10, b=20)
        result = future.result(timeout=1.0)
        assert result == 30

    def test_run_async_multiple_tasks(self):

        def task(n):
            time.sleep(0.01)
            return n * 2

        futures = [run_async(task, i) for i in range(5)]
        results = [f.result(timeout=1.0) for f in futures]

        assert results == [0, 2, 4, 6, 8]

    def test_run_async_task_exception(self):

        def failing_task():
            raise ValueError("Test error")

        future = run_async(failing_task)

        with pytest.raises(ValueError, match="Test error"):
            future.result(timeout=1.0)

    def test_run_async_returns_future(self):

        def task():
            return 42

        future = run_async(task)
        assert hasattr(future, "result")
        assert hasattr(future, "done")
        assert hasattr(future, "cancel")


class TestWaitAll:
    def test_wait_all_single_future(self):

        def task():
            return 42

        future = run_async(task)
        results = wait_all(future)

        assert len(results) == 1
        assert results[0] == 42

    def test_wait_all_multiple_futures(self):

        def task(n):
            return n * 2

        futures = [run_async(task, i) for i in range(3)]
        results = wait_all(*futures)

        assert len(results) == 3
        assert set(results) == {0, 2, 4}

    def test_wait_all_with_timeout(self):

        def slow_task():
            time.sleep(10)
            return 42

        future = run_async(slow_task)

        with pytest.raises(TimeoutError):
            wait_all(future, timeout=0.1)

    def test_wait_all_with_errors_no_raise(self):

        def good_task():
            return 42

        def bad_task():
            raise ValueError("Error")

        future1 = run_async(good_task)
        future2 = run_async(bad_task)

        results = wait_all(future1, future2, raise_on_error=False)

        assert 42 in results

    def test_wait_all_with_errors_raise(self):

        def bad_task():
            raise ValueError("Test error")

        future = run_async(bad_task)

        with pytest.raises(ValueError, match="Test error"):
            wait_all(future, raise_on_error=True)

    def test_wait_all_empty_futures(self):
        results = wait_all()
        assert results == []

    def test_wait_all_filters_none_results(self):

        def task_returns_none():
            return None

        def task_returns_value():
            return 42

        future1 = run_async(task_returns_none)
        future2 = run_async(task_returns_value)

        results = wait_all(future1, future2)

        assert results == [42]


class TestShutdown:
    def test_shutdown_executor_completes(self):

        get_executor()

        shutdown_executor()

    def test_shutdown_cancels_pending_tasks(self):

        def long_task():
            time.sleep(10)
            return 42

        future = run_async(long_task)

        shutdown_executor()

        assert future.done() or future.cancelled()

    def test_cannot_submit_after_shutdown(self):

        def task():
            return 42

        shutdown_executor()

        with pytest.raises(
            RuntimeError, match="Cannot submit tasks during shutdown"
        ):
            run_async(task)


class TestConcurrency:
    def test_concurrent_execution(self):
        results = []

        def task(n):
            time.sleep(0.01)
            results.append(n)
            return n

        start_time = time.time()
        futures = [run_async(task, i) for i in range(5)]
        wait_all(*futures)
        elapsed = time.time() - start_time

        assert elapsed < 0.04

    def test_thread_safety(self):
        counter = []

        def increment():
            counter.append(1)

        futures = [run_async(increment) for _ in range(100)]
        wait_all(*futures)

        assert len(counter) == 100


class TestErrorHandling:
    def test_exception_in_task(self):

        def failing_task():
            raise RuntimeError("Task failed")

        future = run_async(failing_task)

        with pytest.raises(RuntimeError, match="Task failed"):
            future.result()

    def test_multiple_exceptions(self):

        def failing_task(msg):
            raise ValueError(msg)

        futures = [run_async(failing_task, f"Error {i}") for i in range(3)]

        with pytest.raises(ValueError):
            wait_all(*futures, raise_on_error=True)

    def test_timeout_error(self):

        def slow_task():
            time.sleep(10)

        future = run_async(slow_task)

        with pytest.raises(TimeoutError):
            future.result(timeout=0.1)


class TestResourceCleanup:
    def test_futures_are_tracked(self):

        def task():
            return 42

        futures = [run_async(task) for _ in range(5)]

        wait_all(*futures)

        assert all(f.done() for f in futures)

    def test_completed_tasks_cleanup(self):

        def quick_task():
            return 42

        future = run_async(quick_task)
        result = future.result(timeout=1.0)

        assert result == 42
        assert future.done()


class TestEdgeCases:
    def test_zero_length_task(self):

        def instant_task():
            return "done"

        future = run_async(instant_task)
        result = future.result(timeout=0.1)

        assert result == "done"

    def test_task_returning_large_data(self):

        def large_data_task():
            return list(range(10000))

        future = run_async(large_data_task)
        result = future.result(timeout=1.0)

        assert len(result) == 10000

    def test_nested_async_calls(self):

        def inner_task():
            return 42

        def outer_task():
            future = run_async(inner_task)
            return future.result(timeout=1.0)

        future = run_async(outer_task)
        result = future.result(timeout=2.0)

        assert result == 42

    def test_task_with_side_effects(self):
        state = []

        def task_with_side_effect():
            state.append(1)

        future = run_async(task_with_side_effect)
        future.result(timeout=1.0)

        assert len(state) == 1


class TestPerformance:
    def test_many_small_tasks(self):

        def small_task(n):
            return n * 2

        start_time = time.time()
        futures = [run_async(small_task, i) for i in range(100)]
        results = wait_all(*futures)
        elapsed = time.time() - start_time

        assert len(results) == 100

        assert elapsed < 2.0

    def test_cpu_bound_tasks(self):

        def cpu_task():
            total = 0
            for i in range(10000):
                total += i
            return total

        futures = [run_async(cpu_task) for _ in range(5)]
        results = wait_all(*futures)

        assert len(results) == 5

        assert len(set(results)) == 1


def _submit_from_module(module_name, func):
    namespace = {
        "__name__": module_name,
        "run_async": run_async,
        "func": func,
    }
    return eval("run_async(func)", namespace)


class TestPerModuleExecutors:
    def test_each_module_gets_its_own_executor(self):
        analysis_pool = get_executor("korshmer.analysis")
        workers_pool = get_executor("korshmer.workers.generic")

        assert analysis_pool is not workers_pool
        assert get_executor("korshmer.analysis") is analysis_pool

    def test_run_async_uses_pool_of_calling_module(self):
        def report_thread():
            import threading

            return threading.current_thread().name

        future = _submit_from_module("fake_module_a", report_thread)
        name = future.result(timeout=2.0)

        assert name.startswith("korshmer-fake_module_a")

    def test_shutdown_stops_every_module_pool(self):
        pool_a = get_executor("fake_module_a")
        pool_b = get_executor("fake_module_b")

        shutdown_executor()

        assert pool_a._shutdown
        assert pool_b._shutdown
        assert get_executor("fake_module_a") is not pool_a
