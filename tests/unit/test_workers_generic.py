import concurrent.futures


class TestWorkerModuleFunctions:
    def test_get_executor_import(self):
        from korshmer.workers.generic import get_executor

        assert callable(get_executor)

    def test_run_async_import(self):
        from korshmer.workers.generic import run_async

        assert callable(run_async)

    def test_wait_all_import(self):
        from korshmer.workers.generic import wait_all

        assert callable(wait_all)

    def test_get_executor_returns_executor(self):
        from korshmer.workers.generic import get_executor

        executor = get_executor()
        assert isinstance(executor, concurrent.futures.ThreadPoolExecutor)

    def test_get_executor_returns_same_instance(self):
        from korshmer.workers.generic import get_executor

        executor1 = get_executor()
        executor2 = get_executor()
        assert executor1 is executor2


class TestWorkerThread:
    def test_worker_thread_import(self):
        from korshmer.workers.generic import WorkerThread

        assert WorkerThread is not None

    def test_worker_thread_initialization(self):
        from korshmer.workers.generic import WorkerThread

        file_paths = ["/path/to/file1.wav", "/path/to/file2.wav"]
        params = {"octave": "Original", "root": "C", "scale": "Major"}

        worker = WorkerThread(file_paths, params)
        assert worker.file_paths == file_paths
        assert worker.params == params
        assert worker.is_running is True

    def test_worker_thread_stop(self):
        from korshmer.workers.generic import WorkerThread

        worker = WorkerThread([], {})
        worker.stop()
        assert worker.is_running is False


class TestCancellationToken:
    def test_run_async_returns_without_deadlock(self):
        from korshmer.workers.generic import run_async

        assert run_async(lambda: 1).result(timeout=2) == 1

    def test_run_async_skips_task_when_token_set(self):
        import threading

        from korshmer.workers.generic import run_async

        token = threading.Event()
        token.set()
        assert (
            run_async(lambda: 1, cancel_token=token).result(timeout=2) is None
        )

    def test_wait_all_ignores_cancelled_future(self):
        from korshmer.workers.generic import wait_all

        f = concurrent.futures.Future()
        f.cancel()
        assert wait_all(f) == []

    def test_stop_cancels_pending_tasks_and_keeps_finished_results(self):
        import threading
        import time

        pytest = __import__("pytest")
        pytest.importorskip("PySide6")
        from korshmer.workers import generic

        generic.get_executor()
        n = generic.get_executor()._max_workers
        total = n + 4
        started = threading.Event()

        def slow(path, *args, progress_callback=None, check_stop_callback=None):
            started.set()
            time.sleep(0.5)
            return [path]

        params = dict.fromkeys(
            "octave root scale grid quantize swing vel_mode vel_val "
            "stems export gm_drums cc_gen wav32".split()
        )
        worker = generic.WorkerThread(
            [__file__] * total, params, process_track_func=slow
        )
        out = {}
        t = threading.Thread(target=lambda: out.update(r=worker.execute()))
        t0 = time.monotonic()
        t.start()
        assert started.wait(2)
        worker.stop()
        t.join(5)
        assert not t.is_alive()
        assert time.monotonic() - t0 < 1.0 + 0.5
        assert 0 < len(out["r"]) <= n
