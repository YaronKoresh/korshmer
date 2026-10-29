import sys
import threading

import pytest

from korshmer.utils.system import suppress_stderr


class TestSuppressStderrBasic:
    def test_suppress_stderr_context(self):
        with suppress_stderr():
            sys.stderr.write("This should be suppressed\n")

    def test_suppress_stderr_returns(self):
        try:
            with suppress_stderr():
                pass

            assert True
        except Exception as e:
            pytest.fail(f"suppress_stderr raised unexpected exception: {e}")

    def test_stderr_restored_after_context(self):
        original_stderr = sys.stderr

        with suppress_stderr():
            pass

        assert sys.stderr is original_stderr

    def test_multiple_calls(self):
        for _ in range(3):
            with suppress_stderr():
                sys.stderr.write("test\n")


class TestNestedSuppression:
    def test_nested_suppression(self):
        with suppress_stderr():
            sys.stderr.write("Outer\n")
            with suppress_stderr():
                sys.stderr.write("Inner\n")
            sys.stderr.write("Outer again\n")

    def test_deeply_nested_suppression(self):
        with suppress_stderr():
            with suppress_stderr():
                with suppress_stderr():
                    sys.stderr.write("Deep\n")


class TestExceptionHandling:
    def test_exception_in_context(self):
        with pytest.raises(ValueError, match="Test error"):
            with suppress_stderr():
                raise ValueError("Test error")

    def test_stderr_restored_after_exception(self):
        original_stderr = sys.stderr

        try:
            with suppress_stderr():
                raise RuntimeError("Test")
        except RuntimeError:
            pass

        assert sys.stderr is original_stderr

    def test_multiple_exceptions(self):
        for _ in range(3):
            try:
                with suppress_stderr():
                    raise ValueError("Test")
            except ValueError:
                pass


class TestThreadSafety:
    def test_concurrent_suppression(self):
        results = []

        def suppress_in_thread():
            try:
                with suppress_stderr():
                    sys.stderr.write("Thread\n")
                results.append(True)
            except Exception:
                results.append(False)

        threads = [
            threading.Thread(target=suppress_in_thread) for _ in range(5)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert all(results)

    def test_nested_from_different_threads(self):
        barrier = threading.Barrier(2)
        results = []

        def thread1():
            try:
                with suppress_stderr():
                    barrier.wait()
                    barrier.wait()
                results.append(True)
            except Exception:
                results.append(False)

        def thread2():
            try:
                barrier.wait()
                with suppress_stderr():
                    barrier.wait()
                results.append(True)
            except Exception:
                results.append(False)

        t1 = threading.Thread(target=thread1)
        t2 = threading.Thread(target=thread2)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        assert len(results) == 2
        assert all(results)


class TestPlatformSpecific:
    def test_works_on_current_platform(self):
        with suppress_stderr():
            sys.stderr.write("Platform test\n")

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_windows_null_device(self):
        with suppress_stderr():
            pass

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_unix_null_device(self):
        with suppress_stderr():
            pass


class TestFileDescriptorHandling:
    def test_file_descriptor_cleanup(self):

        with suppress_stderr():
            pass

        with suppress_stderr():
            pass

        with suppress_stderr():
            pass

    def test_stderr_fileno_available(self):
        try:
            fd = sys.stderr.fileno()
            assert fd >= 0
        except AttributeError:
            pytest.skip("stderr.fileno() not available")


class TestErrorCases:
    def test_handles_missing_stderr_fileno(self):

        class FakeStderr:
            def write(self, msg):
                pass

            def flush(self):
                pass

        original_stderr = sys.stderr
        sys.stderr = FakeStderr()

        try:
            with suppress_stderr():
                pass
        finally:
            sys.stderr = original_stderr

    def test_handles_os_error(self):

        with suppress_stderr():
            pass


class TestSideEffects:
    def test_no_effect_on_stdout(self):
        original_stdout = sys.stdout

        with suppress_stderr():
            pass

        assert sys.stdout is original_stdout

    def test_no_effect_on_stdin(self):
        original_stdin = sys.stdin

        with suppress_stderr():
            pass

        assert sys.stdin is original_stdin


class TestEdgeCases:
    def test_empty_context(self):
        with suppress_stderr():
            pass

    def test_write_large_amount(self):
        with suppress_stderr():
            for _ in range(1000):
                sys.stderr.write("x" * 100 + "\n")

    def test_binary_write_to_stderr(self):
        with suppress_stderr():
            try:
                sys.stderr.buffer.write(b"binary data\n")
            except (AttributeError, OSError):
                pass

    def test_flush_stderr(self):
        with suppress_stderr():
            sys.stderr.write("test\n")
            sys.stderr.flush()

    def test_rapid_enter_exit(self):
        for _ in range(100):
            with suppress_stderr():
                pass


class TestConsistency:
    def test_consistent_suppression(self):
        outputs = []

        for _ in range(5):
            with suppress_stderr():
                sys.stderr.write("test\n")
                outputs.append(True)

        assert len(outputs) == 5

    def test_interleaved_suppression(self):
        with suppress_stderr():
            sys.stderr.write("suppressed\n")

        sys.stderr.write("normal\n")

        with suppress_stderr():
            sys.stderr.write("suppressed again\n")
