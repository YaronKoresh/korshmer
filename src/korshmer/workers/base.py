import time
from abc import abstractmethod

from PySide6.QtCore import QThread, Signal


class TaskWorker(QThread):
    finished_signal = Signal(object)
    error_signal = Signal(str)
    progress_signal = Signal(int, str)

    def __init__(self, parent=None, timeout_seconds=None):
        super().__init__(parent)
        self._running = True
        self.timeout_seconds = timeout_seconds
        self._started_at = None

    @property
    def is_running(self):
        return self._running

    @is_running.setter
    def is_running(self, value):
        self._running = value

    def stop(self):
        self._running = False

    def elapsed_seconds(self):
        if self._started_at is None:
            return 0.0
        return time.monotonic() - self._started_at

    def is_timed_out(self):
        if self.timeout_seconds is None or self._started_at is None:
            return False
        return self.elapsed_seconds() >= self.timeout_seconds

    def _timeout_message(self):
        return f"Timed out after {self.timeout_seconds:g} seconds"

    def check_stopped(self):
        if not self._running:
            raise InterruptedError("Stopped by user")
        if self.is_timed_out():
            raise TimeoutError(self._timeout_message())

    def report_progress(self, value, message=""):
        if self._running:
            self.progress_signal.emit(int(value), message)

    def run(self):
        self._started_at = time.monotonic()
        try:
            result = self.execute()
            if self._running:
                self.finished_signal.emit(result)
        except InterruptedError:
            self.error_signal.emit("Cancelled")
        except TimeoutError as e:
            self.error_signal.emit(str(e) or self._timeout_message())
        except Exception as e:
            self.error_signal.emit(str(e))

    @abstractmethod
    def execute(self):
        pass


class CallableWorker(TaskWorker):
    def __init__(
        self, func, *args, parent=None, timeout_seconds=None, **kwargs
    ):
        super().__init__(parent, timeout_seconds=timeout_seconds)
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def execute(self):
        return self._func(*self._args, **self._kwargs)
