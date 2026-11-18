from collections.abc import Callable
from typing import Protocol, runtime_checkable

from korshmer.core.di_container import DIContainer as Container
from korshmer.event_bus import Event, EventBus

WorkerThread: Callable[..., object] | None
try:
    from korshmer.workers.generic import WorkerThread

    _WORKER_AVAILABLE = True
except ImportError:
    WorkerThread = None
    _WORKER_AVAILABLE = False


class _WorkerSignal(Protocol):
    def connect(self, slot: Callable[[], object]) -> object: ...


@runtime_checkable
class _ConversionWorker(Protocol):
    @property
    def started(self) -> _WorkerSignal: ...

    @property
    def finished(self) -> _WorkerSignal: ...

    def start(self) -> object: ...

    def requestInterruption(self) -> object: ...

    def wait(self) -> object: ...

    def isRunning(self) -> bool: ...


class SessionController:
    def __init__(self, container: Container) -> None:
        self.container = container
        self._current_worker: _ConversionWorker | None = None
        self._event_bus = EventBus()

    def start_conversion(
        self,
        file_paths: list[str],
        params: dict[str, object],
        process_track_func: Callable[..., object],
    ) -> _ConversionWorker | None:
        if self._current_worker is not None:
            return None

        if not _WORKER_AVAILABLE or WorkerThread is None:
            return None

        worker = WorkerThread(file_paths, params, process_track_func)
        if not isinstance(worker, _ConversionWorker):
            return None

        self._current_worker = worker
        worker.started.connect(
            lambda: self._event_bus.publish(Event.PROCESS_STARTED)
        )
        worker.finished.connect(
            lambda: self._event_bus.publish(Event.PROCESS_FINISHED)
        )
        worker.start()

        return worker

    def stop_conversion(self, worker: _ConversionWorker | None) -> None:
        if worker is not None:
            worker.requestInterruption()
            worker.wait()
            self._current_worker = None

    @property
    def is_processing(self) -> bool:
        return (
            self._current_worker is not None
            and self._current_worker.isRunning()
        )
