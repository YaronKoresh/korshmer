import enum
import heapq
import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from korshmer.utils.tracing import propagate
from korshmer.utils.tracked_lock import TrackedLock
from korshmer.utils.worker_count import compute_worker_count

_logger = logging.getLogger(__name__)


class TaskPriority(enum.IntEnum):
    REALTIME = 0
    INTERACTIVE = 1
    BATCH = 2


class TaskStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(order=True)
class QueuedTask:
    priority: int
    sequence: int = field(compare=True)
    task_id: str = field(
        default_factory=lambda: str(uuid.uuid4()), compare=False
    )
    fn: Callable = field(default=None, compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    future: Future = field(default=None, compare=False)
    cancel_token: threading.Event = field(
        default_factory=threading.Event, compare=False
    )
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    submitted_at: float = field(default_factory=time.monotonic, compare=False)


class ProcessingQueue:
    def __init__(self, max_workers: int = None):
        if max_workers is None:
            max_workers = compute_worker_count("compute")
        else:
            max_workers = max(1, max_workers)

        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._heap = []
        self._heap_lock = TrackedLock("ProcessingQueue.heap")
        self._sequence_counter = 0
        self._tasks_dict: dict[str, QueuedTask] = {}
        self._tasks_lock = TrackedLock("ProcessingQueue.tasks")
        self._running_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._stats_lock = TrackedLock("ProcessingQueue.stats")
        self._shutdown = False
        self._wake = threading.Event()

        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop, daemon=True
        )
        self._dispatcher_thread.start()

    def submit(
        self, fn, *args, priority=TaskPriority.INTERACTIVE, **kwargs
    ) -> QueuedTask:
        if self._shutdown:
            raise RuntimeError("Cannot submit tasks after shutdown")

        task = QueuedTask(
            priority=priority,
            sequence=0,
            fn=propagate(fn),
            args=args,
            kwargs=kwargs,
            future=Future(),
        )
        with self._tasks_lock:
            self._tasks_dict[task.task_id] = task
        with self._heap_lock:
            task.sequence = self._sequence_counter
            self._sequence_counter += 1
            heapq.heappush(self._heap, task)

        self._wake.set()
        return task

    def _cancel_pending(self, task: QueuedTask) -> bool:
        if task.status != TaskStatus.PENDING or not task.future.cancel():
            return False
        task.cancel_token.set()
        task.status = TaskStatus.CANCELLED
        return True

    def cancel(self, task_id: str) -> bool:
        with self._tasks_lock:
            task = self._tasks_dict.get(task_id)

        if task is None:
            return False

        return self._cancel_pending(task)

    def drain(self, timeout: float = 30.0):
        start_time = time.monotonic()
        while True:
            with self._stats_lock:
                running = self._running_count
                with self._heap_lock:
                    pending = len(self._heap)

            if running == 0 and pending == 0:
                break

            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                _logger.warning(
                    f"Drain timeout after {elapsed:.2f}s with {running} "
                    f"tasks still running and {pending} pending"
                )
                break

            time.sleep(0.01)

    def shutdown(self):
        with self._tasks_lock:
            tasks = list(self._tasks_dict.values())
        for task in tasks:
            self._cancel_pending(task)

        self.drain(timeout=30.0)
        self._shutdown = True
        self._wake.set()
        self._dispatcher_thread.join(timeout=5.0)
        self._executor.shutdown(wait=True)

    def stats(self) -> dict:
        with self._heap_lock:
            pending = len(self._heap)

        with self._stats_lock:
            running = self._running_count
            completed = self._completed_count
            failed = self._failed_count

        return {
            "pending_count": pending,
            "running_count": running,
            "completed_count": completed,
            "failed_count": failed,
        }

    def _forget(self, task: QueuedTask) -> None:
        with self._tasks_lock:
            self._tasks_dict.pop(task.task_id, None)

    def _next_task(self) -> QueuedTask | None:
        with self._stats_lock:
            while self._running_count < self.max_workers:
                with self._heap_lock:
                    if not self._heap:
                        return None
                    task = heapq.heappop(self._heap)
                if not task.future.set_running_or_notify_cancel():
                    task.status = TaskStatus.CANCELLED
                    self._forget(task)
                    continue
                self._running_count += 1
                return task
        return None

    def _run_task(self, t: QueuedTask) -> None:
        try:
            result = t.fn(*t.args, **t.kwargs)
        except BaseException as e:
            t.status = TaskStatus.FAILED
            with self._stats_lock:
                self._failed_count += 1
            t.future.set_exception(e)
        else:
            t.status = TaskStatus.DONE
            with self._stats_lock:
                self._completed_count += 1
            t.future.set_result(result)
        finally:
            self._forget(t)
            with self._stats_lock:
                self._running_count -= 1
            self._wake.set()

    def _dispatcher_loop(self):
        while not self._shutdown:
            self._wake.clear()
            task = self._next_task()
            if task is None:
                self._wake.wait(timeout=0.1)
                continue
            task.status = TaskStatus.RUNNING
            self._executor.submit(self._run_task, task)


_default_queue: ProcessingQueue | None = None


def get_queue() -> ProcessingQueue:
    global _default_queue
    if _default_queue is None:
        _default_queue = ProcessingQueue()
    return _default_queue


def submit(fn, *args, **kwargs):
    return get_queue().submit(fn, *args, **kwargs)
