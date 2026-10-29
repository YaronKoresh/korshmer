import threading
from collections import OrderedDict


class SegmentCache:
    def __init__(self, max_bytes):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise ValueError(
                f"Cache size must be an integer byte count, got {max_bytes!r}"
            )
        if max_bytes <= 0:
            raise ValueError(f"Cache size must be positive, got {max_bytes}")
        self._max_bytes = max_bytes
        self._entries = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.Lock()

    @property
    def max_bytes(self):
        return self._max_bytes

    @property
    def size_bytes(self):
        with self._lock:
            return self._total_bytes

    def __len__(self):
        with self._lock:
            return len(self._entries)

    def get(self, key):
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key, value):
        nbytes = int(value.nbytes)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._total_bytes -= int(old.nbytes)
            if nbytes > self._max_bytes:
                return False
            while (
                self._entries and self._total_bytes + nbytes > self._max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._total_bytes -= int(evicted.nbytes)
            self._entries[key] = value
            self._total_bytes += nbytes
            return True

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0
