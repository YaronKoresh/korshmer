import threading

from korshmer.project_management.project_manager import ReadWriteLock


def writer_gets_in_within(lock, seconds):
    entered = threading.Event()

    def write():
        lock.acquire_write()
        entered.set()
        lock.release_write()

    thread = threading.Thread(target=write, daemon=True)
    thread.start()
    return entered.wait(seconds), thread


def read_inside_write(lock):
    lock.acquire_write()
    lock.acquire_read()
    lock.release_read()
    lock.release_write()


def test_a_reader_still_keeps_writers_out_after_a_read_inside_a_write():
    lock = ReadWriteLock()
    read_inside_write(lock)
    lock.acquire_read()
    entered_early, thread = writer_gets_in_within(lock, 0.3)
    lock.release_read()
    thread.join(5)
    assert not entered_early
    assert not thread.is_alive()
