import threading

from korshmer.midi.event_processor import MidiEvent, MidiEventQueue


def test_sorted_events_is_stable_and_ordered_by_time():
    queue = MidiEventQueue()
    queue.push(MidiEvent(0.5, "noteon", 0, 60, 90, 1))
    queue.push(MidiEvent(0.1, "noteon", 0, 61, 90, 2))
    queue.push(MidiEvent(0.5, "noteoff", 0, 61, 0, 2))

    ordered = queue.sorted_events()

    assert [e.time for e in ordered] == [0.1, 0.5, 0.5]
    assert [e.etype for e in ordered] == ["noteon", "noteon", "noteoff"]
    assert len(queue) == 3


def test_concurrent_pushes_are_all_kept():
    queue = MidiEventQueue()
    per_thread = 500

    def push_many(offset):
        for i in range(per_thread):
            queue.push(MidiEvent(float(i), "noteon", offset, i % 128, 1, i))

    threads = [threading.Thread(target=push_many, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    ordered = queue.sorted_events()
    assert len(ordered) == 8 * per_thread
    times = [e.time for e in ordered]
    assert times == sorted(times)


def test_sorted_events_returns_copy():
    queue = MidiEventQueue()
    queue.push(MidiEvent(0.0, "noteon", 0, 60, 90, 1))
    snapshot = queue.sorted_events()
    snapshot.clear()
    assert len(queue) == 1
