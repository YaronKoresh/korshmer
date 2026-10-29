from korshmer.midi_processing.midi_file_enhancement import CuePointManager


class FakeEngine:
    def __init__(self):
        self.ms = 0

    def setPosition(self, ms):
        self.ms = ms

    def position(self):
        return self.ms


def test_jump_to_cue_sets_position_in_milliseconds():
    manager = CuePointManager()
    engine = FakeEngine()
    manager.add_cue_point(12.5, "verse")
    assert manager.jump_to_cue(0, engine) is True
    assert engine.position() == 12500


def test_jump_to_missing_cue_leaves_position_unchanged():
    manager = CuePointManager()
    engine = FakeEngine()
    engine.setPosition(300)
    manager.add_cue_point(1.0, "intro")
    assert manager.jump_to_cue(5, engine) is False
    assert manager.jump_to_cue(-1, engine) is False
    assert engine.position() == 300
