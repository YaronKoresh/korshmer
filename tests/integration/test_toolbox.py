import os
import sys

import pretty_midi
import pytest
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

if not QApplication.instance():
    QApplication(sys.argv)

try:
    from korshmer.__main__ import MainWindow
except Exception as import_error:
    MainWindow = None
    MAIN_IMPORT_ERROR = str(import_error)
else:
    MAIN_IMPORT_ERROR = ""

from korshmer.ui.widgets.material.button import MaterialButton
from korshmer.ui.widgets.material.mode_tool import ModeToolCard

pytestmark = pytest.mark.integration

DIRECT_HANDLERS = (
    "generate_chords_action",
    "run_piano_roll_img",
    "run_spectrum_action",
    "run_audio_to_midi_conversion",
    "run_stem_separation",
    "open_controller_mapping_dialog",
    "run_range_eq_action",
)

EXPECTED = {
    "Polish MIDI": {"Polish": "process_polish"},
    "Clean up": {
        "Remove short notes": "process_cleanup",
        "Remove empty bars": "process_remove_silence",
        "Merge glitch notes": "process_merge_glitch",
    },
    "Edit": {
        "Transpose": "process_transpose",
        "Snap to scale": "process_scale_snap",
        "Invert chords": "process_inversions",
        "Limit range": "process_limit_range",
        "Remap drums": "process_drum_remap",
        "Create variation": "process_variation",
    },
    "Analyze": {
        "Statistics": "process_stats",
        "Rhythm": "process_rhythm",
        "Count instruments": "_build_instrument_report",
        "Detect chords": "process_chord_detection",
        "Detect scales": "process_scale_detection",
        "Chord analysis (audio)": "generate_chords_action",
        "Piano roll image": "run_piano_roll_img",
        "Frequency spectrum": "run_spectrum_action",
    },
    "Timing": {
        "Smart loop": "process_smart_loop",
        "Speed x2 (faster)": "process_speed",
        "Speed x0.5 (slower)": "process_speed",
        "Groove": "process_groove",
        "Ratchet": "process_ratchet",
        "Advanced quantize": "process_advanced_quantize",
        "Strum chords": "process_strum",
        "MIDI delay": "process_midi_delay",
        "Thin out": "process_density",
    },
    "Arpeggiator": {
        "Arpeggiate": "process_arpeggiator",
        "Advanced arpeggio": "process_arpeggiator_advanced",
    },
    "Voicing": {
        "Harmonize": "process_smart_harmony",
        "Voice lead": "process_voice_leading",
        "Open voicing": "process_open_voice",
        "Drop 2": "process_drop2",
        "Force mono": "process_mono",
        "Split by pitch": "process_split_pitch",
        "Double octave": "process_doubling",
    },
    "Dynamics": {
        "Humanize": "process_humanize",
        "Velocity ramp": "process_vel_ramp",
        "Crescendo": "process_crescendo",
        "Invert velocity": "process_invert_vel",
        "Lo-Fi velocity": "process_lofi",
        "Random velocity": "process_rand_vel",
        "LFO": "process_lfo",
    },
    "Audio effect": {
        "Parametric EQ": "process_parametric_eq",
        "Compressor": "process_compressor",
        "Reverb": "process_reverb",
        "Delay": "process_delay",
        "Chorus": "process_chorus",
        "Limiter": "process_limiter",
        "Filter": "process_audio_filter",
        "Gate": "process_audio_gate",
        "Transient shaper": "process_transient",
    },
    "Audio fix": {
        "Normalize": "process_audio_norm",
        "Pitch shift": "process_audio_pitch",
        "Mono mix": "process_audio_mono",
        "Reverse": "process_audio_reverse",
        "Slice by silence": "process_slicer",
    },
    "Mastering": {
        "Smart mastering": "process_smart_mastering_file",
        "Range EQ": "run_range_eq_action",
    },
}


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    if MainWindow is None:
        pytest.skip(f"MainWindow unavailable here: {MAIN_IMPORT_ERROR}")
    tmp_path = tmp_path_factory.mktemp("toolbox")
    monkeypatch = pytest.MonkeyPatch()
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(
            QMessageBox, name, staticmethod(lambda *a, **k: None)
        )
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    calls = []
    for name in DIRECT_HANDLERS:
        monkeypatch.setattr(
            MainWindow,
            name,
            (lambda nm: lambda self, *a, **k: calls.append((nm, a)))(name),
        )
    win = MainWindow()
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(90, 60, 0, 1))
    pm.instruments.append(inst)
    midi = str(tmp_path / "a.mid")
    pm.write(midi)
    wav = str(tmp_path / "a.wav")
    sf.write(wav, np.zeros((4410, 2), dtype="float32"), 44100)
    win.get_selected_midi = lambda: midi
    win._get_selected_file = lambda is_input=None, requires_midi=False, **k: (
        midi if requires_midi else wav
    )
    win._run_action = lambda func, *a, **k: calls.append((func.__name__, a))
    win._run_in_background = lambda func, *a, **k: (
        calls.append((getattr(func, "__name__", "background"), a)) or True
    )
    win._calls = calls
    yield win
    win.close()
    QApplication.processEvents()
    monkeypatch.undo()


def _cards(window):
    cards = {}
    for i in range(window.list_categories.count()):
        page = window.stacked_content.widget(i)
        for card in page.findChildren(ModeToolCard):
            cards[
                card.title_label.text().replace("<b>", "").replace("</b>", "")
            ] = card
    return cards


def _buttons(window, name):
    for i in range(window.list_categories.count()):
        if window.list_categories.item(i).text() == name:
            page = window.stacked_content.widget(i)
            return page.findChildren(MaterialButton)
    raise AssertionError(name)


def test_categories_are_the_six_merged_groups(window):
    names = [
        window.list_categories.item(i).text()
        for i in range(window.list_categories.count())
    ]
    assert names == [
        "Polish & Analyze",
        "Rhythm & Timing",
        "Harmony & Voicing",
        "MIDI Generator",
        "Dynamics",
        "Audio Tools",
    ]


def test_the_tools_pages_have_far_fewer_buttons(window):
    total = sum(
        len(window.stacked_content.widget(i).findChildren(MaterialButton))
        for i in range(window.stacked_content.count())
    )
    assert total <= 20


def test_each_group_has_a_single_apply_button(window):
    cards = _cards(window)
    assert set(cards) == set(EXPECTED)
    for card in cards.values():
        assert len(card.findChildren(MaterialButton)) == 1


def test_every_old_tool_is_still_a_mode(window):
    cards = _cards(window)
    for title, modes in EXPECTED.items():
        assert cards[title].mode_labels() == list(modes)


@pytest.mark.parametrize("title", list(EXPECTED))
def test_every_mode_runs_its_own_handler(window, title):
    card = _cards(window)[title]
    for label, expected in EXPECTED[title].items():
        window._calls.clear()
        card.set_mode(label)
        card.apply()
        assert [c[0] for c in window._calls][:1] == [expected], label


def test_polish_passes_the_chosen_target_and_length(window):
    window.combo_polish_target.setCurrentText("Lead")
    window.combo_polish_ms.setCurrentText("100ms")
    window._calls.clear()
    _cards(window)["Polish MIDI"].apply()
    name, args = window._calls[0]
    assert name == "process_polish"
    assert args[1:] == ("Lead", 100)


def test_the_polish_target_choices_match_the_pipeline(window):
    from korshmer.midi.polish import TARGETS

    items = [
        window.combo_polish_target.itemText(i)
        for i in range(window.combo_polish_target.count())
    ]
    assert items == list(TARGETS)


def test_variation_passes_the_chosen_strength_and_a_fresh_seed(window):
    from korshmer.midi.variation import STRENGTHS

    items = [
        window.combo_variation_strength.itemText(i)
        for i in range(window.combo_variation_strength.count())
    ]
    assert items == list(STRENGTHS)
    window.combo_variation_strength.setCurrentText("Strong")
    card = _cards(window)["Edit"]
    card.set_mode("Create variation")
    window._calls.clear()
    card.apply()
    name, args = window._calls[0]
    assert name == "process_variation"
    assert args[1] == "Strong"
    assert isinstance(args[2], int)


def test_parameter_fields_follow_the_chosen_mode(window):
    clean = _cards(window)["Clean up"]
    clean.set_mode("Remove short notes")
    assert not window.spin_ms.isHidden()
    clean.set_mode("Merge glitch notes")
    assert window.spin_ms.isHidden()
    effect = _cards(window)["Audio effect"]
    effect.set_mode("Filter")
    assert not window.combo_filt_type.isHidden()
    assert window.spin_gate.isHidden()
    effect.set_mode("Gate")
    assert window.combo_filt_type.isHidden()
    assert not window.spin_gate.isHidden()


def test_model_tools_are_the_only_ai_buttons(window):
    texts = {b.text() for b in _buttons(window, "Audio Tools")}
    assert {"Audio to MIDI", "Stem Separation"} <= texts
    for i in range(window.list_categories.count()):
        for b in window.stacked_content.widget(i).findChildren(MaterialButton):
            assert "AI" not in b.text().split()
