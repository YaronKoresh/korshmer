import pytest
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog, QLineEdit

from korshmer.utils import text_input
from korshmer.utils.search_input import MAX_SEARCH_QUERY_LENGTH
from korshmer.utils.text_input import (
    MAX_TEXT_INPUT_LENGTH,
    get_limited_text,
    is_within_max_length,
    limit_line_edit,
)


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication([])


def _fake_exec(typed, accept=True):
    def exec_(dialog):
        line_edit = dialog.findChild(QLineEdit)
        line_edit.clear()
        line_edit.insert(typed)
        dialog.setTextValue(line_edit.text())
        return QDialog.Accepted if accept else QDialog.Rejected

    return exec_


def test_limit_value():
    assert MAX_TEXT_INPUT_LENGTH == 4096
    assert MAX_SEARCH_QUERY_LENGTH == MAX_TEXT_INPUT_LENGTH


def test_is_within_max_length():
    assert is_within_max_length("")
    assert is_within_max_length(None)
    assert is_within_max_length("a" * 4096)
    assert not is_within_max_length("a" * 4097)


def test_limit_line_edit_blocks_paste():
    edit = limit_line_edit(QLineEdit())
    assert edit.maxLength() == 4096
    edit.insert("x" * 5000)
    assert len(edit.text()) == 4096


def test_get_limited_text_blocks_long_paste(monkeypatch):
    monkeypatch.setattr(QInputDialog, "exec", _fake_exec("n" * 5000))
    text, ok = get_limited_text(None, "New Project", "Project Name:")
    assert ok is True
    assert len(text) == 4096


@pytest.mark.parametrize("value", ["My Song", "", "שיר"])
def test_get_limited_text_regular_values(monkeypatch, value):
    monkeypatch.setattr(QInputDialog, "exec", _fake_exec(value))
    assert get_limited_text(None, "T", "L") == (value, True)


def test_get_limited_text_cancel(monkeypatch):
    monkeypatch.setattr(QInputDialog, "exec", _fake_exec("abc", accept=False))
    _, ok = get_limited_text(None, "T", "L")
    assert ok is False


def test_overlong_value_rejected_with_warning(monkeypatch):
    def exec_(dialog):
        dialog.setTextValue("z" * 10)
        return QDialog.Accepted

    warnings = []

    monkeypatch.setattr(QInputDialog, "findChild", lambda self, cls: None)
    monkeypatch.setattr(QInputDialog, "exec", exec_)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        lambda *a, **k: warnings.append(a),
    )
    assert get_limited_text(None, "T", "L", max_length=5) == ("", False)
    assert warnings


def test_ui_free_text_entry_points_use_limit():
    import inspect

    from korshmer.mixins import presets_action
    from korshmer.mixins.actions import project_management

    for module in (project_management, presets_action):
        source = inspect.getsource(module)
        assert "QInputDialog.getText" not in source
        assert "get_limited_text(" in source


def test_track_name_edit_limited():
    import pretty_midi

    from korshmer.ui.widgets.arranger.track_panel import TrackPanel
    from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll

    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="Keys")
    inst.notes.append(pretty_midi.Note(100, 60, 0.0, 0.5))
    midi.instruments.append(inst)
    roll = InteractivePianoRoll()
    roll.load_notes(midi)
    panel = TrackPanel(roll.model)
    row = panel._rows[0]
    assert row.name_edit.maxLength() == 4096
    row.name_edit.insert("y" * 5000)
    assert len(row.name_edit.text()) <= 4096


def test_module_has_no_qt_import_at_top():
    import inspect

    source = inspect.getsource(text_input)
    top = source.split("def ")[0]
    assert "PySide6" not in top
