import sys

import pytest
from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline

ACTION_BUTTONS = (
    "btn_clear",
    "btn_merge",
    "btn_sync_play",
    "btn_export_midi",
    "btn_export_mix",
    "btn_export_clip",
)
LOOP_BUTTONS = ("btn_layer", "btn_layer_undo", "btn_loop_clear")


def _shown(width):
    arranger = ArrangerTimeline()
    arranger.resize(width, 600)
    arranger.show()
    QApplication.processEvents()
    return arranger


@pytest.mark.parametrize("width", [620, 850, 1200])
def test_toolbar_buttons_keep_room_for_their_text(width):
    arranger = _shown(width)
    try:
        for name in ACTION_BUTTONS + LOOP_BUTTONS:
            button = getattr(arranger, name)
            assert button.width() >= button.sizeHint().width(), name
    finally:
        arranger.close()


def test_file_actions_follow_the_recording_buttons_in_one_bar():
    arranger = _shown(850)
    try:
        record = arranger.btn_rec.geometry()
        for name in ACTION_BUTTONS:
            spot = getattr(arranger, name).geometry()
            assert (spot.y(), spot.x()) > (record.y(), record.x())
            assert not spot.intersects(record)
            assert spot.right() <= arranger.width()
    finally:
        arranger.close()


@pytest.mark.parametrize("width", [620, 850, 1200])
def test_drop_hint_fits_inside_the_piano_roll(width):
    arranger = _shown(width)
    try:
        label = arranger.lbl_placeholder
        area = arranger.piano_roll.rect()
        assert label.x() >= 0
        assert label.x() + label.width() <= area.width()
        assert label.y() >= 0
        assert label.y() + label.height() <= area.height()
    finally:
        arranger.close()


def test_drop_hint_follows_piano_roll_resize():
    arranger = _shown(1200)
    try:
        wide = arranger.lbl_placeholder.width()
        arranger.piano_roll.resize(300, 300)
        QApplication.processEvents()
        assert arranger.lbl_placeholder.width() < wide
        assert arranger.lbl_placeholder.width() <= 300
    finally:
        arranger.close()


def test_drop_hint_wraps_instead_of_scrolling():
    arranger = _shown(850)
    try:
        label = arranger.lbl_placeholder
        assert label.wordWrap()
        assert label.text().startswith("Drop MIDI or Audio Here")
    finally:
        arranger.close()
