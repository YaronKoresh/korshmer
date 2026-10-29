import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QCheckBox, QWidget

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.ui.widgets.material.adaptive import FlowLayout, make_scrollable
from korshmer.ui.widgets.material.card import CARD_MIN_HEIGHT, MaterialCard


def _flow(count):
    host = QWidget()
    layout = FlowLayout(host, spacing=4)
    boxes = [QCheckBox(f"Option number {i}", host) for i in range(count)]
    for box in boxes:
        layout.addWidget(box)
    return host, layout, boxes


def test_flow_layout_wraps_when_narrow():
    host, layout, boxes = _flow(6)
    one_row = layout.heightForWidth(2000)
    narrow = layout.heightForWidth(boxes[0].sizeHint().width() + 10)
    assert narrow > one_row
    host.resize(boxes[0].sizeHint().width() + 10, narrow)
    layout.setGeometry(host.rect())
    ys = {box.y() for box in boxes}
    assert len(ys) == 6


def test_flow_layout_fits_in_one_row_when_wide():
    host, layout, boxes = _flow(3)
    width = sum(b.sizeHint().width() + 4 for b in boxes) + 20
    host.resize(width, 100)
    layout.setGeometry(host.rect())
    assert len({box.y() for box in boxes}) == 1


def test_flow_layout_minimum_is_the_widest_item():
    _, layout, boxes = _flow(4)
    widest = max(b.minimumSizeHint().width() for b in boxes)
    assert layout.minimumSize().width() == widest


def test_scrollable_never_scrolls_sideways_and_keeps_content_size():
    page = QWidget()
    scroll = make_scrollable(page)
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert scroll.widgetResizable()
    assert scroll.widget() is page


def test_card_minimum_height_follows_its_content():
    card = MaterialCard()
    assert card.minimumHeight() == 0
    assert card.minimumSizeHint().height() >= CARD_MIN_HEIGHT
