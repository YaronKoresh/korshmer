from PySide6.QtWidgets import QLineEdit

from korshmer.midi_generation import seeds

SEED_HELP = (
    "Type a number of any length or any word or phrase. The same seed "
    "always gives the same song. Leave it empty (or 0) for a fresh "
    "random seed every time."
)


class SeedBox(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Random")
        self.setMaxLength(seeds.MAX_SEED_TEXT)
        self.setClearButtonEnabled(True)
        self.setToolTip(SEED_HELP)
        self.setAccessibleName("Seed of the generated song")

    def seed(self):
        return seeds.seed_number(self.text())

    def value(self):
        return self.seed() or 0

    def setValue(self, value):
        if value is None or value == 0 or value == "":
            self.clear()
        else:
            self.setText(str(value))

    def label(self):
        return seeds.seed_label(self.text())
