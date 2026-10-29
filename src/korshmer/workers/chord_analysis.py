import logging

from korshmer.workers.base import TaskWorker

try:
    from korshmer.analysis import create_chord_sheet
except ImportError:

    def create_chord_sheet(audio_path, output_dir, precision_mode="1 Bar"):
        logging.error(
            "create_chord_sheet not available - main module not imported"
        )
        return None


class ChordAnalysisThread(TaskWorker):
    def __init__(self, audio_path, output_dir, precision_mode="1 Bar"):
        super().__init__()
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.precision_mode = precision_mode

    def execute(self):
        return create_chord_sheet(
            self.audio_path, self.output_dir, self.precision_mode
        )
