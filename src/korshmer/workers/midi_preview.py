import gc

from PySide6.QtCore import Signal

from korshmer.workers.base import TaskWorker


class MidiPreviewThread(TaskWorker):
    preview_ready = Signal(str)
    preview_failed = Signal(str)

    def __init__(
        self,
        midi_path,
        fluid_dir,
        sf3_path,
        output_path,
        bpm=120.0,
        render_func=None,
    ):
        super().__init__()
        self.midi_path = midi_path
        self.fluid_dir = fluid_dir
        self.sf3_path = sf3_path
        self.output_path = output_path
        self.bpm = bpm
        self.render_func = render_func

    def execute(self):
        try:
            render_func = self.render_func
            if render_func is None:
                from korshmer.audio_processors import render_midi_to_wav

                render_func = render_midi_to_wav

            res = render_func(
                self.midi_path,
                self.output_path,
                self.fluid_dir,
                self.sf3_path,
                bpm=self.bpm,
                stop_callback=lambda: not self.is_running,
            )
            if res:
                self.preview_ready.emit(self.output_path)
            else:
                self.preview_failed.emit(
                    "Preview generation failed: the MIDI could not be "
                    "rendered (check that the SoundFont file is present "
                    "and valid)."
                )
            return res
        except Exception as e:
            self.preview_failed.emit(str(e))
            raise
        finally:
            gc.collect()
