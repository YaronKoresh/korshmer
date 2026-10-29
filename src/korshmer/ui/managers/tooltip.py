class TooltipManager:
    def __init__(self):
        self.tooltips = {}
        self.help_mode = False
        self._init_default_tooltips()

    def _init_default_tooltips(self):
        self.tooltips = {
            "play_pause": "Play/Pause (Space)\nStart or pause playback",
            "stop": "Stop (Escape)\nStop playback and return to start",
            "record": "Record (R)\nStart recording audio",
            "bpm": "Tempo (BPM)\nSet the project tempo in beats per minute",
            "quantize": "Quantize (Q)\nAlign notes to the grid",
            "humanize": "Humanize (U)\nAdd natural timing variations",
            "transpose": "Transpose\nShift pitch up or down",
            "save": "Save (Ctrl+S)\nSave the current project",
            "export": "Export (Ctrl+E)\nExport audio to file",
            "undo": "Undo (Ctrl+Z)\nRevert last action",
            "redo": "Redo (Ctrl+Y)\nReapply undone action",
            "zoom_in": "Zoom In (Ctrl++)\nIncrease zoom level",
            "zoom_out": "Zoom Out (Ctrl+-)\nDecrease zoom level",
            "snap": "Snap (N)\nSnap note edits to the grid",
            "duplicate": "Duplicate (Ctrl+D)\nCopy the selection right after itself",
            "build_song": "Build Song (Ctrl+B)\nGenerate a full arrangement into the editor",
            "audio_clip": "Audio Clip\nDrag to move in time, double-click to mute",
            "lane": "Lane\nClick to make it the target for new notes",
        }

    def get(self, key):
        return self.tooltips.get(key, "")

    def set(self, key, tooltip):
        self.tooltips[key] = tooltip

    def enable_help_mode(self):
        self.help_mode = True

    def disable_help_mode(self):
        self.help_mode = False

    def is_help_mode(self):
        return self.help_mode
