__all__ = [
    "AudioEngine",
    "ControlBar",
    "FluidSynth",
]


def __getattr__(name):
    if name == "AudioEngine":
        from korshmer.core.audio_engine import AudioEngine

        globals()[name] = AudioEngine
        return AudioEngine
    elif name == "ControlBar":
        from korshmer.core.control_bar import ControlBar

        globals()[name] = ControlBar
        return ControlBar
    elif name == "FluidSynth":
        from korshmer.core.fluidsynth import FluidSynth

        globals()[name] = FluidSynth
        return FluidSynth
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
