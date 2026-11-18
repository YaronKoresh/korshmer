from types import ModuleType

try:
    import numpy as _np
    import soundfile as _sf

    np: ModuleType | None = _np
    sf: ModuleType | None = _sf
    _AUDIO_DEPS = True
except ImportError:
    np = None
    sf = None
    _AUDIO_DEPS = False

try:
    from korshmer.audio_file import AudioFile as _AudioFile

    AudioFile: type[_AudioFile] | None = _AudioFile
except ImportError:
    AudioFile = None

from korshmer.plugins.base import AudioEffectPlugin, PluginParam


def _load_audio(audio_path: str) -> "_AudioFile":
    if AudioFile is None:
        raise ImportError(
            "Audio plugins need the 'numpy' and 'soundfile' packages to load audio files."
        )
    loaded: _AudioFile = AudioFile.load(audio_path).ensure_2d()
    return loaded


def _soundfile() -> ModuleType:
    if sf is None:
        raise ImportError(
            "Audio plugins need the 'soundfile' package to save audio files."
        )
    return sf


class ReverbPlugin(AudioEffectPlugin):
    name = "Reverb"
    version = "1.0.0"
    category = "Space"

    @classmethod
    def params(cls):
        return [
            PluginParam("room_size", "Room Size", "float", 0.5, 0, 1.0),
            PluginParam("decay", "Decay", "float", 0.5, 0, 1.0),
            PluginParam("wet", "Wet Mix", "float", 0.3, 0, 1.0),
            PluginParam("pre_delay_ms", "Pre-Delay (ms)", "float", 0, 0, 200),
            PluginParam(
                "decay_time_s", "Decay Time (s)", "float", 2.0, 0.1, 30.0
            ),
        ]

    def process(
        self, audio_path: str, output_path: str, **params: object
    ) -> str:
        from korshmer.audio_processing import Reverb

        af = _load_audio(audio_path)
        fx = Reverb(af.sr)
        af.y = fx.process(af.y, **params)
        _soundfile().write(output_path, af.to_write_format(), af.sr)
        return output_path


class DelayPlugin(AudioEffectPlugin):
    name = "Delay"
    version = "1.0.0"
    category = "Space"

    @classmethod
    def params(cls):
        return [
            PluginParam("delay_ms", "Delay (ms)", "float", 250, 1, 2000),
            PluginParam("feedback", "Feedback", "float", 0.3, 0, 0.95),
            PluginParam("wet", "Wet Mix", "float", 0.3, 0, 1.0),
        ]

    def process(
        self, audio_path: str, output_path: str, **params: object
    ) -> str:
        from korshmer.audio_processing import Delay

        af = _load_audio(audio_path)
        fx = Delay(af.sr)
        af.y = fx.process(af.y, **params)
        _soundfile().write(output_path, af.to_write_format(), af.sr)
        return output_path


class ChorusPlugin(AudioEffectPlugin):
    name = "Chorus"
    version = "1.0.0"
    category = "Modulation"

    @classmethod
    def params(cls):
        return [
            PluginParam("rate_hz", "Rate (Hz)", "float", 1.5, 0.1, 10.0),
            PluginParam("depth_ms", "Depth (ms)", "float", 5.0, 0.1, 20.0),
            PluginParam("mix", "Mix", "float", 0.5, 0, 1.0),
        ]

    def process(
        self, audio_path: str, output_path: str, **params: object
    ) -> str:
        from korshmer.audio_processing import Chorus

        af = _load_audio(audio_path)
        fx = Chorus(af.sr)
        af.y = fx.process(af.y, **params)
        _soundfile().write(output_path, af.to_write_format(), af.sr)
        return output_path


class FlangerPlugin(AudioEffectPlugin):
    name = "Flanger"
    version = "1.0.0"
    category = "Modulation"

    @classmethod
    def params(cls):
        return [
            PluginParam("rate_hz", "Rate (Hz)", "float", 0.5, 0.05, 5.0),
            PluginParam("depth_ms", "Depth (ms)", "float", 2.0, 0.1, 10.0),
            PluginParam("feedback", "Feedback", "float", 0.3, 0, 0.95),
            PluginParam("mix", "Mix", "float", 0.5, 0, 1.0),
        ]

    def process(
        self, audio_path: str, output_path: str, **params: object
    ) -> str:
        from korshmer.audio_processing import Flanger

        af = _load_audio(audio_path)
        fx = Flanger(af.sr)
        af.y = fx.process(af.y, **params)
        _soundfile().write(output_path, af.to_write_format(), af.sr)
        return output_path


class PhaserPlugin(AudioEffectPlugin):
    name = "Phaser"
    version = "1.0.0"
    category = "Modulation"

    @classmethod
    def params(cls):
        return [
            PluginParam("rate", "Rate (Hz)", "float", 0.5, 0.05, 5.0),
            PluginParam("depth", "Depth", "float", 0.7, 0, 1.0),
            PluginParam("stages", "Stages", "int", 4, 2, 12),
            PluginParam("mix", "Mix", "float", 0.5, 0, 1.0),
        ]

    def process(
        self, audio_path: str, output_path: str, **params: object
    ) -> str:
        from korshmer.audio_processing import Phaser

        af = _load_audio(audio_path)
        fx = Phaser(af.sr)
        af.y = fx.process(af.y, **params)
        _soundfile().write(output_path, af.to_write_format(), af.sr)
        return output_path


class TremoloPlugin(AudioEffectPlugin):
    name = "Tremolo"
    version = "1.0.0"
    category = "Modulation"

    @classmethod
    def params(cls):
        return [
            PluginParam("rate_hz", "Rate (Hz)", "float", 5.0, 0.1, 20.0),
            PluginParam("depth", "Depth", "float", 0.5, 0, 1.0),
            PluginParam(
                "shape",
                "Shape",
                "choice",
                "sine",
                choices=["sine", "square", "triangle"],
            ),
        ]

    def process(
        self, audio_path: str, output_path: str, **params: object
    ) -> str:
        from korshmer.audio_processing import Tremolo

        af = _load_audio(audio_path)
        fx = Tremolo(af.sr)
        af.y = fx.process(af.y, **params)
        _soundfile().write(output_path, af.to_write_format(), af.sr)
        return output_path


class StereoWidenerPlugin(AudioEffectPlugin):
    name = "Stereo Widener"
    version = "1.0.0"
    category = "Imaging"

    @classmethod
    def params(cls):
        return [
            PluginParam("width", "Width", "float", 1.5, 0, 3.0),
        ]

    def process(
        self, audio_path: str, output_path: str, **params: object
    ) -> str:
        from korshmer.audio_processing import StereoWidener

        af = _load_audio(audio_path)
        fx = StereoWidener(af.sr)
        af.y = fx.process(af.y, **params)
        _soundfile().write(output_path, af.to_write_format(), af.sr)
        return output_path


class NormalizePlugin(AudioEffectPlugin):
    name = "Normalize"
    version = "1.0.0"
    category = "Utility"

    @classmethod
    def params(cls):
        return []

    def process(
        self, audio_path: str, output_path: str, **params: object
    ) -> str:
        from korshmer.audio_processing import apply_normalize

        apply_normalize(audio_path, output_path)
        return output_path


class ReversePlugin(AudioEffectPlugin):
    name = "Reverse"
    version = "1.0.0"
    category = "Utility"

    @classmethod
    def params(cls):
        return []

    def process(
        self, audio_path: str, output_path: str, **params: object
    ) -> str:
        from korshmer.audio_processing import apply_reverse_audio

        apply_reverse_audio(audio_path, output_path)
        return output_path


class FilterPlugin(AudioEffectPlugin):
    name = "Filter"
    version = "1.0.0"
    category = "EQ"

    @classmethod
    def params(cls):
        return [
            PluginParam(
                "filter_type",
                "Type",
                "choice",
                "low",
                choices=["low", "high"],
            ),
            PluginParam("cutoff", "Cutoff (Hz)", "float", 1000, 20, 20000),
        ]

    def process(
        self, audio_path: str, output_path: str, **params: object
    ) -> str:
        from korshmer.audio_processing import apply_filter

        apply_filter(audio_path, output_path, **params)
        return output_path


class NoiseGatePlugin(AudioEffectPlugin):
    name = "Noise Gate"
    version = "1.0.0"
    category = "Dynamics"

    @classmethod
    def params(cls):
        return [
            PluginParam(
                "threshold_db", "Threshold (dB)", "float", -40.0, -80.0, 0.0
            ),
        ]

    def process(
        self, audio_path: str, output_path: str, **params: object
    ) -> str:
        from korshmer.audio_processing import apply_noise_gate

        apply_noise_gate(audio_path, output_path, **params)
        return output_path


class TransientShaperPlugin(AudioEffectPlugin):
    name = "Transient Shaper"
    version = "1.0.0"
    category = "Dynamics"

    @classmethod
    def params(cls):
        return [
            PluginParam("attack", "Attack", "float", 1.0, 0.0, 3.0),
            PluginParam("sustain", "Sustain", "float", 1.0, 0.0, 3.0),
            PluginParam("amount", "Amount", "float", 0.5, 0.0, 2.0),
        ]

    def process(
        self, audio_path: str, output_path: str, **params: object
    ) -> str:
        from korshmer.audio_processing import apply_transient_shaper

        apply_transient_shaper(audio_path, output_path, **params)
        return output_path


class SmartMasteringPlugin(AudioEffectPlugin):
    name = "Smart Mastering"
    version = "1.0.0"
    category = "Mastering"

    @classmethod
    def params(cls):
        return []

    def process(self, audio_path, output_path, **params):
        from korshmer.audio_processing import SmartMastering, save_audio

        af = _load_audio(audio_path)
        processor = SmartMastering()
        af.y, _ = processor.process(af.y, af.sr)
        save_audio(path=output_path, y=af.y, sr=af.sr)
        return output_path


BUILTIN_PLUGINS = [
    ReverbPlugin,
    DelayPlugin,
    ChorusPlugin,
    FlangerPlugin,
    PhaserPlugin,
    TremoloPlugin,
    StereoWidenerPlugin,
    NormalizePlugin,
    ReversePlugin,
    FilterPlugin,
    NoiseGatePlugin,
    TransientShaperPlugin,
    SmartMasteringPlugin,
]


def register_builtins(registry=None):
    if registry is None:
        from korshmer.plugins.registry import PluginRegistry

        registry = PluginRegistry()
    count = 0
    for plugin_cls in BUILTIN_PLUGINS:
        if registry.register(plugin_cls):
            count += 1
    return count
