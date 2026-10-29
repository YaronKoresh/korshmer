import math
import numbers
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable


class PluginType(Enum):
    MIDI_PROCESSOR = auto()
    AUDIO_EFFECT = auto()
    EXPORTER = auto()


class PluginParam:
    __slots__ = (
        "name",
        "label",
        "kind",
        "default",
        "min_val",
        "max_val",
        "choices",
    )

    def __init__(
        self,
        name,
        label,
        kind="float",
        default=0,
        min_val=0,
        max_val=100,
        choices=None,
    ):
        self.name = name
        self.label = label
        self.kind = kind
        self.default = default
        self.min_val = min_val
        self.max_val = max_val
        self.choices = choices or []

    def validate(self, value):
        if self.kind == "choice":
            return value in self.choices
        if self.kind in ("int", "float"):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                return False
            if math.isnan(value) or math.isinf(value):
                return False
            if self.kind == "int" and value != int(value):
                return False
            return bool(self.min_val <= value <= self.max_val)
        if self.kind == "bool":
            return isinstance(value, bool)
        return True


class PluginBase:
    name = "Unnamed Plugin"
    description = ""
    version = "1.0.0"
    author = ""
    category = "General"
    plugin_type = None

    @classmethod
    def params(cls):
        return []

    @classmethod
    def param_defaults(cls):
        return {p.name: p.default for p in cls.params()}


class MidiProcessorPlugin(PluginBase):
    plugin_type = PluginType.MIDI_PROCESSOR

    @abstractmethod
    def process(self, midi_path, output_dir, **params):
        pass


class AudioEffectPlugin(PluginBase):
    plugin_type = PluginType.AUDIO_EFFECT

    @abstractmethod
    def process(self, audio_path, output_path, **params):
        pass


class ExporterPlugin(PluginBase):
    plugin_type = PluginType.EXPORTER

    @abstractmethod
    def export(self, midi_path, output_path, **params):
        pass


@runtime_checkable
class IAudioProcessor(Protocol):
    def process(
        self, audio_path: str, output_path: str, **params: Any
    ) -> str: ...


@runtime_checkable
class IMidiProcessor(Protocol):
    def process(
        self, midi_path: str, output_dir: str, **params: Any
    ) -> str: ...


@runtime_checkable
class IExporter(Protocol):
    def export(
        self, midi_path: str, output_path: str, **params: Any
    ) -> str: ...


@runtime_checkable
class IAnalyzer(Protocol):
    def analyze(self, file_path: str, **params: Any) -> dict[str, Any]: ...


@dataclass
class PluginMetadata:
    name: str
    version: str
    author: str
    category: str
    plugin_type: PluginType
    params: list[PluginParam] = field(default_factory=list)
    description: str = ""
    tags: list[str] = field(default_factory=list)
