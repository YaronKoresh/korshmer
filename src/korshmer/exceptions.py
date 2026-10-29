from __future__ import annotations

import time


class KorshmerError(Exception):
    def __init__(
        self, message: str, recovery_hint: str = "", context: dict | None = None
    ):
        super().__init__(message)
        self.message = message
        self.recovery_hint = recovery_hint
        self.context = context or {}
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "recovery_hint": self.recovery_hint,
            "context": self.context,
            "timestamp": self.timestamp,
        }


class AudioError(KorshmerError):
    pass


class AudioFileNotFoundError(AudioError):
    pass


class AudioFormatError(AudioError):
    pass


class AudioFileTooLargeError(AudioError):
    pass


class AudioProcessingError(AudioError):
    pass


class InvalidAudioInputError(AudioProcessingError, ValueError):
    pass


class StemSeparationError(AudioProcessingError):
    pass


class AudioToMidiError(AudioProcessingError):
    pass


class AudioPlaybackError(AudioError):
    pass


class AudioRenderError(AudioError):
    pass


class MidiError(KorshmerError):
    pass


class MidiFileNotFoundError(MidiError):
    pass


class MidiFormatError(MidiError):
    pass


class MidiProcessingError(MidiError):
    pass


class QuantizationError(MidiProcessingError):
    pass


class ScaleSnapError(MidiProcessingError):
    pass


class MidiExportError(MidiError):
    pass


class ProjectError(KorshmerError):
    pass


class ProjectNotFoundError(ProjectError):
    pass


class ProjectCorruptedError(ProjectError):
    pass


class ProjectVersionError(ProjectError):
    pass


class ProjectSaveError(ProjectError):
    pass


class ProjectLoadError(ProjectError):
    pass


class PluginError(KorshmerError):
    pass


class PluginNotFoundError(PluginError):
    pass


class PluginLoadError(PluginError):
    pass


class PluginExecutionError(PluginError):
    pass


class PluginVersionError(PluginError):
    pass


class ProcessingError(KorshmerError):
    pass


class ProcessingCancelledError(ProcessingError):
    pass


class ProcessingTimeoutError(ProcessingError):
    pass


class ModelNotAvailableError(ProcessingError):
    pass


class GPUNotAvailableError(ProcessingError):
    pass


class CacheError(ProcessingError):
    pass


class ValidationError(KorshmerError):
    pass


class InvalidParameterError(ValidationError):
    pass


class InvalidFileError(ValidationError):
    pass


class InvalidStateError(ValidationError):
    pass


class CollaborationError(KorshmerError):
    pass


class SessionNotFoundError(CollaborationError):
    pass


class SessionConflictError(CollaborationError):
    pass


class ConnectionError(CollaborationError):
    pass
