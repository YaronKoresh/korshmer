__version__ = "1.0.0"

__all__ = [
    "audio",
    "audio_processing",
    "constants",
    "core",
    "managers",
    "midi",
    "midi_processing",
    "mixins",
    "project_management",
    "ui",
    "utils",
    "visualization",
    "workers",
]


def __getattr__(name):
    if name in __all__:
        from importlib import import_module

        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
