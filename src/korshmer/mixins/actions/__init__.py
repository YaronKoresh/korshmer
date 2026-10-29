from .audio import AudioActionsMixin
from .base import BaseActionsMixin
from .core import CoreActionsMixin
from .editor import EditorActionsMixin
from .harmony import ArpHarmonyActionsMixin
from .modifiers import ModifiersActionsMixin
from .project_management import ProjectManagementActionsMixin
from .stems import StemsActionsMixin

__all__ = [
    "ActionsMixin",
    "ArpHarmonyActionsMixin",
    "AudioActionsMixin",
    "BaseActionsMixin",
    "CoreActionsMixin",
    "EditorActionsMixin",
    "StemsActionsMixin",
    "ModifiersActionsMixin",
    "ProjectManagementActionsMixin",
]


class ActionsMixin(
    EditorActionsMixin,
    StemsActionsMixin,
    ArpHarmonyActionsMixin,
    ModifiersActionsMixin,
    AudioActionsMixin,
    CoreActionsMixin,
    ProjectManagementActionsMixin,
):
    pass
