from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from korshmer.constants import MAX_EFFECT_CHAIN_DEPTH


def validate_effect_chain_depth(effects: Sequence[Callable[[Any], Any]]):
    if isinstance(effects, (str, bytes)) or not isinstance(effects, Sequence):
        raise ValueError(
            f"Effect chain must be a list of effects, got {type(effects).__name__}"
        )
    if len(effects) > MAX_EFFECT_CHAIN_DEPTH:
        raise ValueError(
            f"Effect chain depth must be at most {MAX_EFFECT_CHAIN_DEPTH} "
            f"effects, got {len(effects)}"
        )
    for index, effect in enumerate(effects):
        if not callable(effect):
            raise ValueError(
                f"Effect {index} in the chain is not callable: {effect!r}"
            )
    return effects


def apply_effect_chain(audio, effects: Sequence[Callable[[Any], Any]]):
    for effect in validate_effect_chain_depth(effects):
        audio = effect(audio)
    return audio
