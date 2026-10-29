import numpy as np
import pytest

from korshmer.audio_processing.effect_chain import (
    MAX_EFFECT_CHAIN_DEPTH,
    apply_effect_chain,
    validate_effect_chain_depth,
)


def _gain(factor):
    def effect(audio):
        return audio * factor

    return effect


class TestValidateEffectChainDepth:
    def test_max_depth_is_16(self):
        assert MAX_EFFECT_CHAIN_DEPTH == 16

    def test_accepts_exactly_max(self):
        chain = [_gain(1.0)] * MAX_EFFECT_CHAIN_DEPTH
        assert validate_effect_chain_depth(chain) is chain

    def test_accepts_empty(self):
        assert validate_effect_chain_depth([]) == []

    def test_rejects_seventeen_with_limit_in_message(self):
        chain = [_gain(1.0)] * (MAX_EFFECT_CHAIN_DEPTH + 1)
        with pytest.raises(ValueError) as exc:
            validate_effect_chain_depth(chain)
        assert "16" in str(exc.value)
        assert "17" in str(exc.value)

    @pytest.mark.parametrize("value", [None, "effects", b"effects", 5])
    def test_rejects_non_sequence(self, value):
        with pytest.raises(ValueError):
            validate_effect_chain_depth(value)

    def test_rejects_generator(self):
        gen = (_gain(1.0) for _ in range(3))
        with pytest.raises(ValueError):
            validate_effect_chain_depth(gen)

    def test_rejects_non_callable_item(self):
        with pytest.raises(ValueError):
            validate_effect_chain_depth([_gain(1.0), 0.5])


class TestApplyEffectChain:
    def test_applies_effects_in_order(self):
        audio = np.ones(4)
        out = apply_effect_chain(audio, [_gain(2.0), lambda a: a + 1.0])
        np.testing.assert_allclose(out, np.full(4, 3.0))

    def test_sixteen_effects_process(self):
        audio = np.ones(8)
        chain = [_gain(1.0)] * MAX_EFFECT_CHAIN_DEPTH
        np.testing.assert_allclose(apply_effect_chain(audio, chain), audio)

    def test_seventeen_effects_rejected_before_any_effect_runs(self):
        calls = []

        def tracking(audio):
            calls.append(1)
            return audio

        chain = [tracking] * (MAX_EFFECT_CHAIN_DEPTH + 1)
        with pytest.raises(ValueError):
            apply_effect_chain(np.ones(4), chain)
        assert calls == []

    def test_empty_chain_returns_audio_unchanged(self):
        audio = np.arange(3.0)
        np.testing.assert_allclose(apply_effect_chain(audio, []), audio)


class TestProjectEffectsChainDepth:
    @staticmethod
    def _data(n):
        return {
            "version": "0.3.0",
            "audio_processing": {"effects_chain": [{}] * n},
        }

    def test_constant_in_constants_module(self):
        from korshmer import constants

        assert constants.MAX_EFFECT_CHAIN_DEPTH == 16

    def test_migration_keeps_sixteen(self):
        from korshmer.project_management.project_migration import (
            migrate_project_data,
        )

        out = migrate_project_data(self._data(16), "0.5.1", "b")
        assert len(out["audio_processing"]["effects_chain"]) == 16

    def test_migration_rejects_seventeen(self):
        from korshmer.project_management.project_migration import (
            migrate_project_data,
        )

        with pytest.raises(ValueError, match="16"):
            migrate_project_data(self._data(17), "0.5.1", "b")

    def test_load_rejects_seventeen(self, tmp_path):
        import json

        from korshmer.project_management.project_manager import ProjectManager

        f = tmp_path / "p.json"
        f.write_text(json.dumps(self._data(17)))
        pm = ProjectManager(base_dir=tmp_path)
        assert pm.load_project(f) is None
        assert pm.current_project_data != self._data(17)
