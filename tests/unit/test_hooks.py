import random

import pytest

from korshmer.midi_generation import genre_feel, hooks
from korshmer.midi_generation.engine import MELODY_MOODS, SONG_STYLES

SEEDS = range(24)
KINDS = ("chorus", "verse", "intro", "outro", "break", "bridge")


def _hook(style="Pop", seed=0, mood="Uplifting"):
    return hooks.make_hook(random.Random(seed), style, mood)


def _fixed_hook():
    notes = ((0, 3, 0), (3, 3, 2), (6, 2, 1), (8, 4, 4), (12, 4, 2))
    return hooks.Hook(notes, 0.9, 74)


def _fits(plan):
    pairs = zip(plan, plan[1:], strict=False)
    return all(o + n <= nxt for (o, n, _), (nxt, _, _) in pairs) and (
        plan[-1][0] + plan[-1][1] <= hooks.STEPS
    )


def _moves(hook):
    steps = [s for _, _, s in hook.notes]
    return [b - a for a, b in zip(steps, steps[1:], strict=False)]


def test_every_style_has_a_hook_palette():
    for style in SONG_STYLES:
        assert genre_feel.feel_for(style).hook.rhythms


@pytest.mark.parametrize("style", SONG_STYLES)
def test_hook_rhythms_fit_one_bar_without_overlap(style):
    for rhythm in genre_feel.feel_for(style).hook.rhythms:
        assert 3 <= len(rhythm) <= 8
        assert all(length >= 1 for _, length in rhythm)
        for (onset, length), (following, _) in zip(
            rhythm, rhythm[1:], strict=False
        ):
            assert onset + length <= following
        assert 0 <= rhythm[0][0]
        assert rhythm[-1][0] + rhythm[-1][1] <= hooks.STEPS


def test_hook_palettes_differ_between_styles():
    palettes = {style: genre_feel.feel_for(style).hook for style in SONG_STYLES}
    assert len({repr(p) for p in palettes.values()}) == len(palettes)
    pools = {p.rhythms for p in palettes.values()}
    assert len(pools) == len(palettes)


@pytest.mark.parametrize("style", SONG_STYLES)
def test_same_seed_gives_the_same_hook(style):
    assert _hook(style, 4) == _hook(style, 4)


@pytest.mark.parametrize("style", SONG_STYLES)
def test_seeds_give_different_hooks(style):
    found = {_hook(style, seed).notes for seed in SEEDS}
    assert len(found) >= 0.75 * len(SEEDS)


@pytest.mark.parametrize("mood", MELODY_MOODS)
@pytest.mark.parametrize("style", SONG_STYLES)
def test_hooks_are_built_to_stick(style, mood):
    spec = genre_feel.feel_for(style).hook
    for seed in SEEDS:
        hook = _hook(style, seed, mood)
        steps = [s for _, _, s in hook.notes]
        moves = _moves(hook)
        leaps = [m for m in moves if abs(m) >= hooks.LEAP]
        assert 3 <= len(hook.notes) <= 8
        assert hooks.LOW_STEP <= min(steps)
        assert max(steps) <= hooks.HIGH_STEP
        assert 0 in moves or len(leaps) == 1
        assert len(leaps) <= 2
        assert 2 <= len(set(steps)) <= 6
        assert max(steps) - min(steps) <= spec.span[1] + 1
        assert hook.gate == spec.gate
        assert hook.register == spec.register


@pytest.mark.parametrize("style", SONG_STYLES)
def test_strong_beats_of_a_hook_are_chord_tones(style):
    for seed in SEEDS:
        for onset, length, step in _hook(style, seed).notes:
            if hooks.strong(onset, length):
                assert step in hooks.CHORD_STEPS


def test_unknown_style_and_mood_fall_back_to_defaults():
    assert hooks.make_hook(random.Random(2), "Polka", "Grumpy").notes
    assert _hook("Polka", 2) == _hook(genre_feel.DEFAULT_FEEL, 2)


def test_strong_marks_downbeats_and_long_notes_only():
    assert hooks.strong(0, 1)
    assert hooks.strong(8, 1)
    assert hooks.strong(6, 4)
    assert not hooks.strong(3, 3)
    assert not hooks.strong(10, 2)


def test_score_prefers_a_hook_with_one_leap_and_a_repeat():
    spec = genre_feel.feel_for("Pop").hook
    rhythm = ((0, 3), (3, 3), (6, 2), (8, 3), (11, 3))
    good = [0, 0, -1, 4, 2]
    run = [0, 1, 2, 3, 4]
    flat = [2, 2, 2, 2, 2]
    scores = {
        name: hooks.score_hook(rhythm, steps, spec, "Uplifting")
        for name, steps in (("good", good), ("run", run), ("flat", flat))
    }
    assert scores["good"] > scores["run"]
    assert scores["good"] > scores["flat"]


def test_score_follows_the_mood():
    spec = genre_feel.feel_for("Pop").hook
    rhythm = ((0, 3), (3, 3), (6, 2), (8, 3))
    falling = [4, 2, 0, 0]
    rising = [0, 0, 2, 4]

    def gap(mood):
        return hooks.score_hook(rhythm, falling, spec, mood) - hooks.score_hook(
            rhythm, rising, spec, mood
        )

    assert gap("Melancholic") > gap("Uplifting")


def test_hook_role_is_the_hook_itself():
    hook = _fixed_hook()
    assert hooks.plan_bar(hook, "hook", random.Random(0)) == list(hook.notes)


def test_lift_moves_every_note_up_two_steps():
    hook = _fixed_hook()
    lifted = hooks.plan_bar(hook, "lift", random.Random(0))
    assert [n[:2] for n in lifted] == [n[:2] for n in hook.notes]
    assert [n[2] - m[2] for n, m in zip(lifted, hook.notes, strict=True)] == [
        2
    ] * len(hook.notes)


def test_inversion_mirrors_the_contour():
    hook = _fixed_hook()
    mirrored = hooks.plan_bar(hook, "inv", random.Random(0))
    assert [s for _, _, s in mirrored] == [-s for _, _, s in hook.notes]


def test_skeleton_keeps_the_downbeat_and_fills_the_gaps():
    hook = _fixed_hook()
    plan = hooks.plan_bar(hook, "skeleton", random.Random(0))
    onsets = [o for o, _, _ in plan]
    assert onsets[0] == hook.notes[0][0]
    assert 2 <= len(plan) < len(hook.notes) + 1
    assert set(onsets) <= {o for o, _, _ in hook.notes}
    assert _fits(plan)


def test_augmentation_states_the_head_at_half_speed():
    hook = _fixed_hook()
    plan = hooks.plan_bar(hook, "aug", random.Random(0))
    head = [n for n in hook.notes if n[0] < hooks.HEAD_END]
    assert [o for o, _, _ in plan] == [2 * o for o, _, _ in head]
    assert [s for _, _, s in plan] == [s for _, _, s in head]
    assert all(o + l <= hooks.STEPS for o, l, _ in plan)


def test_variation_keeps_the_start_and_stays_inside_the_bar():
    hook = _fixed_hook()
    seen = set()
    for seed in range(60):
        plan = hooks.plan_bar(hook, "vary", random.Random(seed))
        seen.add(tuple(plan))
        assert plan[0][0] == hook.notes[0][0]
        assert _fits(plan)
    assert len(seen) >= 3


@pytest.mark.parametrize("role", ["half", "close"])
def test_answers_quote_the_head_and_end_on_a_target(role):
    hook = _fixed_hook()
    plan = hooks.plan_bar(hook, role, random.Random(0))
    *head, last = plan
    limit = hooks.ANSWER_HEAD[role]
    assert 1 <= len(head) <= limit
    assert last == (hooks.HEAD_END, hooks.ANSWER_LENGTH, None)
    assert all(o + l <= hooks.HEAD_END for o, l, _ in head)
    assert head == [
        (o, min(l, hooks.HEAD_END - o), s)
        for o, l, s in hook.notes[: len(head)]
    ]


def test_answers_differ_in_length_when_the_head_allows():
    hook = _fixed_hook()
    half = hooks.plan_bar(hook, "half", random.Random(0))
    close = hooks.plan_bar(hook, "close", random.Random(0))
    assert len(half) > len(close)


@pytest.mark.parametrize("kind", KINDS)
def test_phrase_templates_are_four_bars_and_use_known_roles(kind):
    known = {"hook", "vary", "lift", "inv", "skeleton", "aug", "half", "close"}
    for phrase in hooks.PHRASES[kind]:
        assert len(phrase) == hooks.PHRASE_BARS
        assert set(phrase) <= known
        assert phrase[-1] in ("half", "close")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("bars", [1, 3, 4, 6, 8, 12, 16])
def test_roles_cover_every_bar_and_end_with_an_answer(kind, bars):
    roles = hooks.bar_roles(kind, bars)
    assert len(roles) == bars
    assert roles[-1] in ("half", "close")


@pytest.mark.parametrize("kind", KINDS)
def test_final_bar_of_the_song_closes(kind):
    assert hooks.bar_roles(kind, 8, final=True)[-1] == "close"
    assert (
        hooks.bar_roles(kind, 8)[:-1]
        == hooks.bar_roles(kind, 8, final=True)[:-1]
    )


def test_chorus_states_the_hook_twice_before_varying_it():
    roles = hooks.bar_roles("chorus", 8)
    assert roles[:2] == ["hook", "hook"]
    assert roles[4:6] == ["hook", "hook"]
    assert roles[3] == "half"
    assert roles[7] == "close"


def test_verse_is_a_lighter_form_of_the_hook():
    assert hooks.bar_roles("verse", 4)[:2] == ["skeleton", "skeleton"]


def test_unknown_kind_is_treated_as_a_chorus():
    assert hooks.bar_roles("coda", 8) == hooks.bar_roles("chorus", 8)


def test_zero_bars_give_no_roles():
    assert hooks.bar_roles("chorus", 0) == []


def test_stab_hits_follow_the_longest_hook_notes():
    hits = hooks.stab_hits(_fixed_hook())
    assert hits == sorted(set(hits))
    assert all(0.0 <= h < 1.0 for h in hits)
    assert len(hits) <= 4
    assert 0.5 in hits
    assert hooks.stab_hits(_fixed_hook(), limit=1) == [0.5]


def test_stab_hits_never_come_back_empty():
    short = hooks.Hook(((0, 1, 0), (2, 1, 1), (4, 1, 2)), 0.9, 74)
    assert hooks.stab_hits(short) == [0.0]
