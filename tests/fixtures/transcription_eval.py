import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from korshmer.audio_processing.transcription_prep import (
    ModelOptions,
    PrepOptions,
    prepare_for_transcription,
    variants_for_stem,
)
from korshmer.midi.transcription_cleanup import (
    CleanupOptions,
    clean_transcription,
    merge_note_variants,
    options_for_cleanup,
)
from tests.fixtures.transcription_fixtures import (
    FIXTURE_DIR,
    SONGS,
    load_truth,
    to_pretty_midi,
)

MODEL_PARAMS = ModelOptions().as_kwargs()
PREVIOUS_MODEL_PARAMS = {
    "onset_threshold": 0.5,
    "frame_threshold": 0.3,
    "minimum_note_length": 80,
    "minimum_frequency": 30,
    "maximum_frequency": 4000,
}
CACHE_DIR = Path(
    os.environ.get(
        "KORSHMER_EVAL_CACHE",
        Path(tempfile.gettempdir()) / "korshmer_transcription_eval",
    )
)


def model_available():
    try:
        from basic_pitch import (
            ICASSP_2022_MODEL_PATH,
            ONNX_PRESENT,
            TF_PRESENT,
            TFLITE_PRESENT,
        )
    except Exception:
        return False
    return bool(
        (ONNX_PRESENT or TF_PRESENT or TFLITE_PRESENT)
        and Path(str(ICASSP_2022_MODEL_PATH)).exists()
    )


def song_names():
    return [s[0] for s in SONGS]


def _cache_key(name, prep, params):
    wav = (FIXTURE_DIR / f"{name}.wav").read_bytes()
    text = json.dumps(
        [hashlib.sha1(wav).hexdigest(), asdict(prep), params], sort_keys=True
    )
    return hashlib.sha1(text.encode()).hexdigest()


def run_model(name, prep=None, params=None, cache=True, variant=0):
    from basic_pitch.inference import predict

    prep = variants_for_stem(name, prep)[0 if variant == 0 else -1]
    params = dict(MODEL_PARAMS, **(params or {}))
    cache_file = CACHE_DIR / f"{_cache_key(name, prep, params)}.json"
    if cache and cache_file.exists():
        return [tuple(n) for n in json.loads(cache_file.read_text())]
    source = str(FIXTURE_DIR / f"{name}.wav")
    with tempfile.TemporaryDirectory() as tmp:
        path = prepare_for_transcription(source, tmp, prep, tag=str(variant))
        _, midi, _ = predict(path, midi_tempo=120.0, **params)
    notes = [
        (int(n.pitch), float(n.start), float(n.end), int(n.velocity))
        for inst in midi.instruments
        for n in inst.notes
    ]
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(notes))
    return notes


def mir_scores(reference, estimated, with_offset=False):
    import mir_eval

    if not reference or not estimated:
        return 0.0, 0.0, 0.0

    def arrays(notes):
        iv = np.array([[n[1], max(n[2], n[1] + 0.01)] for n in notes])
        hz = mir_eval.util.midi_to_hz(np.array([n[0] for n in notes]))
        return iv, hz

    ref_iv, ref_hz = arrays(reference)
    est_iv, est_hz = arrays(estimated)
    precision, recall, f1, _ = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_iv,
            ref_hz,
            est_iv,
            est_hz,
            onset_tolerance=0.05,
            pitch_tolerance=50.0,
            offset_ratio=0.2 if with_offset else None,
        )
    )
    return precision, recall, f1


def post_process(name, notes, cleanup):
    if cleanup is None or not cleanup.enabled:
        return list(notes)
    midi = to_pretty_midi(notes)
    cleaned, _ = clean_transcription(
        midi, options=options_for_cleanup(name, cleanup)
    )
    return [
        (n.pitch, n.start, n.end, n.velocity)
        for inst in cleaned.instruments
        for n in inst.notes
    ]


def transcribe(name, prep=None, params=None, cache=True, ensemble=True):
    variants = variants_for_stem(name, prep)
    if not ensemble:
        variants = variants[:1]
    runs = [
        run_model(name, prep, params, cache, variant=i)
        for i in range(len(variants))
    ]
    if len(runs) == 1:
        return runs[0]
    merged = merge_note_variants([to_pretty_midi(r) for r in runs])
    return [
        (int(n.pitch), float(n.start), float(n.end), int(n.velocity))
        for inst in merged.instruments
        for n in inst.notes
    ]


def evaluate(
    prep=None,
    cleanup=None,
    params=None,
    names=None,
    cache=True,
    ensemble=True,
):
    results = {}
    for name in names or song_names():
        truth = load_truth(name)
        estimated = post_process(
            name, transcribe(name, prep, params, cache, ensemble), cleanup
        )
        p, r, f1 = mir_scores(truth, estimated)
        _, _, f1o = mir_scores(truth, estimated, with_offset=True)
        results[name] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "f1_offset": f1o,
            "notes": len(estimated),
        }
    keys = ("precision", "recall", "f1", "f1_offset")
    results["mean"] = {
        k: float(np.mean([v[k] for n, v in results.items()])) for k in keys
    }
    return results


def format_table(results):
    lines = [f"{'song':<20}{'P':>7}{'R':>7}{'F1':>7}{'F1+off':>8}{'n':>6}"]
    for name, v in results.items():
        lines.append(
            f"{name:<20}{v['precision']:>7.3f}{v['recall']:>7.3f}"
            f"{v['f1']:>7.3f}{v['f1_offset']:>8.3f}{v.get('notes', ''):>6}"
        )
    return "\n".join(lines)


def main():
    raw_prep = PrepOptions(enabled=False)
    configs = {
        "raw model, thresholds 0.5/0.3": (
            raw_prep,
            None,
            PREVIOUS_MODEL_PARAMS,
            False,
        ),
        "raw model, thresholds 0.6/0.4": (raw_prep, None, None, False),
        "prep, one audio variant": (None, None, None, False),
        "prep, two audio variants": (None, None, None, True),
        "prep, two variants, cleanup": (None, CleanupOptions(), None, True),
    }
    for label, (prep, cleanup, params, ensemble) in configs.items():
        print(f"== {label}")
        print(format_table(evaluate(prep, cleanup, params, ensemble=ensemble)))


if __name__ == "__main__":
    main()
