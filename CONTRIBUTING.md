# Contributing to KORSHMER

Thank you for helping. KORSHMER is MIT licensed, and contributions are accepted under the same licence.

## Setup

```bash
git clone https://github.com/YaronKoresh/korshmer.git
cd korshmer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
poe hook                         # installs the pre-commit hooks
```

On Windows `scripts/install.bat` does all of this. On Linux install the system libraries first:

```bash
sudo apt install libegl1 libgl1 libxkbcommon0 libportaudio2 libsndfile1 fluidsynth ffmpeg
```

The extras are split by purpose: `test` (pytest and its plugins), `lint` (ruff, mypy, vulture, poethepoet, pre-commit), `build` (Nuitka, PyInstaller, Briefcase, build) and `dev`, which is all three.

## Running the tests

```bash
QT_QPA_PLATFORM=offscreen pytest -q tests/unit tests/integration tests/regression
QT_QPA_PLATFORM=offscreen pytest -q tests/unit/test_plugins.py
QT_QPA_PLATFORM=offscreen pytest -q tests/performance      # wall-clock benchmarks
```

`QT_QPA_PLATFORM=offscreen` lets the Qt tests run without a display. Tests must not need the network, a real audio device or files outside pytest's `tmp_path`, and must not sleep to wait for another thread: wait on an event or a queue instead. The benchmarks in `tests/performance` measure wall-clock time, so CI runs them in a separate job that cannot block a change.

## Test first

1. Write a test that shows the problem or the new behaviour through a public entry point (a public class, function or widget, not a private helper).
2. Run it against the unchanged code and watch it fail on an assertion about behaviour, not on an import error.
3. Change the code where the state first goes wrong, not where the symptom shows.
4. Run it again, together with the existing tests for the files you touched.

One behaviour per test. Name tests as plain sentences a musician or any developer understands, for example `test_a_plugin_file_in_the_folder_is_not_run_without_consent`. A test that already passes on the unchanged code is only acceptable as a labelled safeguard next to one that fails. No ticket numbers or internal jargon in test or file names.

## Style

- Formatting and linting: `ruff format` and `ruff check` (line length 80, Python 3.10 syntax). `poe format` and `poe lint` run them on the whole tree.
- Types: `mypy src/korshmer --ignore-missing-imports` must not report errors in the files you changed.
- Dead code: `python -m vulture src/ --min-confidence 80` must print nothing. Everything new must be reachable from the running application.
- No comments and no docstrings in code. This is project policy: `scripts/strip_comments.py` removes them as part of `poe check`. Make names and small functions carry the meaning, and put explanations in the pull request and in the documentation. Markdown and YAML files may contain normal prose.
- No `print` debugging, no `TODO`/`FIXME` markers, no placeholder functions, no secrets.
- Keep changes small and in the style of the surrounding code.

`poe check` runs the full gate (compile, dead code, comment stripping, pre-commit hooks and tests).

## Pull requests

- Describe what changed for the user and why.
- Add a line to the Unreleased section of [CHANGELOG.md](CHANGELOG.md). A changed default, file format key, environment variable or public function contract also needs a line under its Migration notes.
- Update [README.md](README.md) when a visible name, file format, setting or environment variable changes.
- CI (`.github/workflows/ci.yml`) runs ruff, mypy and the unit, integration and regression tests on Python 3.10 to 3.13; it must be green.

## Security

Please report vulnerabilities privately as described in [SECURITY.md](SECURITY.md), not in public issues.
