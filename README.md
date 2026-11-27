# KORSHMER

![Version](https://img.shields.io/badge/Version-1.1.1-red?style=flat-square)

KORSHMER is an audio workstation (DAW) that combines a multi-lane MIDI editor, audio clips, AI audio-to-MIDI conversion, stem separation and a song generator in one window.

- [Installation options](#installation-options)
- [Run from source (no compiling)](#run-from-source-no-compiling)
- [Build korshmer.exe yourself](#build-korshmerexe-yourself)
- [Use korshmer.exe](#use-korshmerexe)
- [Quick start](#quick-start)
- [Features](#features)
- [Settings](#settings)
- [File formats](#file-formats)
- [Where files are stored](#where-files-are-stored)
- [Troubleshooting](#troubleshooting)
- [Testing and development](#testing-and-development)
- [License and Third-Party Licenses](#license-and-third-party-licenses)

# Installation options

| You want to | Do this | You need |
|-------------|---------|----------|
| Run KORSHMER on Windows without compiling | [`scripts\install.bat`, then `scripts\run.bat`](#windows) | Internet for the first install |
| Run KORSHMER on Linux or macOS | [Install with pip](#linux-and-macos) | Python 3.10+ and a few system libraries |
| Make a `korshmer.exe` that runs without Python | [`scripts\compile.bat`](#build-korshmerexe-yourself) | Everything above, plus Visual Studio Build Tools |
| Run a `korshmer.exe` you already have | [Use korshmer.exe](#use-korshmerexe) | Nothing else |

Requirements for every option: a 64-bit system and 8 GB of RAM or more for the AI tools. FFmpeg is optional and adds AAC/M4A import, MP3/AAC export and audio from video files.

# Run from source (no compiling)

## Windows

```cmd
git clone https://github.com/YaronKoresh/korshmer.git
cd korshmer
scripts\install.bat
scripts\run.bat
```

`install.bat`:

1. Uses Python 3.10 or newer if it finds one, and otherwise installs Python 3.11.
2. Installs whatever else is missing: the Windows SDK signing tools, FFmpeg and Clang. It asks for administrator rights for these, and after each install it asks you to run it again.
3. Runs a downloaded Python, Windows SDK or LLVM installer only when its digital signature is valid and names the expected publisher. FFmpeg comes from winget, and only when winget fails is it downloaded unsigned from gyan.dev, so install FFmpeg yourself first if you want to check it by hand.
4. Creates `.venv` and installs KORSHMER with its development tools into it.

`run.bat` starts KORSHMER from `.venv`. Any arguments you pass go on to `python -m korshmer`.

FluidSynth and a General MIDI SoundFont (MuseScore General) are already in the `fluidsynth/` folder, so MIDI plays without any further setup.

## Linux and macOS

Install the system libraries first:

```bash
sudo apt install libsndfile1 libportaudio2 fluidsynth ffmpeg
```

On macOS:

```bash
brew install libsndfile portaudio fluid-synth ffmpeg
```

Then install and run KORSHMER:

```bash
git clone https://github.com/YaronKoresh/korshmer.git
cd korshmer
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
korshmer
```

`python -m korshmer` also starts it. The SoundFont in `fluidsynth/` is used on every system.

Optional extras, installed with `pip install -e ".[name]"`:

| Extra | Adds |
|-------|------|
| `link` | Ableton Link tempo sync |
| `ndi` | NDI video output |
| `cuda` / `opencl` | GPU acceleration |
| `test`, `lint`, `build` | Test tools, lint tools, Nuitka/PyInstaller/Briefcase |
| `dev` | `test`, `lint` and `build` together |

# Build korshmer.exe yourself

Before you build:

1. Run `scripts\install.bat` once, as described in [Windows](#windows).
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the **Desktop development with C++** workload. Without MSVC, Nuitka builds with the zig compiler, which cannot link the build's link-time optimization (`--lto=yes`) and fails with `lld-link: error: undefined symbol`. MSVC also lets Nuitka bundle the Visual C++ runtime DLLs, so people who run the program do not need to install them.

Then build:

```cmd
scripts\compile.bat
```

The script installs Nuitka if it is missing, downloads the AI models into `models/`, and compiles KORSHMER with the models, `fluidsynth/` and the icon included. The build takes a long time and needs several GB of free disk space. The result is the folder `build\korshmer.dist\`, with `korshmer.exe` inside it.

Alternative: `poe build-exe` makes a PyInstaller build in `dist/korshmer`, and `poe build-exe-signed` also signs it.

| Signing on | Set these environment variables |
|------------|---------------------------------|
| Windows | `KORSHMER_WIN_CERT_FILE`, `KORSHMER_WIN_CERT_PASSWORD`, optionally `KORSHMER_TIMESTAMP_URL` |
| macOS | `KORSHMER_MACOS_IDENTITY` (a Developer ID Application identity) |

If signing or its verification fails, the build fails.

# Use korshmer.exe

1. Keep the whole `korshmer.dist` folder together, because `korshmer.exe` does not run on its own. To share the program, zip the whole folder.
2. Put the folder somewhere you can write to, such as Documents or your Desktop, and not in `Program Files`. KORSHMER creates its `saved_data\` and `temp\` folders next to `korshmer.exe`.
3. Double-click `korshmer.exe`. You do not need Python, and FluidSynth, the SoundFont and the AI models are inside the folder.
4. For FFmpeg formats, install FFmpeg (for example `winget install Gyan.FFmpeg.Essentials`).
5. If Windows SmartScreen warns about a build that is not signed, choose **More info**, then **Run anyway**.

# Quick start

| To | Do |
|----|----|
| Convert audio to MIDI | Add the file with **✚ File** (or drop it on the window), select it and press **Start Processing**. It is also under **EFFECTS → Audio Tools → Audio to MIDI**. |
| Import or merge MIDI | Use **✚ File** or drop the file on the window. **EDITOR → ✚ Import/Merge MIDI** adds it to the open song. |
| Separate stems | **EFFECTS → Audio Tools → Stem Separation**. The **STEMS** tab chooses which stems a conversion uses. Accept **Load into editor** to get the vocals as an audio clip and the instruments as MIDI lanes. |
| Put an audio file on the timeline | Press 🎤 next to it in **My music**, or drag it onto the editor. |
| Fix a clip | Right-click it: align to the beat, remove noise, improve the level, fix automatically, or move it one beat earlier or later. |
| Generate a song | **MIDI Generator**: choose a style, a mood and roles, then **Generate → Editor**. The same seed always gives the same song, and an empty seed gives a new one. |
| Add tracks to a song you have | **MIDI Generator → Layer over editor**. **Continue editor** extends the song past its end. |
| Change the tempo | Use the BPM box or **Tap tempo** in the **THEORY** tab. Decimals are allowed. |
| Save, open, export | Ctrl+S saves, Ctrl+O opens, Ctrl+E exports. Each song gets its own folder. |
| Select many tracks | Ctrl-click, Shift-click or Ctrl+A in the track list. Mute, solo, delete, colour, instrument, level and pan then apply to all of them as one undo step. |

# Features

- **Editor**: a lane per instrument with mute, solo and visibility, plus velocity, quantize, humanize, transpose and full undo. You get time signatures from 2/4 to 12/8, a grid that starts at the real first bar line, adaptive snapping, and pitch slides that stay with their notes.
- **Audio clips**: you can trim a clip by dragging its edges. Fixes work on new files, so the original recording is never changed.
- **Fix Clip Automatically**: splits a clip into vocals and music, splits each part at its pauses, cleans each part its own way, sets its level, and places it on the beat. Background noise or voice that the separation leaves behind is dropped. One undo brings back the original clip.
- **Tempo**: changing the tempo keeps notes and clips on their beats and stretches clips from their original recording. Tempo detection gives a decimal BPM, and the bar lines follow the real downbeat, including a pickup.
- **Conversion**: audio-to-MIDI uses basic-pitch, and stem separation uses Demucs plus an MDX vocal model. Drums are detected hit by hit into a General MIDI drum track.
- **Song generator**: rule-based and needs no download. It builds sections, key and mode, chord movement, mood and instrument changes from a seed, and can layer new tracks over an existing song.
- **Key detection**: names modes and other scales, such as Dorian or Maqam Hijaz, from where the music rests.
- **Mixing and export**: pan per track and per clip, BS.1770 loudness, a true-peak ceiling (-1 dBTP by default), TPDF dither for 16-bit files, and Smart Mastering that you can cancel.
- **Songs**: each song has a folder with its clips, MIDI, versions, backups, recordings and exports. Paths inside it are relative, so you can move the folder. A damaged song opens from its backups.

The changes in each version are listed in [CHANGELOG.md](CHANGELOG.md).

# Settings

Set these as environment variables before starting KORSHMER.

| Variable | Effect |
|----------|--------|
| `KORSHMER_SOUNDFONT` | Path to a different SoundFont |
| `KORSHMER_ENABLE_USER_PLUGINS=1` | Loads every plugin in `saved_data/plugins/` (see [Plugins](#plugins)) |
| `KORSHMER_HEALTH_PORT` | Starts the local [health server](#health-server) on this port |
| `KORSHMER_LOG_FORMAT=json` | One JSON object per log line (`time`, `level`, `logger`, `file`, `line`, `thread`, `message`, `exception`) |
| `KORSHMER_LOG_CONSOLE_LEVEL` / `KORSHMER_LOG_FILE_LEVEL` | Lowest level for the console and for `korshmer.log`: `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. The defaults are `INFO` and `DEBUG`. |
| `KORSHMER_LOG_INCLUDE` / `KORSHMER_LOG_EXCLUDE` | Comma-separated components to keep or drop, for example `korshmer.audio,korshmer.network` |
| `KORSHMER_OBS_PASSWORD`, `KORSHMER_TWITCH_TOKEN`, `KORSHMER_COPILOT_API_KEY` | Secrets. These are read before the system keychain. |

To keep a secret in the system keychain instead, run this and type the value at the hidden prompt:

```bash
python -m korshmer.utils.credential_store set OBS_PASSWORD
```

The same command works with `TWITCH_TOKEN` and `COPILOT_API_KEY`. Use `delete` instead of `set` to remove a secret.

## Health server

With `KORSHMER_HEALTH_PORT` set, KORSHMER answers HTTP requests on `127.0.0.1` only. A request whose `Host` header is not `127.0.0.1`, `localhost` or `[::1]` with that port gets HTTP 403.

| Endpoint | Returns |
|----------|---------|
| `GET /health` | `ok` and the uptime |
| `GET /health/state` | JSON with the status, heartbeat, playback, processing, CPU and memory. It returns HTTP 503 when the window has not updated for 5 seconds. |
| `GET /health/metrics` | Counts, errors and mean/p95/max milliseconds for file, separation, tempo and transcription work |
| `GET /health/traces` | Recent operations as trees of spans |
| `GET /health/logs` | The last 1000 log lines at INFO or above. It accepts `since`, `level`, `component` and `limit`. |
| `GET /health/logs/stream` | The same lines as Server-Sent Events, with up to 8 connections. A connection that stays silent for 30 seconds is closed. |

## Plugins

A plugin is Python code and runs with KORSHMER's own rights. Built-in plugins and plugins installed as packages always load. A file in `saved_data/plugins/` runs only in one of two cases:

1. It is listed in `saved_data/approved_plugins.json` with its SHA-256 checksum, for example `{"my_effect.py": "9f86d0..."}`. A file that changes after you approve it is skipped, and the log shows its new checksum.
2. KORSHMER was started with `KORSHMER_ENABLE_USER_PLUGINS=1`.

# File formats

| Import | Extensions | Needs |
|--------|------------|-------|
| WAV, FLAC, MP3, AIFF, AU | `.wav` `.flac` `.mp3` `.aif` `.aiff` `.au` | Nothing extra |
| OGG Vorbis / Opus | `.ogg` `.oga` `.opus` | Nothing extra. FFmpeg is needed for files that libsndfile cannot read. |
| AAC / M4A | `.m4a` `.aac` | FFmpeg |
| MIDI | `.mid` `.midi` | Nothing extra |

| Export | Options | Tags | Needs |
|--------|---------|------|-------|
| WAV | 16/24-bit PCM, 32-bit float | RIFF INFO | Nothing extra |
| FLAC | 16/24-bit, compression 0-8 | Vorbis comments | Nothing extra |
| OGG | Quality -1 to 10 | Vorbis comments | Nothing extra |
| MP3 | CBR/VBR, 64-320 kbps | ID3v2 | FFmpeg |
| AAC | 64-320 kbps (`.m4a`) | iTunes atoms | FFmpeg |

Audio with more than two channels is imported but exported as stereo or mono. DSD is not supported. When FFmpeg or a resample is missing, the export stops with an error and does not write a wrong file.

# Where files are stored

Everything is stored next to `src/`, or next to `korshmer.exe` in a compiled build.

| Path | Contents |
|------|----------|
| `saved_data/<song>/<song>.kproj` | A song, with `Clips`, `MIDI`, `Versions`, `Backups`, `Recordings` and `Exports` beside it |
| `saved_data/logs/` | `korshmer.log`, `audit.log` and `crash_reports/`. Full logs are rotated to `.gz` files. |
| `saved_data/plugins/`, `saved_data/approved_plugins.json` | Your plugins and the ones you approved |
| `saved_data/audio_host.json` | Output device and buffer size |
| `saved_data/window_layout.json`, `saved_data/shortcuts_user.json` | Window layout and your shortcuts |
| `temp/` | Working files. You can delete them while KORSHMER is closed. |
| `fluidsynth/` | FluidSynth for Windows and `soundfont.sf3` |
| `models/` | AI models. `compile.bat` downloads them, and when you run from source the vocal model is downloaded the first time a clip is split. |

Secrets are kept in the system keychain or in environment variables, never in these folders.

# Troubleshooting

| Problem | Fix |
|---------|-----|
| `install.bat` or `pip install` fails | Check that `python --version` is 3.10 or newer, run `python -m pip install --upgrade pip`, and check the internet connection (some packages come from GitHub). |
| The build fails with `lld-link: error: undefined symbol` | Install Visual Studio Build Tools with **Desktop development with C++** (see [Build korshmer.exe yourself](#build-korshmerexe-yourself)). |
| DLL errors when starting from source on Windows | Install the [Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist). |
| KORSHMER does not start | Run `pip install -e .` again in the active `.venv`, then read `saved_data/logs/korshmer.log`. |
| KORSHMER closed unexpectedly | Attach the newest file in `saved_data/logs/crash_reports/` to an issue. Native crashes are in `fatal-errors.log` in the same folder. |
| No sound, crackling or dropouts | Choose the right device and a larger buffer (512 or 1024) in the **Audio preferences** card, and stop running conversions. |
| MIDI is silent | Check that `fluidsynth/soundfont.sf3` exists (or set `KORSHMER_SOUNDFONT`), and on Linux/macOS that the `fluidsynth` package is installed. |
| A drumless song gets a drum track | Untick **Drums** in the conversion options. |
| A file does not import | Install FFmpeg for AAC/M4A, or convert the file to WAV/FLAC/MP3. Move it to a path without special characters. |
| Out of memory | Close other programs, process shorter audio, and keep 8 GB or more free for the AI tools. |
| Songs do not save | Make sure `saved_data/` is writable and has free space. The folder must not be under `Program Files`. |
| macOS blocks the app | Allow it in **System Settings → Privacy & Security**. |
| Linux has no audio device access | Add yourself to the audio group with `sudo usermod -a -G audio $USER`, then log in again. |

# Testing and development

Setup, tests and style rules are in [CONTRIBUTING.md](CONTRIBUTING.md). In short:

```bash
pip install -e ".[dev]"
QT_QPA_PLATFORM=offscreen pytest -q tests/unit
ruff format --check .
ruff check .
poe check
```

`poe check` runs the full gate: compile, dead code, comment stripping, the pre-commit hooks, mypy and all the tests. On Windows, `scripts\check.bat` runs the same gate.

Report vulnerabilities as described in [SECURITY.md](SECURITY.md).

# License and Third-Party Licenses

KORSHMER is released under the [MIT License](LICENSE). The bundled files in `fluidsynth/` keep their own licences. Include those licences when you share a build:

| File | Project | License |
|------|---------|---------|
| `fluidsynth.exe`, `libfluidsynth-3.dll` | [FluidSynth](https://github.com/FluidSynth/fluidsynth) | LGPL-2.1-or-later |
| `SDL3.dll` | [SDL](https://www.libsdl.org/) | zlib |
| `sndfile.dll` | [libsndfile](https://github.com/libsndfile/libsndfile) | LGPL-2.1-or-later |
| `soundfont.sf3` | MuseScore General | See `fluidsynth/MuseScore_General_License.md` |

FFmpeg runs as a separate program under its own LGPL or GPL licence. The Python dependencies keep their own licences.
