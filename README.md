# KORSHMER | Redefining the DAW.

![Version](https://img.shields.io/badge/Version-1.0.0-red?style=flat-square)

KORSHMER is a next-generation audio workstation (DAW), combining studio production and score editing in one interface. The system is built with Python and a unique hardware-inspired user interface, designed in a futuristic style.

# Key Features

* Hybrid Workflow: A unique combination of a linear timeline (DAW) and a classic score editor (Score Editor).
* Multi-Lane MIDI Editor: Every instrument lives on its own colored lane with mute/solo/visibility, instrument selection, snapping grid, velocity editing, copy/paste/duplicate, quantize, humanize, transpose and full undo/redo.
* Vocals & Stems On The Timeline: After stem separation the vocals are cleaned of separation leftovers (the original stem is kept untouched) and appear as a draggable audio clip, set to a loudness just under the MIDI render so it does not mask the instruments; instrument stems load as MIDI lanes, and the editor plays and exports everything mixed together.
* MIDI Variations: The Edit tool Create variation saves a new MIDI built on the selected one (Light, Medium or Strong): the first notes of every second bar and the longest notes stay, the rest moves along the key with passing notes, shifted starts and new chord voicings, so it feels like another song that still leans on the original. Drums are never changed.
* Song Builder: One click generates a complete, section-aware arrangement (drums, bass, chords, melody, pad, arp) in the chosen key, style and tempo, straight into the editor. The Layer over editor mode adds new tracks on top of the song already in the editor: the generator reads the chord of every bar and the quiet and loud parts (intro, verse, chorus, break, outro) from its notes, and the generator itself decides how closely the new tracks follow those chords, from how much of the song plays and how fast its chords change. Long songs follow a story of acts: instruments and drum kits change between acts, the music moves through other modes made of the same notes of your key, a quiet passage leads to a climax, and the opening sound returns at the end. Auto key detection (for MIDI, clips and audio) names modes and other scales such as Lydian, Dorian or Arabian (Maqam Hijaz), not only Major and Minor.
* Real Drum Tracks: The drums stem is not run through a pitched model. Every hit is detected as an onset in the low, middle and high bands and named by its spectrum (kick, snare, toms, closed and open hi-hat, crash), so the converted MIDI has a real General MIDI drum track.
* AI-Powered Core: Advanced audio-to-MIDI conversion, stem separation using Demucs, and generative music creation (MusicGen).
* Next-Gen UI: Custom user interface (without standard Windows widgets) with hardware acceleration, including realistic buttons, faders and dynamic visuals.
* Advanced DSP: Real-time sound engine including spectral analysis (K-Spectrum), multi-channel dynamic processing (Smart Master), and unique parametric EQ.
* Exact Tempo: The tempo is a decimal BPM (for example 123.4) everywhere: the tempo box, tap tempo, Ableton Link, and the automatic analysis of a song, which refines the tempo from the beat positions instead of rounding it.
* Real Bar Lines: The bar grid follows the real downbeat of the song instead of assuming that the audio starts on a bar line. The beat phase is found from the onsets, the bar line is the beat with the biggest accent (bass and loudness), a pickup (anacrusis) before the first bar line is reported in beats and seconds, leading silence is not counted as a pickup, and a song that stops in the middle of a bar reports how many beats its last bar holds.
* Chord Sheet With Pickup: The chord sheet cuts its bars from the real downbeat found in the audio, lists the chords of a pickup under their own Pickup heading before Bar 1, keeps every bar the exact length of the tempo (no drift over long songs), and marks a last bar that stops early, for example Bar 5 (2 of 4 beats).
* Piano Roll Grid Offset: When a MIDI is loaded the editor finds where its first bar line really is. The grid lines, the bar numbers, the snapping of notes and the quantize command start from that offset (it is saved with the project) instead of from time zero, so a song with a pickup keeps its notes on the beat.
* Adaptive Snapping: At the end of every audio-to-MIDI conversion each melodic note is snapped to the grid step that suits it: long notes to quarter steps, medium notes to eighths, short notes to sixteenths. A coarse step is taken only when the note is already within 0.08 beats of it, so real syncopation stays. The Grid setting is the finest step allowed. Drum hits are kept at the onsets they were detected at rather than snapped, so the drum part keeps the source's own timing (the beat grid is still read from the kick and snare).
* Groove Control: Advanced control over groove and rhythm, with an emphasis on a "human" feel in sequencing.

# Tech Stack

The project is written in Python 3.10+ and uses:
* UI/UX: `PySide6` (Qt for Python) for graphics rendering.
* Audio Processing: `librosa`, `numpy`, `scipy` for digital signal processing (DSP).
* AI & ML: `torch`, `torchaudio`, `demucs`, `basic-pitch` for AI tasks.
* Playback: `sounddevice`, `soundfile` and an interface to `FluidSynth`.

# System Requirements
* Python 3.10 or higher.
* System libraries (such as Visual C++ Redistributable on Windows).

# Installation
1. Clone the repository:
```bash
git clone [https://github.com/YaronKoresh/korshmer.git](https://github.com/YaronKoresh/korshmer.git)
cd korshmer
```
2. Run the automatic installation script (Windows):
```cmd
scripts/install.bat
```
*Or manually install the dependencies:*
```bash
pip install -e .
```

# Run
To run the software, use the run script:
```cmd
scripts/run.bat
```

Or run directly via Python:

```bash
python -m korshmer
```

# Test

Install test dependencies:
```bash
pip install -e ".[dev]"
```

Run all tests:
```bash
pytest
```

Run unit tests only:
```bash
pytest tests/unit -v
```

Run tests with coverage:
```bash
pytest --cov=src/korshmer --cov-report=html
```

Run specific test file:
```bash
pytest tests/unit/test_piano_roll.py -v
```

# Code Quality

Check code formatting:
```bash
ruff format --check src/korshmer tests
```

Format code:
```bash
ruff format src/korshmer tests
```

Run linter:
```bash
ruff check src/korshmer
```

Run type checker:
```bash
mypy src/korshmer --ignore-missing-imports
```

# FAQ

Q: What platforms does KORSHMER support?
A: KORSHMER runs on Windows 10/11, macOS (Intel and Apple Silicon), and Linux (Ubuntu, Fedora, Arch). Python 3.10 or higher is required.

Q: Do I need any special hardware?
A: No special hardware is required, but a MIDI controller is recommended for the best experience. Any audio interface supported by your OS will work.

Q: Can I use KORSHMER for commercial projects?
A: Please contact the developer for licensing information regarding commercial use.

Q: The installation fails with "Visual C++ Redistributable" error (Windows)
A: Download and install the latest Microsoft Visual C++ Redistributable from the official Microsoft website.

Q: I get "FluidSynth not found" errors
A: Ensure FluidSynth is installed on your system. On Windows, the FluidSynth DLL should be in the `fluidsynth/` folder. On Linux: `sudo apt install fluidsynth`. On macOS: `brew install fluid-synth`.

Q: The SoundFont file is missing
A: Download a GM-compatible SoundFont (`.sf3` file), name it `soundfont.sf3` and place it in the `fluidsynth/` directory, or set the `KORSHMER_SOUNDFONT` environment variable to its path. Free options include [MuseScore General](https://musescore.org/en/handbook/3/soundfonts-and-sfz-files) (MIT) and FluidR3_GM.sf3. See [Replacing or upgrading the SoundFont](#replacing-or-upgrading-the-soundfont).

Q: How do I convert audio to MIDI?
A: Load an audio file, then go to the Audio Tools tab and click "Audio to MIDI" or "Audio to MIDI (basic-pitch)". The conversion may take several minutes depending on file length.

Q: Can I import my existing MIDI files?
A: Yes, use File → Import → MIDI File or drag-and-drop MIDI files directly into the application.

Q: How do I change the BPM?
A: Click the BPM button in the Theory Tab to adjust tempo. This is the master tempo for the entire project and overrides imported file tempos.

Q: Why is my audio playback stuttering?
A: Try increasing the audio buffer size in Settings. Close other CPU-intensive applications. Consider freezing/rendering heavy tracks.

Q: How do I separate vocals from a song?
A: Select the audio file and press Separate Stems (BUILDER tab or Audio Tools). The stems (vocals, drums, bass, guitar, piano, other) are saved next to the output and you are offered to load them into the editor: the vocals become a draggable audio clip and the instruments become MIDI lanes. This requires significant processing time and memory.

Q: How do I put the separated vocals on the MIDI timeline?
A: Accept the "Load into editor" prompt after separation, press the 🎤 button next to any audio file in the output list, or drag the file from the output list onto the editor. The clip sits in the audio lane above the notes; drag it to move it in time and double-click it to mute. Play Editor and Export Mix render the MIDI lanes together with the clips.

Q: How do I build a full song quickly?
A: Open the MIDI Generator, choose a style and mood, tick the roles you want, and press Generate → Editor. Key and tempo come from the THEORY tab (Auto detects them from the editor).

Q: How do I add new tracks to a song that is already in the editor?
A: In the MIDI Generator choose the mode Layer over editor and tick the roles to add. The length, the chords and the parts of the song (intro, verse, chorus, break, outro) are read from the notes in the editor, so the new tracks are as long as the song and change with it. How closely they follow its chords is decided by the generator: a song whose chords change from bar to bar is followed bar by bar, a song with slower chords is followed with one chord per two bars, and a song that is mostly silent gets chords of its own in the same key. A song that already has drums gets no new drum track, and a generated bass locks to its kicks. Continue editor instead extends the song after its end, starting on the next real bar line (a song with a pickup or one that stops in the middle of a bar is continued on its own bar grid, not on a grid counted from zero).

Q: Why does the converted MIDI have little bass?
A: The bass stem is transcribed once with softer model thresholds and its repeated notes are kept apart: a note is cut where the bass audio has a new attack and equal notes are not merged into one long note.

Q: The song has no drums, but the converted MIDI added a drum track. How do I stop that?
A: Untick Drums in the conversion options before converting. Stem separation always produces a "drums" stem, and on a song with no drums that stem is only faint bleed from the other instruments; because every instrument onset looks like a hit, converting it can invent a dense drum part. There is no reliable way to tell a drumless bleed stem from a real but quiet or sparse one automatically — doing so would either erase real drums on other songs or keep inventing them — so the choice is left to you: leave Drums ticked for songs that have drums, untick it for songs that do not. When Drums is ticked, the detected hits are kept at their own onset positions (not snapped to the grid), so a real drum part matches the source instead of being pulled onto a mechanical every-eighth grid.

Q: How can a script check that KORSHMER is running and responsive?
A: Start KORSHMER with `KORSHMER_HEALTH_PORT` set to a free port (for example `KORSHMER_HEALTH_PORT=8765 python -m korshmer`). It then answers HTTP requests on `127.0.0.1` only: `GET /health` returns status `ok` and the uptime while the app is alive, and `GET /health/state` returns JSON with the overall status, uptime, the main-thread heartbeat, playback and processing state, CPU and memory figures and whether OSC, OBS and Ableton Link are active. `/health/state` returns HTTP 503 when the interface has not updated for more than 5 seconds. `GET /health/metrics` returns call counts, error counts and mean, p95 and max durations in milliseconds for project file operations, stem separation, tempo detection and audio-to-MIDI transcription. `GET /health/traces` returns the most recent traces: each operation that is timed for the metrics, together with the work it started on other threads through the shared executors and the processing queue, appears as a tree of spans with trace id, span id, parent id, thread, duration and error text. Without the variable no port is opened. With the variable set, the application keeps its most recent 1000 log lines of level INFO and above in memory. `GET /health/logs` returns them as JSON, with `last_seq` and a `dropped` flag that is true when lines were lost before the requested position; pass `since=<last_seq>` to receive only newer lines, `level=WARNING` for a minimum severity, `component=korshmer.audio` for one component and `limit=<n>` (at most 1000). `GET /health/logs/stream` keeps the connection open and pushes every new line as a Server-Sent Event (`id` is the line number, `data` is the same JSON object); a monitoring dashboard can connect with the browser `EventSource` API, which resumes from the last line after a dropped connection, and the same `level` and `component` filters apply. Every response of the health server, including error pages and the event stream, carries a `Content-Security-Policy` header that allows no content sources and no framing (`default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`), together with `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` and `Cross-Origin-Resource-Policy: same-origin`.

Q: How can a script or log collector read the logs?
A: Start KORSHMER with `KORSHMER_LOG_FORMAT=json`. The console and `saved_data/logs/korshmer.log` then get one JSON object per line with the keys `time`, `level`, `logger`, `file`, `line`, `thread` and `message`, plus `exception` when a stack trace is attached. Any other value, or no value, keeps the plain text format. `audit.log` keeps its own format.

Q: How can I show fewer log lines, or only some parts of the app?
A: `KORSHMER_LOG_CONSOLE_LEVEL` and `KORSHMER_LOG_FILE_LEVEL` set the minimum severity (`DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`) for the console and for `korshmer.log`; the defaults are `INFO` and `DEBUG`. `KORSHMER_LOG_INCLUDE` and `KORSHMER_LOG_EXCLUDE` take comma-separated component names such as `korshmer.audio,korshmer.network`; a name also matches everything under it. With an include list only those components are logged, and anything in the exclude list is dropped. The component is the logger name, or the module that wrote the line. Both filters apply to the console and `korshmer.log`, not to `audit.log`. Invalid levels are ignored with a warning.

Q: Where do I keep the OBS password, Twitch token or AI assistant key?
A: In the system keychain. Run `python -m korshmer.utils.credential_store set OBS_PASSWORD` (or `TWITCH_TOKEN`, `COPILOT_API_KEY`) and type the value at the hidden prompt; `delete` removes it. The matching environment variable (`KORSHMER_OBS_PASSWORD`, `KORSHMER_TWITCH_TOKEN`, `KORSHMER_COPILOT_API_KEY`) is read first and wins over the keychain. Without a usable keychain only the environment variables work.

Q: The application won't start
A: Check that Python 3.10+ is installed and all dependencies are present. Run `pip install -e .` to reinstall dependencies. Check the logs in `saved_data/logs/korshmer.log`. When `korshmer.log` or `audit.log` reaches its size limit it is rotated into a gzip-compressed backup such as `korshmer.log.1.gz`.

Q: KORSHMER closed unexpectedly. Where is the error?
A: Every unhandled error writes a crash report with the full stack trace, the version, the Python version, the platform and the thread name to `saved_data/logs/crash_reports/crash-<date>-<time>-<pid>-<n>.txt`, and the same trace is written to `saved_data/logs/korshmer.log`. The newest 20 reports are kept. Hard crashes inside native libraries (for example a segmentation fault) are dumped to `saved_data/logs/crash_reports/fatal-errors.log`. Attach the report when you open an issue.

Q: I'm getting "Out of Memory" errors
A: Close other applications to free up RAM. Consider processing shorter audio segments. Ensure you have at least 8GB of RAM available for AI operations.

Q: MIDI playback has no sound
A: Verify FluidSynth is properly installed and the SoundFont file exists at the configured path. Check audio device settings in your system.

Q: My changes aren't being saved
A: Ensure the `saved_data/` directory exists and is writable. Check disk space availability. Use File → Save Project explicitly.

# Troubleshooting

## No sound during playback
1. Check audio device: Verify your audio output device is properly connected and selected as the system default.
2. Check volume levels: Ensure master volume and track volumes are not muted or at zero.
3. Buffer underruns: Increase audio buffer size in settings (Settings → Audio → Buffer Size).
4. Driver issues: Update your audio drivers to the latest version.
5. Exclusive mode: Close other applications that might be using the audio device exclusively.

## Crackling or distorted audio
1. CPU overload: Reduce the number of active effects or freeze some tracks.
2. Buffer size too small: Increase buffer size to 512 or 1024 samples.
3. Sample rate mismatch: Ensure project sample rate matches your audio interface settings.
4. Clipping: Check for red peak indicators and reduce gain or enable limiting.

## MIDI playback is silent
1. FluidSynth not loaded: Check logs for FluidSynth initialization errors.
2. SoundFont missing: Verify `.sf3` file exists in the correct location.
3. MIDI channel muted: Unmute MIDI tracks in the mixer.
4. No MIDI data: Ensure MIDI notes exist in the piano roll.

## Application is slow or freezing
1. Close heavy operations: Cancel any running AI operations (stem separation, transcription).
2. Reduce visual updates: Disable real-time waveform visualization during playback.
3. Clear cache: Delete temporary files in the `temp/` directory.
4. Update dependencies: Run `pip install --upgrade -e .` to ensure latest optimized versions.

## High CPU usage
1. Freeze tracks: Render tracks with heavy effects to audio.
2. Limit polyphony: Reduce the number of simultaneous MIDI notes.
3. Disable unnecessary effects: Bypass effects that aren't needed.
4. Check background processes: Close other CPU-intensive applications.

## Memory leaks / RAM usage grows over time
1. Restart application: Long sessions may accumulate cached data.
2. Clear undo history: Edit → Clear Undo History (if available).
3. Unload unused audio: Close projects or stems you're not actively using.

## Cannot import audio files
1. Unsupported format: Convert to WAV, FLAC, or MP3 using a tool like Audacity or ffmpeg.
2. File corruption: Try opening the file in another audio editor to verify integrity.
3. File permissions: Ensure you have read permissions for the file.
4. Path contains special characters: Move file to a simpler path without Unicode or special characters.

## Cannot export audio
1. Insufficient disk space: Free up space on your drive.
2. Write permissions: Ensure the output directory is writable.
3. File in use: Close any applications that might be using the output file.

## MIDI import fails
1. Corrupted MIDI file: Validate the MIDI file with a MIDI validator tool.
2. Non-standard MIDI: Some MIDI files use proprietary extensions; try re-exporting from the source DAW.
3. Large MIDI files: Files with tens of thousands of notes may take time to process.

## `pip install` fails
1. Python version: Ensure Python 3.10 or higher is installed (`python --version`).
2. pip outdated: Update pip: `python -m pip install --upgrade pip`.
3. Network issues: Check internet connection; some packages are downloaded from GitHub.
4. Visual C++ missing (Windows): Install Microsoft Visual C++ 14.0 or greater.
5. Permission denied: Use `pip install --user -e .` or run with administrator privileges.

## Missing dependencies at runtime
1. Reinstall: Run `pip install -e .` again to reinstall all dependencies.
2. Virtual environment: Ensure you're activating the correct Python environment.
3. System libraries: Some packages require system-level libraries (e.g., libsndfile for soundfile).

## Windows
- DLL not found errors: Install Visual C++ Redistributable and ensure FluidSynth DLLs are in the correct folder.
- Long path issues: Enable long path support in Windows or use shorter project paths.

## macOS
- "Unsigned developer" warnings: Allow the application in System Preferences → Security & Privacy.
- Signing a release build: `python scripts/build_all.py --pyinstaller --sign` signs `dist/korshmer`. On macOS set `KORSHMER_MACOS_IDENTITY` (a Developer ID Application identity). On Windows set `KORSHMER_WIN_CERT_FILE` and `KORSHMER_WIN_CERT_PASSWORD`, and optionally `KORSHMER_TIMESTAMP_URL`. The build fails if signing or verification fails.
- Apple Silicon compatibility: Use Rosetta 2 or install ARM-compatible Python packages.

## Linux
- Missing system libraries: Install: `sudo apt install libsndfile1 fluidsynth portaudio19-dev`
- Permission issues: Ensure user is in the `audio` group: `sudo usermod -a -G audio $USER`

# Audio Format Compatibility

## Import Formats (Read)

| Format | Extension | Bit Depth | Sample Rate | Channels | Status |
|--------|-----------|-----------|-------------|----------|--------|
| WAV | .wav | 16/24/32-bit PCM, 32/64-bit float | Any | Mono/Stereo/Multi | ✅ Full support |
| FLAC | .flac | 16/24-bit | 8-192 kHz | Mono/Stereo/Multi | ✅ Full support |
| MP3 | .mp3 | N/A (lossy) | 8-48 kHz | Mono/Stereo | ✅ Full support |
| OGG Vorbis | .ogg | N/A (lossy) | Any | Mono/Stereo | ✅ Full support |
| AAC/M4A | .m4a, .aac | N/A (lossy) | Any | Mono/Stereo | ✅ Via ffmpeg |
| AIFF | .aif, .aiff | 16/24/32-bit | Any | Mono/Stereo/Multi | ✅ Full support |
| AU | .au | 8/16/24/32-bit | Any | Mono/Stereo | ✅ Full support |

## Export Formats (Write)

| Format | Extension | Bit Depth Options | Quality Settings | Metadata | Status |
|--------|-----------|-------------------|------------------|----------|--------|
| WAV | .wav | 16/24/32-bit PCM | Lossless | ✅ Yes | ✅ Full support |
| FLAC | .flac | 16/24-bit | Compression 0-8 | ✅ Yes | ✅ Full support |
| MP3 | .mp3 | N/A | CBR/VBR, 64-320 kbps | ✅ ID3v2 | ✅ Full support |
| OGG | .ogg | N/A | Quality -1 to 10 | ✅ Vorbis Comments | ✅ Full support |
| AAC | .m4a | N/A | 64-320 kbps | ✅ Yes | ⚠️ Via ffmpeg |

# Known Limitations

- Multi-channel audio (>2) are Supported for import but exported as stereo or mono
- DSD formats are not supported; convert to PCM first
- Very long files may require increased memory allocation

# External Dependencies

| Library | Purpose | Required For |
|---------|---------|--------------|
| libsndfile | WAV, FLAC, AIFF, AU | Core audio I/O |
| soundfile | Python interface to libsndfile | Core functionality |
| librosa | MP3, advanced processing | Audio analysis |
| ffmpeg | AAC, M4A, video extraction | Optional formats |

# Metadata Support

| Format | Supported Tags |
|--------|----------------|
| WAV | RIFF INFO chunks (title, artist, etc.) |
| FLAC | Vorbis comments (full tagging) |
| MP3 | ID3v2 (title, artist, album, genre, year, cover art) |
| OGG | Vorbis comments (full tagging) |
| M4A | iTunes-style atoms (full tagging) |

# Third-Party Licenses (`fluidsynth/` directory)

The `fluidsynth/` directory contains pre-built binaries and assets from third-party projects, each distributed under its own license:

| File | Project | License |
|------|---------|---------|
| `fluidsynth.exe` | [FluidSynth](https://github.com/FluidSynth/fluidsynth) | LGPL-2.1-or-later |
| `libfluidsynth-3.dll` | [FluidSynth](https://github.com/FluidSynth/fluidsynth) | LGPL-2.1-or-later |
| `SDL3.dll` | [SDL (Simple DirectMedia Layer)](https://www.libsdl.org/) | zlib License |
| `sndfile.dll` | [libsndfile](https://github.com/libsndfile/libsndfile) | LGPL-2.1-or-later |
| `soundfont.sf3` | [MuseScore General License](./fluidsynth/MuseScore_General_License.md) | Custom License |

---

*Created by Yaron, with ❤️ and Python.*