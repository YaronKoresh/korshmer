# Changelog

All notable changes to KORSHMER are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Editor: time signatures 2/4, 3/4, 4/4, 5/4, 6/8, 7/8 and 12/8, drawn in the grid and saved with the song (in the x/8 signatures the tempo beat is the eighth note); a Follow playhead toggle; Zoom to fit.
- Transport and window: Ctrl+S saves the song; the time display cycles through mm:ss, milliseconds, h:mm:ss and bars|beats on click; shortcuts Escape, Home, Ctrl+N, Ctrl+O and Ctrl+E; a Keyboard Shortcuts window generated from the real bindings; custom shortcuts loaded from `saved_data/shortcuts_user.json`; the window layout is remembered in `saved_data/window_layout.json`; files can be dropped anywhere on the window.
- Audio preferences card: output device and buffer size, saved in `saved_data/audio_host.json`.
- Mixer: pan per track and per audio clip (equal-power), written to and read from MIDI as CC 10; level and pan gestures can be undone.
- Export: true-peak ceiling (default -1 dBTP, export preset key `true_peak_ceiling`); TPDF dither for 16-bit files (preset keys `dither` and `dither_seed`).
- Songs: project files carry `format_version` 2 with migration of older files, validation, and a clear error for files written by a newer version; a damaged song opens from the backups beside it.
- Collaboration: starting a session starts a real local session server and reports its address, link and participants; edits are sent to the musicians who joined; the client and the manager accept a shared secret token. Features the server does not have (joining a remote session, chat, resource locks, edit callbacks) report themselves as unavailable.
- Activity reports can be exported as CSV as well as JSON.
- `.m4a`, `.aac` and `.au` files are accepted by the file list; AAC and M4A are converted with FFmpeg when added.
- Cloud sync status carries the last configuration error.
- `CONTRIBUTING.md`, `SECURITY.md`, this changelog, and a GitHub Actions workflow: ruff, mypy and the unit, integration and regression tests on Python 3.10 to 3.13, with the wall-clock benchmarks in a separate job that cannot block a change.
- Package metadata: keywords, classifiers, documentation and changelog links, SPDX licence expression with the `LICENSE` file shipped in the wheel; optional dependency groups `test`, `lint` and `build`.

### Changed

- Startup: the transcription library is loaded the first time a conversion needs it, so the application opens even when that library cannot load; the model download script no longer loads it on import.
- Separated vocals start 1.5 dB under the notes (was 2.5 dB).
- Editor drawing: a moving playhead repaints only the strips that changed and no longer sends note-change notifications; fonts, colours and waveform shapes are cached; clearing or loading the editor no longer waits for waveform analysis.
- Window: a tempo change is handled once; the beat light, meters and visualizer rest when nothing plays; the CPU reading is measured; reveal-in-folder works on Windows, macOS and Linux; background jobs are stopped when the window closes.
- Audio engine: no logging or allocation in the audio callback; large streamed files are no longer read from disk inside the callback; the looper saves layers off the audio thread; the preferred buffer size and output device are honoured.
- Export: loudness is measured as BS.1770 LUFS; peaks are held under the true-peak ceiling instead of being clipped; a failed resample or a missing FFmpeg fails the export job and sets `job["error"]`.
- `save_audio` writes the format of the requested extension, returns the path it wrote, and writes plain WAV below 4 GB.
- Compressor ratio is applied in dB with linked stereo; multiband crossovers sum flat (Linkwitz-Riley).
- Mixer: track rows are updated in place; renders are no longer peak-boosted when saved.
- Backups contain the current unsaved work; audio that is already compressed is stored uncompressed in the song archive.
- The AI assistant picks a default model per provider (`llama3` for Ollama, `gpt-4o-mini` for OpenAI, `claude-opus-5-5` for Anthropic) instead of sending `llama3` to every provider; a model chosen explicitly is kept.
- Heavy conversions (stem separation and transcription) run on a pool sized by processor cores and installed memory (about 6 GB per job) instead of one job per core.
- An unknown activity report format raises an error that names the supported formats.
- `scripts/build_all.py` exits with a non-zero code whenever a wheel or PyInstaller build fails, not only with `--sign`.
- README: corrected the licence answer (MIT, commercial use allowed), the clone command, menu and control names, format tables, FluidSynth and SoundFont instructions and the Linux and macOS steps; added sections on what is new, plugin safety, data locations and the licence.
- Removed the unused `PyWavelets` dependency, the unused `pydantic` development dependency, and the Briefcase `icon` setting that pointed at a missing file.

### Fixed

- Smart mastering now keeps to the memory it asks for on any computer: it no longer starts one band per processor core, so a song accepted as fitting in free memory can no longer use several times more on a machine with many cores. The result is unchanged.
- Notes that use exactly the notes of a major scale with no clear home note are now named by their relative minor in every key and at any loudness, not only in some keys.
- Playback position is clamped at the end of the song.
- Recorder failures are reported instead of being lost.
- Exporting a folder can no longer delete its own source.
- The Anthropic provider of the AI assistant sends the required `anthropic-version` header and reads the answer when the reply starts with a thinking block.
- Two songs with the same file name from different folders no longer share, and delete, each other's working folder during conversion; a conversion stopped at the very start leaves no working folder, and folders left by a crash are removed after a day.
- Silence trimming before stem separation keeps stereo songs in stereo (through every name the function is reachable by), reports problems in the log instead of printing them, and no longer copies a file it cannot decode under a `.wav` name.
- Cloud sync "sync all" finds songs saved as `saved_data/<name>/<name>.kproj` (songs in the old `saved_data/projects` folder are still included), and connecting to an unsupported provider says which providers are supported.

### Security

- Third-party plugin files in `saved_data/plugins/` no longer run automatically. They load only when listed with their SHA-256 checksum in `saved_data/approved_plugins.json` (kept beside the plugin folder, not inside it), or when `KORSHMER_ENABLE_USER_PLUGINS=1` is set. Skipped files are named in the log together with their checksum.
- The health server answers only requests whose `Host` header names this computer with its own port (HTTP 403 otherwise), which stops DNS-rebinding web pages from reading logs and traces. The live log stream serves at most 8 clients at once (HTTP 503 above that) and silent connections are dropped after 30 seconds.
- The collaboration server listens on `127.0.0.1` by default, refuses any other address without a shared secret token, compares the token in constant time, and refuses connections whose browser `Origin` is not this computer.
- A relative clip path in a song file is opened only when it stays inside the song folder (also after following links). A refused path stays in the song file and a status message names it. Absolute paths work as before.
- The processing cache stores entries as NumPy `.npz` data without `pickle`.
- Audio extraction from video runs FFmpeg with `-nostdin`, a 10-minute time limit and the file passed as `file:<absolute path>`.
- YouTube and Twitch streams use encrypted `rtmps://` endpoints (Facebook already did), and stream keys are masked in logged commands. The key is still on the FFmpeg command line and visible to local processes.
- `scripts/install.bat` runs the downloaded Python, LLVM and Windows SDK installers only when their Authenticode signature is valid and names the expected publisher.

### Migration notes

- Smart mastering estimates its peak memory as 15 copies of the song instead of 12, and works on at most two bands at once, so a song slightly shorter than before may be refused when free memory is low.
Plugins and servers

- Plugins in `saved_data/plugins/` are off until you set `KORSHMER_ENABLE_USER_PLUGINS=1` or list each file as `{"file.py": "<sha256>"}` in `saved_data/approved_plugins.json`.
- Health server clients must send a `Host` header of `127.0.0.1`, `localhost`, `[::1]` or the bound address, with the port (the port may be left out only on port 80). At most 8 clients of `/health/logs/stream`; idle sockets close after 30 seconds.
- `CollaborationServer`: the default `host` changed from `0.0.0.0` to `127.0.0.1`; any other host needs the new `token` argument, and clients pass it as `?token=` in the address or as `Authorization: Bearer` (`CollaborationClient(..., token=...)`). Browser clients must be served from this computer. `server.port` holds the real port after `start()`.
- `CollaborationManager(base_dir, host, port, token)`: `start_session` now returns a session link and opens a port; `get_session_status()` has the new keys `name`, `address`, `url`, `token_required` and `unavailable`.
- Streaming: `youtube_live.INGEST_URL` is `rtmps://a.rtmps.youtube.com:443/live2` and Twitch uses `rtmps://live.twitch.tv:443/app`; the endpoint table `PLATFORM_INGEST` now lives in `korshmer.network.youtube_live` (still importable from `multi_streamer`).
- The README example for `KORSHMER_HEALTH_PORT` is now 8787, so it no longer collides with the collaboration server's default port 8765.

Files and formats

- Song files are written with `format_version` 2. Older files are migrated when opened; a file from a newer version is refused with a clear error.
- New song keys: time signature, pan per track and per audio clip. MIDI export writes pan as CC 10 and import reads it.
- New settings files: `saved_data/audio_host.json`, `saved_data/window_layout.json`, `saved_data/shortcuts_user.json`, `saved_data/approved_plugins.json`.
- Export preset keys: `true_peak_ceiling` (default -1 dBTP), `dither`, `dither_seed`. Loudness targets now mean BS.1770 LUFS, so the same target gives a different level than before.
- A failed resample or missing FFmpeg now fails the export job (`job["error"]`) instead of producing a file.
- Relative clip paths that leave the song folder are no longer opened.
- The processing cache format changed from `.pkl` to `.npz`; old entries are deleted at start, and only plain data and NumPy arrays can be cached.
- The file list accepts `.m4a`, `.aac` and `.au`.

Defaults

- Separated vocals start at -1.5 dB relative to the notes (was -2.5 dB).
- Saved renders are no longer peak-boosted.
- Compressor ratio is in dB with linked stereo, and multiband crossovers are Linkwitz-Riley, so presets sound different.
- In 6/8, 7/8 and 12/8 the tempo counts eighth notes.

Functions and commands

- `save_audio` returns the written path and follows the requested extension.
- `trim_audio_silence` keeps the channel layout and may return a path that does not end in `.wav` for a file it cannot decode.
- `MusicCopilot(model=...)` defaults to `None`, meaning the provider's default; the Anthropic request uses API version `2026-10-21` and `max_tokens` 16000.
- `AnalyticsManager.export_report` raises `ValueError` for an unknown format (it used to return `False`) and accepts `"csv"`.
- `CloudSyncManager.get_sync_status()` has the new key `last_error`.
- `ImportManager.extract_audio_from_video` gives up after 10 minutes.
- Conversion working folders are named `saved_data/_temp_work/<file name>_<random>`.
- Extras: `pip install -e ".[dev]"` still installs everything; `test`, `lint` and `build` can be installed on their own. Building needs hatchling 1.27 or newer.
- `scripts/build_all.py` exits 1 on any failed build; `scripts/install.bat` stops when an installer's signature is invalid or from an unexpected publisher.

## [1.0.0]

- First stable release.
