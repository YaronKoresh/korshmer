# Changelog

All notable changes to KORSHMER are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `korshmer.exe` built with `scripts\compile.bat` starts again: the build tells Nuitka to leave out the step in which PyTorch renames its functions, which stopped the program at start with `RuntimeError: __code__ is not writable in Nuitka`, and it always installs Nuitka 4.3.5, so an older Nuitka no longer leaves Python 3.14 builds printing `'import warnings' failed` at start.

## [1.1.1] - 2026-11-27

### Changed

- The README test that checks the script names it mentions accepts Windows paths such as `scripts\install.bat` as well as `scripts/install.bat`.

### Fixed

- The pull request bot asks for a major release when the title marks a breaking change with `!` (for example `feat(api)!:`) or the description has a `BREAKING CHANGE:` line, and labels the pull request `breaking-change`; before, such a pull request got a minor release or none.
- The pull request bot labels a `perf` pull request that changes source files `performance` and includes it in the next release as a patch; before, it got no labels and was left out of the release.
- Fixing a single audio clip that has silence at its ends keeps 0.5 s of the room sound around its sound, as the parts of a split clip do, instead of trimming it to 0.05 s from the sound; an end is trimmed only when that removes at least 0.1 s.
- The pull request bot decides whether a pull request goes into a release, and as which version step, in one place from all of its evidence: the change type and `!` mark in the title and in every commit, `BREAKING CHANGE:` lines in the description and in the commits, the section of each new changelog entry (Added and Deprecated are a minor release, Removed a major one, Fixed, Changed and Security a patch), and, only when none of these say anything, the branch type, the title wording and changed tests. A declared `refactor`, `docs`, `test`, `style`, `build`, `ci` or `chore` type keeps a pull request out of the release unless a commit or a changelog entry records a change, and the summary lists the evidence for the decision and any conflict. Before, a source change whose title had no change type or fix wording was left out of the release.
- The pull request bot names the kind of change in its label (`bug`, `enhancement`, `performance`, `security` or `breaking-change`) instead of the generic `runtime` label, and its label explanations no longer show a confidence level.
- The pull request bot labels `security` when the source adds a protective check, such as `hmac.compare_digest`, a checksum or signature comparison, `secure_filename`, `shlex.quote` or input escaping, even when the title does not say security and the file is not in a security folder.
- The pull request bot labels `leak risk` from released references it adds (`.clear()`, `del`, `weakref`, `gc.collect`) and the new `thread safety` label from locks and other thread-safe constructs, instead of only `runtime`.
- The pull request bot no longer labels `request tracing` or `monitoring` from ordinary names such as `span` or `counter`, no longer offers `prod deploy` for production wording in documentation, and no longer treats a source file that loses more lines than it gains, or a deleted line with the word remove, as an `import path` or `breaking-change` change; only a removed module-level function or class counts.
- The pull request bot states the kind of change on its own line and gives its reasons in plain words on the next line, naming each piece of evidence once: for example `Change kind: fix.` followed by `Why: the commit "fix(app): keep empty values" is typed fix; CHANGELOG.md gains 4 new entries under "Fixed".` Changelog entries in the same section are counted, only the first two commits of a long list are quoted, and the label explanation points to the decision instead of repeating its evidence.
- The pull request bot applies the change kind labels (`bug`, `enhancement`, `breaking-change`, `security`, `performance`) to the pull request instead of dropping them with the other non-technical labels, so a pull request whose release rests on its change kind is no longer shown as "Semver: n/a" and left out of the release.
- The issue bot labels feature requests written in plain words ("Please add …", "It would be great …", "I'd like to be able to …"), security reports that name an attack such as remote code execution or path traversal (`vulnerability`), and defects described without the word bug ("does not work", "freezes", "wrong"); it no longer reads `api` inside other words or the word request as an API route.
- The pull request bot reads the whole diff of each file instead of only its first 700 characters, every commit of a pull request with more than 250 commits, and changelogs larger than 1 MB, so evidence late in a large change is no longer missed. The local preview takes the message of a commit that is not made yet from `AUTOBOT_LOCAL_COMMIT_MESSAGE` or `AUTOBOT_LOCAL_COMMIT_MESSAGE_FILE`.
- The pull request bot treats new functions or classes in interface files as a feature when nothing else describes the change, labels `cache hit` only for code that uses a cache rather than code that drops cache entries, and matches its label words as whole words, so names such as `navigate`, `prevent`, `product` or `run_smoke_suite` no longer add deploy, rollout or webhook labels. The issue bot no longer labels `oauth scope` or `integration test` for the words scope and regression.
- Improve level and quality no longer cuts the noise or balances the tone a second time on a clip that Remove background noise already cleaned (in Fix clip automatically, or a file the noise removal made), so the room sound that the parts of a clip keep is no longer erased and a cleaned vocal no longer gets up to 15 dB more brightness, which made its level jump by about 3 dB.
- The phrase and word level rides of Improve level and quality are one gain with one limit of 6 dB per pass instead of two rides one after the other, and on a cleaned vocal they only turn loud stretches down, because the noise removal already lifted the quiet phrases. Each pass measures only what it needs instead of the whole clip.
- After a clip is split into vocals and music, a weak music part (more than 20 dB under the vocals) that sounds only while the voice sings and falls silent in its pauses is treated as voice left behind by the separation and is not kept.
- When the parts of a split clip are moved onto the beat, each part moves by the same number of beats from its own place, so under a tempo map with tempo changes every part lands on the beat instead of moving by the seconds measured at the start of the clip.

## [1.1.0] - 2026-11-21

### Added

- Remove background noise, Improve level and quality and Fix clip automatically first split a clip whose content is not known yet into a vocals clip and a music clip (with the vocal separation model), and then fix each with the chain made for it. A part that is essentially silent is dropped, so a clip of only singing or only instruments stays one clip and is marked as vocals or music. Both parts keep the clip's place, trim, level, pan, colour, mute and solo; splitting and fixing run as one job with one progress bar and are one undo step. When a take is split, both parts are moved onto the beat together by the same amount, measured on the original take. Clips already marked as vocals or music (for example the vocals from stem separation) are not split again. The vocal model (66 MB) is downloaded the first time it is needed; if it cannot run, the clip is simply fixed whole as before.
- Music clips have their own clean-up: rumble under 25 Hz and steady noise are removed and stretches holding only noise are lowered, while the bass, the tone, noisy sounds that belong to the music and the level are kept; their level is set to the level of the instruments rather than above the song's notes.
- Vocal clean-up turns breaths down by 12 dB (only noise-like sounds of at least 0.15 s, mostly under 4 kHz and 12 dB or more under the singing around them, so an h inside a word and s-sounds are kept), takes out pops of air on the microphone (lows that jump at least 12 dB more than the middle range and are louder than it), lifts phrases that are much quieter than the rest (80 % of the gap beyond 1.5 dB, at most 9 dB, never lowering the loud parts), and balances the tone in sections of 2 seconds so a duller phrase gets the same brightness as the rest.
- Several tracks and audio clips can be selected together in the track list: Ctrl-click (Cmd-click on macOS) adds or removes a row, Shift-click selects every row in between, Ctrl+A selects all, Escape clears. Mute, solo, show/hide, delete, colour, instrument, level and pan set on any selected row apply to the whole selection, as one undo step and one update of the editor, so changing 30 tracks takes one action instead of 30. With the track list focused, Delete removes the selection, M and S switch mute and solo, and a right-click menu offers the same actions.
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

- Vocal clean-up (after stem separation and for Remove background noise) is one steady chain: steady noise is turned down under the singing as well as in pauses, measured between the voice's harmonics so held notes are kept; pauses are lowered by a smooth expander that opens slightly before each phrase and closes after it, instead of a gate that switched on and off every 12 ms; and the tone is balanced against a typical vocal, cutting muddy low middles and giving a dull or muffled voice its presence and air back (at most 9 dB, never on bright s-sounds). The loudness of the phrases is kept.
- Splitting a clip into separate sounds measures the background noise as it really is in that clip and compares each stretch with the sound around it (within 1.5 seconds) instead of with the loudest moment of the whole clip, so the same soft phrase is kept or cut alike in every clip, and noise is still cut.
- Changes made from the track list (mute, solo, show/hide, colour, rename, instrument, delete) and clip edits (move by a beat, align, tempo follow, split, fixes) update only the notes and clips they affect instead of redrawing the whole song. Deleting selected notes does the same. On a song with 9,000 notes one mute now takes about 0.1 s instead of 1.3 s, and showing or hiding a track about 0.05 s.
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

- Separated vocals no longer pump in level when a phrase swells or fades, and no longer sound muffled.
- A clip with one loud part no longer loses soft phrases that a clip without it keeps.
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

- `clip_polish.remove_noise_file`, `improve_quality_file`, `fix_clip` and `fix_clip_parts` take a `kind` (`"vocals"` or `"music"`; anything else is treated as vocals, as before). `clip_level.improve` takes `kind` too. Cleaned files are named with `_cleanvocals_` or `_cleanmusic_`.
- `vocal_cleanup` makes its noise tools public: `highpass`, `noise_floor_power`, `noise_gains`, `pause_gains`, `loudness_gain`, `istft`.
- New `separation_engine.read_stereo` and `separation_engine.vocals_of`.
- Cleaned vocals sound different from 1.0.0: noise under the singing is reduced and the tone may be corrected by up to 9 dB per band. Clips that were already cleaned are not changed until they are cleaned again.
- A sound standing on its own well above the noise is now kept as its own part even when another part of the clip is much louder; before, anything 15 dB under the loudest part was cut.
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
