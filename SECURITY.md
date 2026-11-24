# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |
| < 1.0   | No        |

Fixes are made on the latest release line. Please confirm a problem on the newest version before reporting it.

## Reporting a vulnerability

Please do not open a public issue for a security problem.

1. Use GitHub's private reporting: on the [repository page](https://github.com/YaronKoresh/korshmer) open the Security tab and choose "Report a vulnerability".
2. Describe what an attacker can do, the steps to reproduce it, the KORSHMER version and your operating system. A small project file or script that shows the problem helps a lot.
3. Do not include real passwords, stream keys or tokens in the report.

You can expect a first answer within about a week. Once a fix is released the report is credited in the changelog unless you prefer otherwise.

## What to keep in mind when using KORSHMER

### Third-party plugins

A plugin is ordinary Python code and runs with the same rights as the application; there is no sandbox. Files in `saved_data/plugins/` are not run unless you approve them:

- list the file name and its SHA-256 checksum in `saved_data/approved_plugins.json`; a file that is missing from the list or changed afterwards is skipped and named in the log, or
- set `KORSHMER_ENABLE_USER_PLUGINS=1` to load every file in the folder.

The approval list is kept beside the plugin folder, not inside it, so that copying files into `plugins/` cannot also approve them. This is a guard against accidents and against files dropped into that one folder; a program that can already write anywhere in `saved_data/` or set environment variables for KORSHMER can approve its own plugin. Only approve plugins you have read or that come from someone you trust. Plugins installed as Python packages through the `korshmer.plugins` entry point group load without this step, because installing a package already runs its code.

### Local network servers

- Health server (`KORSHMER_HEALTH_PORT`): off by default. It listens on `127.0.0.1` and answers only requests whose `Host` header names this computer (`127.0.0.1`, `localhost`, `[::1]` or the address it was bound to) with its port; others get HTTP 403, which stops DNS-rebinding web pages. It serves at most 8 live log streams at once and drops silent connections after 30 seconds. It has no login, so any program running on your computer can read the logs it serves.
- Collaboration server: listens on `127.0.0.1` by default. It refuses to start on any other address unless a shared secret token is configured, and then every client must present that token. Connections that carry a browser `Origin` header from anywhere but this computer are refused, so a web page you visit cannot join a local session. The connection itself is not encrypted and the token travels in clear text, so use it on a network you trust or through a VPN or TLS tunnel. On a loopback server without a token, any local program can join.
- OSC remote control, OBS and Ableton Link are off until you tick them on the THEORY tab.

### Secrets

Keep the OBS password, Twitch token and AI assistant key in the system keychain (`python -m korshmer.utils.credential_store set <NAME>`) or in environment variables. They are redacted in `audit.log`.

Stream keys for YouTube, Twitch and Facebook are sent over `rtmps://` (TLS) and are masked in the command line that KORSHMER writes to its log. The key is still passed to FFmpeg as part of its command line, so other programs running under your account (and, on some systems, other local users) can see it in the process list while a stream runs. Do not stream from a shared computer you do not trust, and reset the key if you think it leaked.

### Project and media files

Open songs and media only from sources you trust.

- A clip path stored relative to the song folder is opened only when it stays inside that folder (after following links). A relative path that leads outside is kept in the song file but not opened, and a status message names it. Absolute clip paths are opened as they are: a song file can therefore still name any file on your disk that your account can read, exactly like a clip you added from elsewhere yourself. KORSHMER only reads such files as audio.
- Cached analysis data is stored as NumPy arrays and JSON, never as `pickle`.
- FFmpeg is called with the input passed as `file:<absolute path>`, without a shell and with a time limit, so a file name cannot be read as an option.
- A crafted media file can still make FFmpeg, libsndfile or a model library misbehave, so keep those up to date.

### The Windows install script

`scripts/install.bat` downloads the Python, LLVM and Windows SDK installers over HTTPS and runs them only when their Authenticode signature is valid and the signer is the expected publisher (Python Software Foundation, LLVM Foundation, Microsoft Corporation). FFmpeg is installed through winget; only if winget fails does the script fall back to an archive from gyan.dev, which is not signature-checked, so install FFmpeg yourself beforehand if that matters to you. Review the script before running it as Administrator.
