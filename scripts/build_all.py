from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_TIMESTAMP_URL = "http://timestamp.digicert.com"
SIGNTOOL_GLOB = "Windows Kits/10/bin/*/x64/signtool.exe"


def build_wheel():
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel"],
        cwd=Path(__file__).parent.parent,
    )
    return result.returncode == 0


def build_pyinstaller(target_name: str = "korshmer"):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--onefile",
            "--name",
            target_name,
            "--add-data",
            "src/korshmer/constants.py:korshmer",
            "src/korshmer/__main__.py",
        ],
        cwd=Path(__file__).parent.parent,
    )
    return result.returncode == 0


def executable_path(target_name: str = "korshmer", system: str | None = None):
    system = system or platform.system()
    suffix = ".exe" if system == "Windows" else ""
    return ROOT / "dist" / (target_name + suffix)


def find_signtool():
    found = shutil.which("signtool")
    if found:
        return found
    for var in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(var)
        if not base:
            continue
        matches = sorted(Path(base).glob(SIGNTOOL_GLOB))
        if matches:
            return str(matches[-1])
    return None


def sign_windows(path, env=None):
    env = os.environ if env is None else env
    cert_file = env.get("KORSHMER_WIN_CERT_FILE")
    cert_password = env.get("KORSHMER_WIN_CERT_PASSWORD")
    if not cert_file or not cert_password:
        print(
            "Windows signing needs KORSHMER_WIN_CERT_FILE and "
            "KORSHMER_WIN_CERT_PASSWORD"
        )
        return False
    signtool = find_signtool()
    if not signtool:
        print("signtool was not found")
        return False
    timestamp_url = env.get("KORSHMER_TIMESTAMP_URL") or DEFAULT_TIMESTAMP_URL
    sign = subprocess.run(
        [
            signtool,
            "sign",
            "/f",
            cert_file,
            "/p",
            cert_password,
            "/fd",
            "SHA256",
            "/tr",
            timestamp_url,
            "/td",
            "SHA256",
            str(path),
        ]
    )
    if sign.returncode != 0:
        return False
    verify = subprocess.run([signtool, "verify", "/pa", str(path)])
    return verify.returncode == 0


def sign_macos(path, env=None):
    env = os.environ if env is None else env
    identity = env.get("KORSHMER_MACOS_IDENTITY")
    if not identity:
        print("macOS signing needs KORSHMER_MACOS_IDENTITY")
        return False
    sign = subprocess.run(
        [
            "codesign",
            "--force",
            "--options",
            "runtime",
            "--timestamp",
            "--sign",
            identity,
            str(path),
        ]
    )
    if sign.returncode != 0:
        return False
    verify = subprocess.run(
        ["codesign", "--verify", "--strict", "--verbose=2", str(path)]
    )
    return verify.returncode == 0


def sign_executable(path, system: str | None = None, env=None):
    system = system or platform.system()
    if not Path(path).is_file():
        print(f"Nothing to sign, {path} does not exist")
        return False
    if system == "Windows":
        return sign_windows(path, env)
    if system == "Darwin":
        return sign_macos(path, env)
    print(f"Code signing is not used on {system}")
    return True


def build_briefcase():
    print("To build with Briefcase, run:")
    print("  briefcase create")
    print("  briefcase build")
    print("  briefcase run")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build KORSHMER DAW")
    parser.add_argument(
        "--wheel", action="store_true", help="Build wheel distribution"
    )
    parser.add_argument(
        "--pyinstaller", action="store_true", help="Build standalone executable"
    )
    parser.add_argument(
        "--sign",
        action="store_true",
        help="Code sign the executable built with --pyinstaller",
    )
    parser.add_argument(
        "--briefcase",
        action="store_true",
        help="Show briefcase build instructions",
    )

    args = parser.parse_args()

    if args.sign and not args.pyinstaller:
        parser.error("--sign needs --pyinstaller")

    if not (args.wheel or args.pyinstaller or args.briefcase):
        parser.print_help()
        return

    if args.wheel:
        print("Building wheel...")
        if build_wheel():
            print("Wheel build successful")
        else:
            print("Wheel build failed")

    if args.pyinstaller:
        print("Building PyInstaller executable...")
        if build_pyinstaller():
            print("PyInstaller build successful")
        else:
            print("PyInstaller build failed")
            if args.sign:
                sys.exit(1)
        if args.sign:
            print("Signing executable...")
            if sign_executable(executable_path()):
                print("Signing successful")
            else:
                print("Signing failed")
                sys.exit(1)

    if args.briefcase:
        build_briefcase()


if __name__ == "__main__":
    main()
