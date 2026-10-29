import subprocess
import sys
import tempfile

TARGET = "tests/performance/test_memory_leaks.py"


def main():
    if sys.platform == "win32":
        print("memray is not supported on Windows; skipping")
        return 0
    try:
        import importlib

        importlib.import_module("pytest_memray")
    except ImportError:
        print("pytest-memray is not installed; skipping")
        return 0
    with tempfile.TemporaryDirectory() as out:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            TARGET,
            "--memray",
            "-p",
            "no:cacheprovider",
            "--memray-bin-path",
            out,
        ]
        return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
