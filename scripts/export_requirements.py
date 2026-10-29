import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def runtime_requirements(pyproject):
    data = tomllib.loads(Path(pyproject).read_text(encoding="utf-8"))
    names = []
    for entry in data["project"]["dependencies"]:
        requirement = entry.split(";")[0].strip()
        if "@" in requirement:
            continue
        names.append(entry.strip())
    return names


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    target = Path(args[0]) if args else ROOT / "pyproject.toml"
    for line in runtime_requirements(target):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
