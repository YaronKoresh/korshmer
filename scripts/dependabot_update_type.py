import sys

PREFIX = "update-type:"
PATCH = "version-update:semver-patch"


def update_types(message):
    types = []
    for line in message.splitlines():
        stripped = line.strip()
        if stripped.startswith(PREFIX):
            types.append(stripped[len(PREFIX) :].strip())
    return types


def is_patch_only(message):
    types = update_types(message)
    return bool(types) and all(kind == PATCH for kind in types)


def main(argv=None):
    message = sys.stdin.read()
    return 0 if is_patch_only(message) else 1


if __name__ == "__main__":
    sys.exit(main())
