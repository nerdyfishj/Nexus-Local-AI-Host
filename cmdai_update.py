"""Update a pipx-installed .cmdAI from its configured package source."""

import shutil
import subprocess
import sys


PACKAGE_NAME = "cmdai-assistant"


def main():
    pipx = shutil.which("pipx")
    if pipx is None:
        print(".cmdAI update error: pipx was not found on PATH.", file=sys.stderr)
        return 1

    try:
        result = subprocess.run([pipx, "upgrade", PACKAGE_NAME], check=False)
    except OSError as error:
        print(".cmdAI update error: could not run pipx: {}".format(error), file=sys.stderr)
        return 1

    if result.returncode:
        print(
            ".cmdAI update failed (pipx exited with status {}).".format(result.returncode),
            file=sys.stderr,
        )
        return result.returncode

    print("Update complete. Exit and restart .cmdAI to use the new version.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
