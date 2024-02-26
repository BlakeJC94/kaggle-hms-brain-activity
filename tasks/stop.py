import os
import signal
from pathlib import Path

from hms_brain_activity import logger


def main():
    foo = [
        (
            fp.parent.stem,
            int((fp.parent / "stat").read_text().split(" ")[21]),
        )
        for fp in Path("/proc").glob("*/cmdline")
        if "train.py" in fp.read_text()
        and int((fp / "status").read_text().split("\n")[8].split("\t")[1])
        == os.getuid()
    ]

    if len(foo) == 0:
        logger.error("No matching processes")
        return

    foo = sorted(foo, key=lambda x: x[1])
    pid = foo[0][0]

    logger.info(f"Sending interrupt to {pid}")
    os.kill(pid, signal.SIGINT)


if __name__ == "__main__":
    main()
