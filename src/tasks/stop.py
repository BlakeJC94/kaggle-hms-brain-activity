import argparse
import os
import signal
from pathlib import Path

from src.hms_brain_activity import logger

logger = logger.getChild(__name__)


def main() -> str:
    return stop(**vars(parse()))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pid_file")
    return parser.parse_args()


def stop(pid_file):
    pid_file = Path(pid_file)
    if not pid_file.exists():
        logger.info(f"Given PID file '{pid_file}' doesn't exist, exiting")
        return

    with open(pid_file, "r") as f:
        pid = int(f.read())

    logger.info(f"Sending interrupt to {pid}")
    os.kill(pid, signal.SIGINT)


if __name__ == "__main__":
    main()
