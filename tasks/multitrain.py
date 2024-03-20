import argparse
import os
import concurrent.futures as cf
from typing import List, Optional

from src.hms_brain_activity import logger
from tasks.train import train

logger = logger.getChild(__name__)


def main() -> str:
    return multitrain(**vars(parse()))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams_paths", nargs="*", type=str)
    parser.add_argument(
        "-g",
        "--gpu-devices",
        nargs="*",
        type=int,
    )
    parser.add_argument("-o", "--offline", action="store_true", default=False)
    return parser.parse_args()


def multitrain(
    hparams_paths: List[str],
    gpu_devices: Optional[List[int]] = None,
    offline: bool = False,
):
    logger.info(f"Process ID: {os.getpid()}")

    if gpu_devices is None:
        gpu_devices = [None]

    with cf.ProcessPoolExecutor(max_workers=len(hparams_paths)) as pool:
        future_to_hparams_path = {}
        for i, hparams_path in enumerate(hparams_paths):
            device = gpu_devices[i % len(gpu_devices)]
            future = pool.submit(train, hparams_path, device=device, offline=offline)
            future_to_hparams_path[future] = hparams_path

        for future in cf.as_completed(future_to_hparams_path):
            hparams_path = future_to_hparams_path[future]
            try:
                _ = future.result()
            except Exception as exc:
                print(f"Error occurred with '{hparams_path}': {exc}")
            else:
                print(f"Experiment '{hparams_path}' has finished.")
