import argparse
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch

from hms_brain_activity import logger
from core.utils import import_script_as_module, print_dict

logger = logger.getChild(__name__)


def main() -> str:
    return predict(**vars(parse()))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams_path")
    parser.add_argument("predict_args", nargs="*")
    return parser.parse_args()


def predict(hparams_path: str, predict_args: List[str]):
    hparams = import_script_as_module(hparams_path).hparams
    logger.info("hparams =")
    logger.info(print_dict(hparams))

    config_path = Path(hparams_path).parent / "__init__.py"
    logger.info(f"Using config at '{config_path}'")
    logger.info(f"Using predict args: {predict_args}")
    config_fn = import_script_as_module(config_path).predict_config
    config = config_fn(hparams, predict_args)

    trainer = pl.Trainer(
        callbacks=config.get("callbacks", []),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else "auto",
    )
    trainer.predict(
        config["model"],
        dataloaders=config["predict_dataloaders"],
        return_predictions=False,
    )

    logger.info("Finished predictions")


if __name__ == "__main__":
    main()
