import argparse
from pathlib import Path
from typing import List

import pytorch_lightning as pl

from hms_brain_activity import logger
from hms_brain_activity.core.utils import import_script_as_module, print_dict
from hms_brain_activity.core.callbacks import SubmissionWriter

logger = logger.getChild(__name__)


def main() -> str:
    return predict(**vars(parse()))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams_path")
    parser.add_argument("weights_path")
    parser.add_argument("dataset_args", nargs="*")
    return parser.parse_args()


def predict(hparams_path: str, weights_path: str, dataset_args: List[str]):
    hparams = import_script_as_module(hparams_path).hparams
    logger.info("hparams =")
    logger.info(print_dict(hparams))

    config_path = Path(hparams_path).parent / "__init__.py"
    logger.info(f"Using config at '{config_path}'")
    logger.info(f"Using weights at '{weights_path}'")
    logger.info(f"Using dataset args: {dataset_args}")
    config_fn = import_script_as_module(config_path).predict_config
    config = config_fn(hparams, weights_path, dataset_args)

    trainer = pl.Trainer(
        callbacks=[
            SubmissionWriter("./"),
        ]
    )
    trainer.predict(
        config["model"],
        dataloaders=config["predict_dataloaders"],
        return_predictions=False,
    )

    logger.info("Finished predictions")


if __name__ == "__main__":
    main()
