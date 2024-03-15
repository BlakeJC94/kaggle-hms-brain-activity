import argparse
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl

from hms_brain_activity import logger
from hms_brain_activity.utils import import_script_as_module, print_dict
from hms_brain_activity.callbacks import SubmissionWriter


def main() -> str:
    return predict(**vars(parse()))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams_path")
    parser.add_argument("weights_path", nargs="?", default=None)
    return parser.parse_args()


def predict(hparams_path: str, weights_path: Optional[str] = None):
    hparams = import_script_as_module(hparams_path).hparams
    logger.info("hparams =")
    logger.info(print_dict(hparams))

    config_path = Path(hparams_path).parent / "__init__.py"
    logger.info(f"Using config at '{config_path}'")
    logger.info(f"Using weights at '{weights_path}'")
    config_fn = import_script_as_module(config_path).predict_config
    config = config_fn(hparams, weights_path)

    trainer = pl.Trainer(
        callbacks=[
            SubmissionWriter(hparams["predict"].get("output_dir", "./")),
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
