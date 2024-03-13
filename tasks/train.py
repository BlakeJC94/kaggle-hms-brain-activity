import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from clearml import Task

from hms_brain_activity import logger
from hms_brain_activity.paths import ARTIFACTS_DIR
from hms_brain_activity.loggers import ClearMlLogger
from hms_brain_activity.callbacks import EpochProgress, NanMonitor
from hms_brain_activity.paths import get_task_dir_name
from hms_brain_activity.utils import import_script_as_module, print_dict


def main() -> str:
    return train(**vars(parse()))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams_path")
    parser.add_argument("-d", "--dev-run", type=float, default=-1.0)
    parser.add_argument("-D", "--pdb", action="store_true", default=False)
    parser.add_argument("-o", "--offline", action="store_true", default=False)
    return parser.parse_args()


def train(
    hparams_path: str,
    dev_run: bool = False,
    pdb: bool = False,
    offline: bool = False,
) -> str:
    """Choo choo"""
    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("high")

    if dev_run:
        logger.info("DEV RUN")
    if pdb:
        logger.info("PDB")
    if offline:
        logger.info("OFFLINE")

    # Get config from hparams
    hparams, config_path = get_hparams_and_config_path(hparams_path, dev_run)
    config = compile_config(hparams, config_path, pdb, "train_config")
    hparams, config = load_weights(hparams, config)

    # Create task name
    task_name = "-".join(Path(hparams_path).parts[-2:]).removesuffix(".py")
    if dev_run:
        task_name = f"dev-{task_name}"

    # Initialise logger
    clearml_logger = ClearMlLogger(
        hparams=hparams,
        config_path=config_path,
        task_name=task_name,
        root_dir=ARTIFACTS_DIR,
        dev_run=dev_run,
        offline=offline,
    )
    task = clearml_logger.task
    save_dir = ARTIFACTS_DIR / f"{get_task_dir_name(task)}/train"

    logger.info("hparams =")
    logger.info(print_dict(hparams))
    logger.info(f"Using config at '{config_path}'")

    # Initialise callbacks
    callbacks = [
        EpochProgress(),
        NanMonitor(),
        pl.callbacks.LearningRateMonitor(),
        *config.get("callbacks", []),
    ]
    if not dev_run:
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                save_dir / "model_weights",
                monitor=hparams["config"]["monitor"],
                save_last=True,
            ),
            *callbacks,
        ]

    # Initialise trainer and kwargs
    trainer_fit_kwargs = hparams["trainer"].get("fit", {})
    trainer_init_kwargs = {
        "logger": clearml_logger,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "callbacks": callbacks,
        "num_sanity_val_steps": 0,
        "max_epochs": -1,
        **hparams["trainer"].get("init", {}),
    }
    trainer = pl.Trainer(**trainer_init_kwargs)

    logger.info("trainer_init_kwargs =")
    logger.info(print_dict(trainer_init_kwargs))
    logger.info("trainer_fit_kwargs =")
    logger.info(print_dict(trainer_fit_kwargs))

    # Validate, then fit model
    try:
        trainer.validate(config["model"], dataloaders=config["val_dataloaders"])
        trainer.fit(
            config["model"],
            train_dataloaders=config["train_dataloaders"],
            val_dataloaders=config["val_dataloaders"],
            **trainer_fit_kwargs,
        )
    except Exception as err:
        logger.error(f"Error when running task: {str(err)}")
        if pdb:
            model = config["model"]
            train_dls = config["train_dataloaders"]
            val_ds = config["val_dataloaders"]
            breakpoint()
        raise err

    return task.id


def get_hparams_and_config_path(
    hparams_path: Path,
    dev_run: bool,
) -> Tuple[Dict[str, Any], str]:
    hparams = import_script_as_module(hparams_path).hparams
    if dev_run:
        hparams = set_hparams_debug_overrides(hparams, dev_run)

    config_path = hparams["config"].get(
        "path",
        str(Path(hparams_path).parent / "__init__.py"),
    )
    return hparams, config_path


def set_hparams_debug_overrides(hparams, dev_run):
    """"""
    # Task overrides
    hparams["task"]["init"]["project_name"] = "test"
    hparams["task"]["init"]["reuse_last_task_id"] = True
    hparams["task"]["init"]["continue_last_task"] = False
    # Config overrides
    hparams["config"]["num_workers"] = 0
    # Trainer overrides
    hparams["trainer"]["init"]["log_every_n_steps"] = 1
    hparams["trainer"]["init"]["overfit_batches"] = dev_run if dev_run > 0 else 1
    return hparams


def compile_config(
    hparams: Dict[str, Any],
    config_path: Path,
    pdb: bool,
    field: str = "train_config",
) -> Dict[str, Any]:
    config_fn = getattr(import_script_as_module(config_path), field)
    logger.info("Setting hparams on config")

    try:
        config = config_fn(hparams)
    except Exception as err:
        logger.error(f"Couldn't import config: {str(err)}")
        if pdb:
            breakpoint()
        raise err

    return config


def load_weights(
    hparams: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
]:
    ckpt_params = hparams["checkpoint"]
    checkpoint_task_id = ckpt_params.get("checkpoint_task_id")
    if not checkpoint_task_id:
        return hparams, config

    checkpoint_name = ckpt_params.get("checkpoint_name", "last")
    weights_only = bool(ckpt_params.get("weights_only", False))
    continue_last_task = bool(
        checkpoint_task_id and (checkpoint_name == "last") and not weights_only
    )
    logger.info(f"{continue_last_task = }")

    hparams["task"] = hparams.get("task", {})
    hparams["task"]["init"] = hparams["task"].get("init", {})
    if not hparams["task"]["init"].get("continue_last_task") is False:
        hparams["task"]["init"]["continue_last_task"] = True

    # TODO update to use remote checkpoint storage
    prev_task = Task.get_task(task_id=checkpoint_task_id)
    temp_path = (
        ARTIFACTS_DIR
        / f"{get_task_dir_name(prev_task)}/train/model_weights/{checkpoint_name}.ckpt"
    )

    # If not weights only, add to fit kwargs
    if weights_only:
        logger.info(f"Loading weights '{checkpoint_name}' from {checkpoint_task_id}")
        ckpt = torch.load(temp_path, map_location="cpu")
        config["model"].load_state_dict(ckpt["state_dict"])
    else:
        logger.info(f"Loading checkpoint '{checkpoint_name}' from {checkpoint_task_id}")
        hparams["trainer"]["fit"] = {
            **hparams["trainer"].get("init", {}),
            "ckpt_path": str(temp_path),
        }

    return hparams, config


if __name__ == "__main__":
    main()
