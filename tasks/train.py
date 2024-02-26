import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from clearml import Task
from pytorch_lightning.loggers import TensorBoardLogger

from hms_brain_activity import logger
from hms_brain_activity.callbacks import EpochProgress
from hms_brain_activity.paths import get_task_artifacts_dir
from hms_brain_activity.utils import import_script_as_module, print_dict


def main() -> str:
    return train(**vars(parse()))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams_path")
    parser.add_argument("-d", "--dev-run", action="store_true", default=False)
    parser.add_argument("-D", "--pdb", action="store_true", default=False)
    parser.add_argument("-o", "--offline", action="store_true", default=False)
    return parser.parse_args()


def train(
    hparams_path: str,
    dev_run: bool = False,
    pdb: bool = False,
    offline: bool=False,
) -> str:
    """Choo choo"""
    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision('high')

    if dev_run:
        logger.info("DEV RUN")
    if pdb:
        logger.info("PDB")
    if offline:
        logger.info("PDB")

    hparams, task, config = setup_task(hparams_path, dev_run, pdb, offline)
    save_dir = get_task_artifacts_dir(task) / "train"
    weights_dir = save_dir / "model_weights"
    for fp in weights_dir.glob("*.ckpt"):
        fp.unlink()

    # Initialise callbacks
    callbacks = [
        EpochProgress(),
        pl.callbacks.LearningRateMonitor(),
        *config.get("callbacks", []),
    ]
    if not dev_run:
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                weights_dir,
                monitor=hparams["config"]["monitor"],
                save_last=True,
            ),
            *callbacks,
        ]

    # Prepare trainer
    trainer_init_kwargs = hparams["trainer"].get("init", {})
    trainer_fit_kwargs = hparams["trainer"].get("fit", {})
    trainer_init_kwargs = {
        "logger": TensorBoardLogger(
            save_dir=str(save_dir / "logs"),
            name="",
            default_hp_metric=False,
        ),
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "callbacks": callbacks,
        "num_sanity_val_steps": 0,
        "max_epochs": -1,
        **trainer_init_kwargs,
    }

    # Download checkpoint
    ckpt_params = hparams["checkpoint"]
    checkpoint_task_id = ckpt_params.get("checkpoint_task_id")
    checkpoint_name = ckpt_params.get("checkpoint_name", "last")
    weights_only = bool(ckpt_params.get("weights_only", False))
    if checkpoint_task_id:
        prev_task = Task.get_task(task_id=checkpoint_task_id)
        temp_path = (
            get_task_artifacts_dir(prev_task)
            / f"train/model_weights/{checkpoint_name}.ckpt"
        )

        weights_path = Path(weights_dir) / f"{checkpoint_name}.ckpt"
        shutil.move(temp_path, weights_path)

        # If not weights only, add to fit kwargs
        if weights_only:
            logger.info(
                f"Loading weights '{checkpoint_name}' from {checkpoint_task_id}"
            )
            ckpt = torch.load(weights_path, map_location="cpu")
            config["model"].load_state_dict(ckpt["state_dict"])
        else:
            logger.info(
                f"Loading checkpoint '{checkpoint_name}' from {checkpoint_task_id}"
            )
            trainer_fit_kwargs["ckpt_path"] = str(weights_path)

    logger.info("trainer_init_kwargs =")
    logger.info(print_dict(trainer_init_kwargs))

    logger.info("trainer_fit_kwargs =")
    logger.info(print_dict(trainer_fit_kwargs))

    # Trainer.fit
    trainer = pl.Trainer(**trainer_init_kwargs)
    try:
        trainer.validate(config["model"], dataloaders=config["val_dataloaders"])
        trainer.fit(
            config["model"],
            train_dataloaders=config["train_dataloaders"],
            val_dataloaders=config["val_dataloaders"],
            **trainer_fit_kwargs,
        )
    except Exception as err:
        if pdb:
            model = config["model"]
            train_dls = config["train_dataloaders"]
            val_ds = config["val_dataloaders"]
            breakpoint()
        raise err

    return task.id


from dataclasses import dataclass
@dataclass
class OfflineTask:
    name: str
    @property
    def id(self):
        return "offline"

def setup_task(
    hparams_path: Path,
    dev_run: bool,
    pdb: bool,
    offline: bool,
) -> Tuple[Dict[str, Any], Task, Dict[str, Any]]:
    # Import hparams
    hparams = import_script_as_module(hparams_path).hparams

    # Set task name and check if it's not already running
    task_name = "-".join(Path(hparams_path).parts[-2:]).removesuffix(".py")
    project_name = hparams["task"]["init"]["project_name"]

    if dev_run:
        hparams = set_hparams_debug_overrides(hparams)
        task_name = f"DEV RUN: {task_name}"
        project_name = "test"
    config_path = hparams["config"].get(
        "path",
        str(Path(hparams_path).parent / "__init__.py"),
    )

    # Print hparams
    logger.info("hparams =")
    logger.info(print_dict(hparams))

    logger.info(f"Using config at '{config_path}'")
    config_fn = import_script_as_module(config_path).train_config
    logger.info("Setting hparams on config")
    try:
        config = config_fn(hparams)
    except Exception as err:
        if pdb:
            logger.error("Couldn't import config")
            breakpoint()
        raise err

    task = OfflineTask(task_name)
    if not offline:
        # Halt if task is currently running for project_name/task_name
        existing_task = Task.get_task(project_name=project_name, task_name=task_name)
        if existing_task is not None and existing_task.status == "in_progress":
            raise FileExistsError(f"Task '{project_name}/{task_name}' is in progress.")

        # Unpack checkpoint kwargs
        ckpt_params = hparams["checkpoint"]
        checkpoint_task_id = ckpt_params.get("checkpoint_task_id")
        checkpoint_name = ckpt_params.get("checkpoint_name", "last")
        weights_only = bool(ckpt_params.get("weights_only", False))
        continue_last_task = bool(
            checkpoint_task_id and (checkpoint_name == "last") and not weights_only
        )
        logger.info(f"{continue_last_task = }")

        # Set debug overrides
        if dev_run and continue_last_task:
            logger.info("(Continuing last task is disabled for debug runs)")
            continue_last_task = False

        # Start ClearML
        # Task.add_requirements("./requirements.txt")
        # Task.add_requirements("./requirements-dev.txt")
        task_init_kwargs = hparams.get("task", {}).get("init", {})
        task_init_kwargs = {
            **task_init_kwargs,
            "task_name": task_name,
            "project_name": project_name,
            "continue_last_task": continue_last_task,
            "auto_connect_frameworks": {
                "matplotlib": True,
                "pytorch": False,
                "tensorboard": True,
            },
        }
        task = Task.init(**task_init_kwargs)
        if checkpoint_task_id:
            task.set_parent(checkpoint_task_id)
        elif parent_task_id := hparams.get("task", {}).get("parent_task_id"):
            task.set_parent(parent_task_id)

        # Connect configurations
        task.connect_configuration(config_path, "config")
        task.connect(hparams, "hparams")


    return hparams, task, config


def set_hparams_debug_overrides(hparams):
    # Task overrides
    hparams["task"]["init"]["project_name"] = "test"
    hparams["task"]["init"]["reuse_last_task_id"] = True
    # Config overrides
    hparams["config"]["num_workers"] = 0
    # Trainer overrides
    hparams["trainer"]["init"]["overfit_batches"] = 1
    hparams["trainer"]["init"]["log_every_n_steps"] = 1
    hparams["trainer"]["init"]["enable_progress_bar"] = True
    return hparams


if __name__ == "__main__":
    main()
