import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from pytorch_lightning.loggers import TensorBoardLogger

from hms_brain_activity.paths import get_task_dir_name

try:
    from clearml import Task
except ImportError:
    Task = None

logger = logging.getLogger(__name__)


@dataclass
class _OfflineTask:
    name: str

    @property
    def id(self):
        return "offline"


class ClearMlLogger(TensorBoardLogger):
    def __init__(
        self,
        hparams: Dict[str, Any],
        config_path: str | Path,
        task_name: str,
        root_dir: str | Path,
        dev_run: bool = False,
        offline: bool = False,
        **kwargs: Any,
    ):
        root_dir = Path(root_dir)

        if offline or Task is None:
            task = _OfflineTask(task_name)
        else:
            task = self.setup_task(hparams, config_path, task_name)

        super().__init__(
            save_dir=str(root_dir / f"{get_task_dir_name(task)}/logs"),
            name="",
            version=None,
            log_graph=False,
            default_hp_metric=False,
            prefix="",
            sub_dir=None,
            **kwargs,
        )

        self.task = task

    @staticmethod
    def setup_task(hparams, config_path, task_name) -> Task:
        project_name = hparams["task"]["init"]["project_name"]
        task_base_name, task_stem_name = task_name.split("-", 1)
        for k, v in {
            "project name": project_name,
            "task base name": task_base_name,
            "task stem name": task_stem_name,
        }.items():
            if "-" in v:
                raise ValueError(f"The character '-' is forbidden in the {k} ('{v}')")

        # Increment version of task
        prev_tasks = Task.get_tasks(project_name=project_name, task_name=f"^{task_name}",)
        max_task_v = -1
        for t in prev_tasks:
            version_suffix = t.name.split("-", 2)[-1]
            version_search = re.search(r"\d+", version_suffix)
            if version_search is None:
                continue
            version = int(version_search.group(0))
            max_task_v = max(max_task_v, version)
        task_name = "-v".join([task_name, str(max_task_v + 1)])
        logger.info(f"Task name: {task_name}")

        # Start ClearML
        task_init_kwargs = hparams.get("task", {}).get("init", {})
        task_init_kwargs = {
            "continue_last_task": False,
            "reuse_last_task_id": False,
            "auto_connect_frameworks": {
                "matplotlib": True,
                "pytorch": False,
                "tensorboard": True,
            },
            **task_init_kwargs,
            "task_name": task_name,
        }
        task = Task.init(**task_init_kwargs)

        ckpt_params = hparams["checkpoint"]
        checkpoint_task_id = ckpt_params.get("checkpoint_task_id")
        if checkpoint_task_id:
            task.set_parent(checkpoint_task_id)

        # Connect configurations
        task.connect_configuration(config_path, "config")
        task.connect(hparams, "hparams")
        return task
