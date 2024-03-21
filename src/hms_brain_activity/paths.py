from pathlib import Path

PROJECT_DIR = Path(__file__).absolute().parent
SRC_DIR = PROJECT_DIR.parent
ROOT_DIR = SRC_DIR.parent

ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

DATA_DIR = ROOT_DIR / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"


def get_task_dir_name(task) -> str:
    return f"{task.name}-{task.id}"


def get_task_artifacts_dir(task) -> Path:
    return ARTIFACTS_DIR / get_task_dir_name(task)
