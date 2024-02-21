from pathlib import Path

PROJECT_DIR = Path(__file__).absolute().parent
ROOT_DIR = PROJECT_DIR.parent

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exists_ok=True)
