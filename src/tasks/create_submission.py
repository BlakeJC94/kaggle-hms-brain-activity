import argparse
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List

import git
import torch
from hms_brain_activity import logger

logger = logger.getChild(__name__)


CODE_DIRS = [
    "src/hms_brain_activity/",
    "src/core/",
    "src/tasks/",
]

MD_FILES = [
    "src/__init__.py",
    "pyproject.toml",
    ".python-version",
    "requirements.lock",
    "requirements-dev.lock",
    "README.md",
    "LICENSE",
]

CONFIG_DIR = Path("./config")


def main() -> str:
    return create_submission(**vars(parse()))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams_path")
    parser.add_argument("predict_args", nargs="*")
    return parser.parse_args()


def create_submission(hparams_path: str, predict_args: List[str]):
    dt = datetime.now()
    repo = git.Repo(".")
    commit_sha, branch_name = repo.rev_parse("HEAD").name_rev.split(" ", 1)

    has_unstaged_changes = repo.is_dirty()
    patch = None
    zip_suffix = ""
    if has_unstaged_changes:
        patch = repo.git.execute(["git", "diff", "--", *CODE_DIRS, *MD_FILES])
        zip_suffix = "_unstaged"

    zip_name = f"submission_{dt.strftime('%Y-%m-%d_%H-%M-%S')}_{branch_name}_{commit_sha[:8]}{zip_suffix}.zip"
    logger.info(f"Creating submission '{zip_name}'")

    run_script_template = create_run_script_template(
        dt,
        branch_name,
        commit_sha,
        has_unstaged_changes,
    )

    with zipfile.ZipFile(zip_name, "w") as zf:
        add_config_to_zip(zf, hparams_path, predict_args, run_script_template)

        for md_file in MD_FILES:
            md_fp = Path(md_file)
            logger.info(f"Adding file '{md_fp}'.")
            zf.write(Path(md_fp))

        for code_dir in CODE_DIRS:
            code_d = Path(code_dir)
            logger.info(f"Adding directory '{code_d}'.")
            for fp in code_d.rglob("*"):
                if "__pycache__" in str(fp):
                    continue
                zf.write(fp)

        if patch:
            patch_fp = Path("unstaged_changes.patch")
            logger.info(f"Creating and adding '{patch_fp}'.")
            zf.writestr(str(patch_fp), patch)

    logger.info("Done!")


def create_run_script_template(
    dt,
    branch_name,
    commit_sha,
    has_unstaged_changes,
):
    run_script_header_template_lines = [
        "# Created: {dt_created}",
        "# Branch name: {branch_name}",
        "# Commit SHA: {commit_sha}",
        "# Unstaged changes: {has_unstaged_changes}",
        "",
    ]
    run_script_header_template = "\n".join(run_script_header_template_lines)
    run_script_header_template = run_script_header_template.format(
        dt_created=dt.isoformat(),
        branch_name=branch_name,
        commit_sha=commit_sha,
        has_unstaged_changes=has_unstaged_changes,
    )
    run_script_template_lines = [
        run_script_header_template,
        "import argparse",
        "from tasks.predict import predict",
        "",
        "parser = argparse.ArgumentParser()",
        "parser.add_argument('predict_args', nargs='*')",
        "",
        "predict_args = parser.parse_args().predict_args",
        "predict({hparams_dest}, [*{predict_args}, *predict_args])",
        "",
    ]
    run_script_template = "\n".join(run_script_template_lines)
    return run_script_template


def compress_weights(weights_path: Path, tmp_dir):
    ckpt = torch.load(weights_path, map_location="cpu")
    ckpt_new = {"state_dict": ckpt["state_dict"]}

    weights_path_new = tmp_dir / weights_path.parts[-4] / weights_path.name
    weights_path_new.parent.mkdir()
    torch.save(ckpt_new, weights_path_new)

    return weights_path_new


def add_config_to_zip(zf, hparams_path, predict_args, run_script_template):
    hparams_path = Path(hparams_path)
    hparams_dest = CONFIG_DIR / Path(*hparams_path.parts[-2:])

    module_path = hparams_path.parent / "__init__.py"
    module_dest = CONFIG_DIR / Path(*module_path.parts[-2:])

    for fp, dest in [
        (hparams_path, hparams_dest),
        (module_path, module_dest),
    ]:
        logger.info(f"Writing '{fp}' to '{dest}'.")
        zf.write(fp, dest)

    predict_args_dest = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for weights_path in predict_args:
            weights_path = Path(weights_path)
            weights_path = compress_weights(weights_path, Path(tmp_dir))
            weights_dest = "/".join(weights_path.parts[-2:])
            weights_dest = CONFIG_DIR / weights_dest
            predict_args_dest.append(str(weights_dest))
            logger.info(f"Writing '{weights_path}' to '{weights_dest}'.")
            zf.write(weights_path, weights_dest)

    run_script = run_script_template.format(
        hparams_dest=repr(str(hparams_dest)),
        predict_args=repr(predict_args_dest),
    )
    run_fp = Path("run.py")
    logger.info(f"Creating and adding '{run_fp}'.")
    zf.writestr(str(run_fp), run_script)


if __name__ == "__main__":
    main()
