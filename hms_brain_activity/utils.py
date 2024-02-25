import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Tuple

from sklearn.model_selection import train_test_split
import pandas as pd

from hms_brain_activity.globals import VOTE_NAMES


def import_script_as_module(config_path) -> Any:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Couldn't find '{str(config_path)}'.")
    if not config_path.is_file():
        raise FileNotFoundError(f"Path '{str(config_path)}' is not a file.")
    name = Path(config_path).stem
    spec = spec_from_file_location(name, config_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_dict(d: dict) -> str:
    return json.dumps(d, indent=2, default=lambda o: str(o))


def saggital_flip_channel(ch: str) -> str:
    if ch == "EKG" or ch[-1] == "z":
        return ch
    pos = "".join([c for c in ch if not c.isnumeric()])
    digit = int("".join([c for c in ch if c.isnumeric()]))
    translation = -1 if digit % 2 == 0 else 1
    return "".join([pos, str(digit + translation)])


def split_annotations_across_patient_by_class(
    ann: pd.DataFrame,
    test_size=0.2,
    random_state=0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    res = ann[["patient_id", *VOTE_NAMES]]
    res = res.groupby("patient_id").sum()
    res["class_1"] = res[VOTE_NAMES].idxmax(axis=1)
    res["class_2"] = (
        res[VOTE_NAMES].subtract(res[VOTE_NAMES].max(axis=1), axis=0).idxmax(axis=1)
    )

    res_train, res_val = train_test_split(
        res,
        test_size=test_size,
        stratify=res[["class_1", "class_2"]],
        shuffle=True,
        random_state=random_state,
    )

    ann_train = ann[ann["patient_id"].isin(res_train.index)]
    ann_val = ann[ann["patient_id"].isin(res_val.index)]
    return ann_train, ann_val
