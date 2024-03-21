from typing import Tuple

from sklearn.model_selection import train_test_split
import pandas as pd

from hms_brain_activity.globals import VOTE_NAMES
from hms_brain_activity.paths import DATA_PROCESSED_DIR


def split_annotations_across_patients(
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


def main():
    annotations = pd.read_csv("./data/hms/train.csv")

    train_annotations, val_annotations = split_annotations_across_patients(
        annotations,
        test_size=0.2,
        random_state=0,
    )

    DATA_PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    train_annotations.to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
    val_annotations.to_csv(DATA_PROCESSED_DIR / "val.csv", index=False)
    print(f"Created train and val splits ({len(train_annotations)}/{len(val_annotations)})")


if __name__ == "__main__":
    main()
