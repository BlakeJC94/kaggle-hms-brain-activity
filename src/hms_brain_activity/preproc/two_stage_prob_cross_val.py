from pathlib import Path

import pandas as pd
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from hms_brain_activity.globals import VOTE_NAMES
from hms_brain_activity.paths import DATA_PROCESSED_DIR


def main():
    ann = pd.read_csv("./data/hms/train.csv")
    output_dir = DATA_PROCESSED_DIR / Path(__file__).stem
    output_dir.mkdir(exist_ok=True, parents=True)

    patient_mean_votes = ann.groupby("patient_id").apply(
        lambda df: df[VOTE_NAMES].sum(axis=1).mean(),
        include_groups=False,
    )
    mask_ann_few = ann["patient_id"].isin(
        patient_mean_votes[patient_mean_votes < 3.5].index
    )
    ann_few = ann[mask_ann_few].copy()
    ann_many = ann[~mask_ann_few].copy()

    # When looking at the majority vote, `ann_many` has a pretty balanced dist
    # Whereas `ann_few` highly skews towards seizures and grda (and away from other and lpd)

    # split ann_many into train/val
    df = ann_many.groupby("patient_id").apply(
        lambda df: mode(df[VOTE_NAMES].to_numpy().argmax(axis=1))[0],
        include_groups=False,
    )
    X, y = df.index.to_numpy(), df.to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    train_index, val_index = next(iter(skf.split(X, y)))

    # Convert to probabilities
    votes = ann_many[VOTE_NAMES].to_numpy().copy()
    ann_many[VOTE_NAMES] = votes / votes.sum(axis=1, keepdims=True)

    train_ann_many = ann_many[ann_many["patient_id"].isin(X[train_index])]
    train_ann_many.to_csv(output_dir / "train_many.csv", index=False)

    val_ann_many = ann_many[ann_many["patient_id"].isin(X[val_index])]
    val_ann_many.to_csv(output_dir / "val_many.csv", index=False)

    print(f"Created split (many) ({len(train_ann_many)}/{len(val_ann_many)})")

    # split ann_few into 5 crossval splits
    df = ann_few.groupby("patient_id").apply(
        lambda df: mode(df[VOTE_NAMES].to_numpy().argmax(axis=1))[0],
        include_groups=False,
    )
    X, y = df.index.to_numpy(), df.to_numpy()
    # Convert to probabilities
    votes = ann_few[VOTE_NAMES].to_numpy().copy()
    ann_few[VOTE_NAMES] = votes / votes.sum(axis=1, keepdims=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        train_ann_few = ann_few[ann_few["patient_id"].isin(X[train_index])]
        train_ann_few.to_csv(output_dir / f"train_few_{i}.csv", index=False)

        val_ann_few = ann_few[ann_few["patient_id"].isin(X[val_index])]
        val_ann_few.to_csv(output_dir / f"val_few_{i}.csv", index=False)

        print(f"Created split (few) {i} ({len(train_ann_few)}/{len(val_ann_few)})")


if __name__ == "__main__":
    main()
