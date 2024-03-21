import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from hms_brain_activity.globals import VOTE_NAMES
from hms_brain_activity.paths import DATA_PROCESSED_DIR


def main():
    ann = pd.read_csv("./data/hms/train.csv")
    output_dir = DATA_PROCESSED_DIR / "prob_cross_val"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Convert to probabilities
    votes = ann[VOTE_NAMES].to_numpy()
    probs = votes / votes.sum(axis=1, keepdims=True)
    ann[VOTE_NAMES] = probs

    # One hot encoding of classes with probabilities within 0.17 (~1/6) of maximum
    max_probs = probs.max(axis=1, keepdims=1)
    mask_significant_classes = max_probs - probs < 0.17
    probs_significant = probs * mask_significant_classes
    classes = (probs_significant > 0).astype(float)

    # Fill values in annotations
    ann_classes = ann.copy()
    ann_classes[VOTE_NAMES] = classes

    # For each patient, give category based on which annotations are present
    df = ann_classes[["patient_id", *VOTE_NAMES]].groupby("patient_id").agg("any").astype(float)

    # Mask overrepresented patients
    mask_patients_overrep = (
        ann_classes.groupby("patient_id").size().sort_values() > len(ann_classes) * 0.001
    )
    patients_overrep = mask_patients_overrep[mask_patients_overrep].index.to_numpy()
    df["overrep"] = df.index.isin(patients_overrep).astype(float)

    # Cross-val splits across the 1950 patients
    X, y = df.index.to_numpy(), df.to_numpy()
    _, n_classes = y.shape
    y = (np.array([2 ** (n_classes - 1 - i) for i in range(n_classes)]) * y).sum(axis=1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        train_ann = ann[ann_classes["patient_id"].isin(X[train_index])]
        train_ann.to_csv(output_dir / f"train_{i}.csv", index=False)

        val_ann = ann[ann_classes["patient_id"].isin(X[val_index])]
        val_ann.to_csv(output_dir / f"val_{i}.csv", index=False)

        print(f"Created split {i} ({len(train_ann)}/{len(val_ann)})")


if __name__ == "__main__":
    main()
