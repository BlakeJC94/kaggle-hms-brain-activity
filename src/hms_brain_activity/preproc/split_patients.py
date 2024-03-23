import pandas as pd
from hms_brain_activity.globals import VOTE_NAMES

ann_og = pd.read_csv("./data/hms/train.csv")

# %% Print overall stats

ann = ann_og.copy()
overall_stats = {
    "n samples": len(ann),
    "n unique labels": ann["label_id"].nunique(),
    "n unique eeg ids": ann["eeg_id"].nunique(),
    "n unique patients": ann["patient_id"].nunique(),
    "percentage by consensus": (
        (ann.groupby("expert_consensus").size() / len(ann)).round(4) * 1e2
    ),
}

for k, v in overall_stats.items():
    print(f"  {k}:")
    print(v)

# %% Clean out the useless columns

ann = ann_og.copy()

ann["sample_id"] = (
    ann["eeg_id"].astype(str) + "-" + ann["eeg_label_offset_seconds"].astype(str)
)

# Combine classes to address under-representation
ann["pd_vote"] = ann["lpd_vote"] + ann["gpd_vote"]
ann["rda_vote"] = ann["lrda_vote"] + ann["grda_vote"]

columns_to_keep = [
    "sample_id",
    # "eeg_id",
    # 'eeg_sub_id',
    # "eeg_label_offset_seconds",
    # 'spectrogram_id',
    # 'spectrogram_sub_id',
    # 'spectrogram_label_offset_seconds',
    # 'label_id',
    "patient_id",
    # "expert_consensus",
    "seizure_vote",
    # "lpd_vote",
    # "gpd_vote",
    "pd_vote",
    # "lrda_vote",
    # "grda_vote",
    "rda_vote",
    "other_vote",
]

ann = ann[columns_to_keep].copy()
ann = ann.set_index("sample_id")
vote_names = [
    "seizure_vote",
    "pd_vote",
    "rda_vote",
    "other_vote",
]


# %% Do any samples have a tie?

votes = ann[vote_names].to_numpy()

mask = votes == votes.max(axis=1, keepdims=True)
index_mask = mask.sum(axis=1) > 1

n_ties = index_mask.sum()
print(f"There are {n_ties} samples with ties ({round(1e2 * n_ties / len(ann), 2)}%)")

ann_ties = ann.loc[index_mask]

ann_ties.head(15)


# %% How many ties in each sub-problem?


def ties_mask(votes):
    votes = votes.to_numpy()
    mask = votes == votes.max(axis=1, keepdims=True)
    return mask.sum(axis=1) > 1


votes_seizure = ann[vote_names].copy()
votes_seizure["not_seizure_vote"] = votes_seizure[
    [c for c in vote_names if not c.startswith("seizure")]
].sum(axis=1)
votes_seizure = votes_seizure[["seizure_vote", "not_seizure_vote"]]

index_mask_sz = ties_mask(votes_seizure)
n_ties = index_mask_sz.sum()
print(f"There are {n_ties} seizure samples with ties ({round(1e2 * n_ties / len(ann), 2)}%)")

votes_pd = ann[vote_names].copy()
votes_pd["not_pd_vote"] = votes_pd[
    [c for c in vote_names if not c.startswith("pd")]
].sum(axis=1)
votes_pd = votes_pd[["pd_vote", "not_pd_vote"]]

index_mask_pd = ties_mask(votes_pd)
n_ties = index_mask_pd.sum()
print(f"There are {n_ties} pd samples with ties ({round(1e2 * n_ties / len(ann), 2)}%)")

votes_rda = ann[vote_names].copy()
votes_rda["not_rda_vote"] = votes_rda[
    [c for c in vote_names if not c.startswith("rda")]
].sum(axis=1)
votes_rda = votes_rda[["rda_vote", "not_rda_vote"]]

index_mask_rda = ties_mask(votes_rda)
n_ties = index_mask_rda.sum()
print(f"There are {n_ties} rda samples with ties ({round(1e2 * n_ties / len(ann), 2)}%)")

# %% Classify each sample by max mask across votes

cls_cols = [c.replace("vote", "cls") for c in vote_names]

ann[cls_cols] = mask.astype(int)

ann_cls = ann[["patient_id", *cls_cols]].copy()


# %% Groupby patient and one-hot multiclass encode based on any sample

pts = ann_cls.groupby("patient_id").agg(
    {c: lambda x: (x.sum() > 0).astype(int) for c in cls_cols}
)
pts["n_samples"] = ann_cls.groupby("patient_id").size()

# import plotly.express as px
# px.histogram(pts["n_samples"])

# Add another class for over representation
pts["overrep"] = (pts["n_samples"] > pts["n_samples"].median()).astype(int)

# Sort cols
pts = pts[
    [
        "n_samples",
        "overrep",
        *cls_cols,
    ]
]

# %% Any patients with only "other"?

mask = (pts["other_cls"] == 1) & (
    pts[[c for c in cls_cols if c != "other_cls"]].sum(axis=1) == 0
)

print(
    f"N patients containing samples with only 'other': {mask.sum()} ({round(1e2 * mask.sum() / len(pts), 2)}%)"
)


# %% Encode into ordinal

import numpy as np

select_cols = ["overrep", *cls_cols]
n_cols = len(select_cols)

base = np.array([2 ** (n_cols - 1 - i) for i in range(n_cols)])
pts["cls"] = (base * pts[select_cols].to_numpy()).sum(axis=1)

cls_counts = pts["cls"].value_counts()
underrep_clss = cls_counts.index[cls_counts == 1]
print(
    f"Out of {pts['cls'].nunique()} classes, {len(underrep_clss)} have only 1 patient"
)

underrep_clss = cls_counts.index[cls_counts < 5]
print(
    f"Out of {pts['cls'].nunique()} classes, {len(underrep_clss)} have less than 5 patients"
)

# %% Split patients into train/val sets

from sklearn.model_selection import train_test_split

test_size = 0.22
random_state = 1234

pt_ids_train, pt_ids_val = train_test_split(
    pts.index.to_numpy(),
    test_size=test_size,
    stratify=pts["cls"].to_numpy(),
    shuffle=True,
    random_state=random_state,
)

ann_train, ann_val = (
    ann[ann["patient_id"].isin(pt_ids_train)],
    ann[ann["patient_id"].isin(pt_ids_val)],
)

split_stats = {
    "sample_val_split_percent": (len(ann_val) / (len(ann_train) + len(ann_val))),
    "patient_val_split_percent": (
        len(pt_ids_val) / (len(pt_ids_train) + len(pt_ids_val))
    ),
}
for k, v in split_stats.items():
    print(f"  {k}:")
    print(v)

# px.histogram(ann_val[cls_cols],  barmode="group", title="val")
# px.histogram(ann_train[cls_cols],  barmode="group", title="train")

# %% Save id lists
# Then transform the annotations in the config

from hms_brain_activity.paths import DATA_PROCESSED_DIR

output_dir = DATA_PROCESSED_DIR / "patient_split"
output_dir.mkdir(exist_ok=True)
pd.DataFrame({"patient_id": pt_ids_train}).to_csv(output_dir / "train_patient_ids.csv")
pd.DataFrame({"patient_id": pt_ids_val}).to_csv(output_dir / "val_patient_ids.csv")
