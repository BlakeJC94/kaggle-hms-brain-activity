from pathlib import Path
import pandas as pd
import plotly.express as px
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from matplotlib import pyplot as plt

from hms_brain_activity.globals import VOTE_NAMES


# %% Load the annotations

csv_path = Path("./data/hms/train.csv")
ann = pd.read_csv(csv_path)
ann = ann[
    [
        "eeg_id",
        "eeg_label_offset_seconds",
        "label_id",
        "patient_id",
        *VOTE_NAMES,
    ]
]


# %% How imbalanced are these votes?
res = ann[VOTE_NAMES].sum()
res = res.apply(lambda x: x / res.sum().sum())

print(res.round(2))
# seizure_vote    0.12
# lpd_vote        0.16
# gpd_vote        0.17
# lrda_vote       0.13
# grda_vote       0.15
# other_vote      0.27

res.plot.bar()
plt.show()
# Not too bad actually, seems like a decent balance!


# %% How imbalanced are these probs?
res = (
    ann[VOTE_NAMES]
    .apply(lambda x: x / ann[VOTE_NAMES].sum(axis=1))
    .sum()
)
res = res.apply(lambda x: x / res.sum().sum())

print(res.round(2))
# seizure_vote    0.21
# lpd_vote        0.13
# gpd_vote        0.13
# lrda_vote       0.14
# grda_vote       0.18
# other_vote      0.21

res.plot.bar()
plt.show()
# Seems to have a nicer balance in this view!

# %% How about the majority vote dist?
res = ann[VOTE_NAMES].apply(lambda x: np.argmax(x), axis=1)
res = pd.get_dummies(res).rename(columns=dict(enumerate(VOTE_NAMES)))
res = res.astype(int).sum()
res = res.apply(lambda x: x / res.sum().sum())

print(res.round(2))
# seizure_vote    0.20
# lpd_vote        0.14
# gpd_vote        0.16
# lrda_vote       0.16
# grda_vote       0.18
# other_vote      0.18

res.plot.bar()
plt.show()
# Also pretty good here

# %% Are there any samples that have no votes?
len(ann[VOTE_NAMES][ann[VOTE_NAMES].sum(axis=1) == 0])
# Nope!

# %%
# Alrighty, I want to assign patient_ids to train/val splits, but I want to maintain the
# distribution of the majority vote for each label...

# Is this the right way to think about it?
# I'll have to stratify across a categorical variable with this method
# What if we stratify across the sorted vote winners? Insert none for cats that werent voted

# Wonder how many ties exist..

# %%
# Bugger it, here's the plan:
# - Combine df to 1 row per patient
# - Figure out the primary class per patient
# - Figure out the secondary class per patient
# - Stratify on these 2 quantities

# %% Combine votes across patients

res = ann[['patient_id', *VOTE_NAMES]]
res = res.groupby('patient_id').sum()
res["class_1"] = res.idxmax(axis=1)
# res["class_2"] = (r).idxmax(axis=1)  # TODO


# %%

random_state = 0
candidate_splits = {
    k: v
    for k, v in zip(
        ["train", "val"],
        list(
            StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=random_state
            ).split(
                np.zeros(len(studies_to_stratify)),
                studies_to_stratify[demographics_to_stratify].values,
            )
        )[0],
    )
}
