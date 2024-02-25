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
def show_vote_balance(ann: pd.DataFrame):
    res = ann[VOTE_NAMES].sum()
    res = res.apply(lambda x: x / res.sum().sum())
    return res


res = show_vote_balance(ann)
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
def show_prob_balance(ann:pd.DataFrame):
    res = ann[VOTE_NAMES].apply(lambda x: x / ann[VOTE_NAMES].sum(axis=1)).sum()
    res = res.apply(lambda x: x / res.sum().sum())
    return res

res = show_prob_balance(ann)
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
def show_majority_vote_balance(ann):
    res = ann[VOTE_NAMES].apply(lambda x: np.argmax(x), axis=1)
    res = pd.get_dummies(res).rename(columns=dict(enumerate(VOTE_NAMES)))
    res = res.astype(int).sum()
    res = res.apply(lambda x: x / res.sum().sum())
    return res



res = show_majority_vote_balance(ann)
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

from sklearn.model_selection import train_test_split

res = ann[["patient_id", *VOTE_NAMES]]
res = res.groupby("patient_id").sum()
res["class_1"] = res[VOTE_NAMES].idxmax(axis=1)
res["class_2"] = (
    res[VOTE_NAMES].subtract(res[VOTE_NAMES].max(axis=1), axis=0).idxmax(axis=1)
)

res_train, res_val = train_test_split(
    res,
    test_size=0.2,
    stratify=res[["class_1", "class_2"]],
    shuffle=True,
    random_state=0,
)

ann_train = ann[ann["patient_id"].isin(res_train.index)]
ann_val = ann[ann["patient_id"].isin(res_val.index)]

# %%
res_train = show_vote_balance(ann_train)
res_val = show_vote_balance(ann_val)

res = pd.DataFrame(dict(train=res_train, val=res_val))
print(res.round(2))

res.plot.bar()
plt.show()

# fig, axs = plt.subplots(1, 2)
# fig.axes.append(res_train.plot.bar(ax=axs[0], rot=0))
# fig.axes.append(res_val.plot.bar(ax=axs[1], rot=0))
# plt.show()



# %%
res_train = show_prob_balance(ann_train)
res_val = show_prob_balance(ann_val)

res = pd.DataFrame(dict(train=res_train, val=res_val))
print(res.round(2))

res.plot.bar()
plt.show()

# %%
res_train = show_majority_vote_balance(ann_train)
res_val = show_majority_vote_balance(ann_val)

res = pd.DataFrame(dict(train=res_train, val=res_val))
print(res.round(2))

res.plot.bar()
plt.show()

