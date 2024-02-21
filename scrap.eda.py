from pathlib import Path
import pandas as pd
import plotly.express as px

# %% Load the annotations

csv_path = Path("./data/hms/train.csv")

annotations = pd.read_csv(csv_path)

# Columns:
# - EEG Locations
#     - `eeg_id`
#     - `eeg_sub_id`
#     - `eeg_label_offset_seconds`
# - Spectrogram Locations
#     - `spectrogram_id`
#     - `spectrogram_sub_id`
#     - `spectrogram_label_offset_seconds`
# - `label_id`
# - `patient_id`
# - `expert_consensus`
#     - Label class
# - Votes
#     - `seizure_vote`
#     - `lpd_vote`
#     - `gpd_vote`
#     - `lrda_vote`
#     - `grda_vote`
#     - `other_vote`


# %% Try load an EEG file and see what's there

eeg_path = Path("./data/hms/train_eegs/1243339574.parquet")
data = pd.read_parquet(eeg_path)

# >>> data.columns
# Index(['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz',
#        'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG'],
#       dtype='object')
# >>> len(data)
# 10000

# %% Filter list down to available files

eeg_files = sorted(Path("./data/hms/train_eegs").glob("*.parquet"))
eeg_ids = set(int(fp.stem) for fp in eeg_files)

spect_files = sorted(Path("./data/hms/train_spectrograms").glob("*.parquet"))
spect_ids = set(int(fp.stem) for fp in spect_files)

ann = annotations[
    annotations["eeg_id"].isin(eeg_ids) & annotations["spectrogram_id"].isin(spect_ids)
]

# %% How many eeg_id appear in more than 2 rows?

eeg_ids_mult = (
    ann.groupby("eeg_id").size().where(lambda x: x > 2).dropna().sort_values()
)  # len 263

# %% What's the deal with the label_offset_seconds shit?

eeg_id = 1311572604
eeg_path = Path(f"./data/hms/train_eegs/{eeg_id}.parquet")
data = pd.read_parquet(eeg_path)

file_ann = ann[ann["eeg_id"] == eeg_id]
# Seems like this file has 6 sub_ids (0, ..., 5)
# But each label offset seconds isn't uniformly spaced.

# From the data tab on the comp:
# > All data was recorded at 200 Hz

# Seems like classification is done on 50s samples in the test dataset
# So I reckon all labels are assumed to be 50s long, and the `eeg_label_offset` is just how the
# subsampling with overlapping is done


# %% How many votes per row?

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
annotations[vote_cols].sum(axis=1).describe()
# min = 1, max = 28

# %% What's up with the submission.csv?

submission_path = Path("./data/hms/sample_submission.csv")
subm = pd.read_csv(submission_path)

# Ah alright that seems alright, simply give class probabilities for each eeg_id

# %% Inspect a spectrogram

spect_id = 137181473
spect_path = Path(f"./data/hms/train_spectrograms/{spect_id}.parquet")
spect_data = pd.read_parquet(spect_path).set_index('time')

# >>> spect_data.columns
# Index(['time', 'LL_0.59', 'LL_0.78', 'LL_0.98', 'LL_1.17', 'LL_1.37',
#        'LL_1.56', 'LL_1.76', 'LL_1.95', 'LL_2.15',
#        ...
#        'RP_18.16', 'RP_18.36', 'RP_18.55', 'RP_18.75', 'RP_18.95', 'RP_19.14',
#        'RP_19.34', 'RP_19.53', 'RP_19.73', 'RP_19.92'],
#       dtype='object', length=401)
# >>> len(spect_data)
# 305

ll_spect = spect_data[sorted((c for c in spect_data.columns if c.startswith("LL_")), reverse=True)]
ll_spect = ll_spect.rename(columns={c: float(c.removeprefix("LL_")) for c in ll_spect.columns})
px.imshow(ll_spect.T, origin='lower')

# A time column? Each spectrogram is supposed to be 10 mins, max seems to be 609, => reported values
# in seconds?

# Rather than decode this, I think I'd rather just write my own spectrogram transform
#
# Modelling approaches:
# - Time series data only:
#     - inputs are 50s @ 200Hz (EEG+ECG)
#     - Annotations are one-hot encoded
#     - outputs are class probabilities
# - Spectrgrams:
#     - sample intputs as above, but split into LL,RL,LP,RP lobes
#     - Multitaper!
#     - Sampe outputs as above
# - Try both these approaches on TUH as well
# - Try make TUH models predict age group and sex

# Note there doesn't seem to be any demographic data attached to this. Wonder if this can be
# transferred from TUH?
# Ah, also no time of day... maybe make TUH predict this too?



