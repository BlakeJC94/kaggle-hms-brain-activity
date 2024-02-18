# kaggle-hms-brain-activity

An attempt at a Kaggle competition for seizure detection.

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview

> There are six patterns of interest for this competition:
> * seizure (SZ),
> * generalized periodic discharges (GPD),
> * lateralized periodic discharges (LPD),
> * lateralized rhythmic delta activity (LRDA),
> * generalized rhythmic delta activity (GRDA),
> * “other”.
> Detailed explanations of these patterns are available [here](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf).

## TODO

EDA:
* [ ] Explore distribution of labels in the training dataset
* [ ] How many channels are in the files?
* [ ] Table of durations per study
* [ ] Plot some example "other" regions
* [ ] (optional) Figure out parameters used for the spectrogram

Training:
* [ ] Create a torch dataset to load time series data from local directory
* [ ] Create a model scaffold with pytorch lighting
* [ ] Define a toynet for segmentation
* [ ] Write train script w/debug mode
* [ ] Setup tensorboard
* [ ] Define and test a basic config
