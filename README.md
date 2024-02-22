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
* [x] How many channels are in the files?
* [x] Table of durations per study
* [ ] Plot some example "other" regions
* [ ] Figure out how to train/val split by patient_id and maintain label balance

Training:
* Foundation with a classification approach
    * [x] Create a torch dataset to load time series data from local directory
    * [x] Create a model scaffold with pytorch lighting
    * [x] Define a toynet for classification
    * [x] Write train script w/debug mode
    * [x] Setup tensorboard
    * [x] Define and test a basic config
    * [x] Scale data
    * [ ] Upgrade placeholder model
    * [ ] Implement random sagital flip
    * [ ] Implement a montage function
    * [ ] Implement a basic Resnet for timeseries
    * [ ] Train a basic resnet for classification to a "pretty good" degree
    * [ ] Write evaluation script and prediction writer for submission
* Classification vs Segmentation (more samples good?)
    * [ ] Expand dataset to segmentation output (will need to rework iteration)
    * [ ] Train same resnet and see how it compares


## Usage

TODO

```bash
# pipx install poetry
$ poetry install
$ poetry shell
```
