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
    * [x] Upgrade placeholder model
    * [ ] Implement random sagittal flip
    * [ ] Implement a montage function
    * [ ] Implement a basic Resnet for timeseries
    * [ ] Train a basic resnet for classification to a "pretty good" degree
    * [ ] Write evaluation script and prediction writer for submission
* Classification vs Segmentation (more samples good?)
    * [ ] Expand dataset to segmentation output (will need to rework iteration)
    * [ ] Train same resnet backbone
    * [ ] Tune model and see how it compares to classification
* Spectrograms
    * [ ] Implement spectrogram transform
    * [ ] Update to a 2d model with a similar backbone
    * [ ] Tune model and see how it compares to 1d models
    * [ ] Compare with multitaper spectrogram
* Ensembling
    * [ ] Freeze 1d and 2d models and train an ensemble
    * [ ] Use pretrained weights and unfreeze train an ensemble and compare
* TUH pretraining
    * [ ] Match up labels in TUH with HMS task
    * [ ] Implement TUH dataset
    * [ ] Pretrain a model using best approach so far for 1d data
    * [ ] Pretrain a model using best approach so far for 2d data
    * [ ] Pretrain a model using best approach so far for ensemble frozen
    * [ ] Pretrain a model using best approach so far for ensemble unfrozen


## Usage

TODO

```bash
# pipx install poetry
$ poetry install
$ poetry shell
```
