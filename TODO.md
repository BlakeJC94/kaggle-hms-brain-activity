# TODO

EDA:
* [x] Explore distribution of labels in the training dataset
* [x] How many channels are in the files?
* [x] Table of durations per study
* [ ] Plot some example "other" regions
* [x] Figure out how to train/val split by patient_id and maintain label balance
    * `from sklearn.model_selection import StratifiedShuffleSplit`?

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
    * [x] Implement random sagittal flip
    * [x] Implement a montage function
    * [x] Implement filters
    * [x] Implement a basic Resnet for timeseries
    * [ ] Train a basic resnet for classification to a "pretty good" degree
    * [ ] Try downsampling the signal
    * [ ] Implement output transforms
    * [ ] Write evaluation script and prediction writer for submission
    * [ ] Track mean value for each class
    * [ ] Track mean error for each class
    * [ ] Write an asymmetric violin plot metric for classes (pred dist vs true dist)
    * [ ] Write an asymmetric violin plot metric for classes
    * [ ] Submit first attempt
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
        * 3 corpuses, overlaps between files and a lot of label combination probably needed
    * [ ] Implement TUH dataset
    * [ ] Pretrain a model using best approach so far for 1d data
    * [ ] Pretrain a model using best approach so far for 2d data
    * [ ] Pretrain a model using best approach so far for ensemble frozen
    * [ ] Pretrain a model using best approach so far for ensemble unfrozen

* Ideas:
    * Try audio-style segmentation with waveunet?



