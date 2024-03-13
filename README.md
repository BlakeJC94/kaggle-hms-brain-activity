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


## Installation

A simple git clone will suffice
```bash
$ git clone https://github.com/BlakeJC94/kaggle-hms-brain-activity
```

The structure of this repo is laid out as:
```
.
├── artifacts    # Git-ignored dir for experiment outputs
├── data         # Links and dirs for raw/processed data 
├── dataset_builders  # Scripts for processing data
├── experiments       # Directory for experiment configs/hyperparams
├── hms_brain_activity  # Core ML project code
├── scrap  # Scrap files for pre-experiment exploration and testing
├── tasks  # Scripts for ML tasks
│   ├── __init__.py
│   ├── predict.py
│   ├── stop.py
│   └── train.py
└── tests  # Unit tests
```

Experiments are arranged as
```bash
experiments/
├── 00_toy_classifier   # Numbered and titled experiment stem
│   ├── baseline.py     # Hyperparameters defined as a JSON-like python dictionary
│   └── __init__.py     # Configuration written as functions of hparams dictionaries
├── 01_resnet1d34
│   ├── baseline.py
│   ├── decrease_lr.py  # Hyperparmeter names appended to experiment name in clearml 
│   └── __init__.py
└── 02_efficientnet_spectro
    ├── baseline.py
    └── __init__.py
```


## Usage

First, download `rye`:

```bash
curl -sSf https://rye-up.com/get | bash
```

And once it's all configured and active, navigate to the project directory and create the environment:
```bash
$ rye sync
```

To launch a training job (after setting up ClearML):
```bash
$ rye run ipython -- tasks/train.py <path/to/hparams.py> [--dev-run] [--offline] [--debug]
```

To lauch an inference job:
```bash
$ rye run ipython -- tasks/predict.py <path/to/hparams.py>
```
