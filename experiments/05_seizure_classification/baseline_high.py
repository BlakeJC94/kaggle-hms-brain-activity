from core.utils import import_script_as_module

hparams = import_script_as_module("./experiments/05_seizure_classification/baseline.py").hparams
hparams["config"]["learning_rate"] = 3 * 1e-3
