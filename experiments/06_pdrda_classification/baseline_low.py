from core.utils import import_script_as_module

hparams = import_script_as_module("./experiments/06_pdrda_classification/baseline.py").hparams
hparams["config"]["learning_rate"] = 1 * 1e-4
