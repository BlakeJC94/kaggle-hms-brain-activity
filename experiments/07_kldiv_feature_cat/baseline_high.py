from core.utils import import_script_as_module

hparams = import_script_as_module("./experiments/07_kldiv_feature_cat/baseline.py").hparams
hparams["config"]["learning_rate"] = 3 * 1e-3
