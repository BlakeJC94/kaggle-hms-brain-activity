from core.utils import import_script_as_module

hparams = import_script_as_module("./experiments/02_efficientnet_spectro/baseline.py").hparams
hparams["config"]["learning_rate"] = 1 * 1e-4
