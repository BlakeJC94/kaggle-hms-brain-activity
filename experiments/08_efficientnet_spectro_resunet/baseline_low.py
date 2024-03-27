from core.utils import import_script_as_module

hparams = import_script_as_module("./experiments/08_efficientnet_spectro_resunet/baseline.py").hparams
hparams["config"]["learning_rate"] = 1 * 1e-4
