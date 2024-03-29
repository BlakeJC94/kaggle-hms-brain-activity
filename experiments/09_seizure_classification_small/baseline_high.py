from pathlib import Path
from core.utils import import_script_as_module

hparams = import_script_as_module(Path(__file__) / "baseline.py").hparams
hparams = import_script_as_module("./experiments/08_efficientnet_spectro_resunet/baseline.py").hparams
hparams["config"]["learning_rate"] = 3 * 1e-3
