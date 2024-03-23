from core.utils import import_script_as_module

hparams = import_script_as_module("./experiments/04_kldiv_prob_crossval/finetune.py").hparams
hparams["config"]["train_ann"] = "./data/processed/two_stage_prob_cross_val/train_few_1.csv"
hparams["config"]["val_ann"] = "./data/processed/two_stage_prob_cross_val/val_few_1.csv"
