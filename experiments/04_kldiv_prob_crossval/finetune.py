hparams = {
    "task": {
        "init": {
            "project_name": "HMS",
        },
    },
    "checkpoint": {
        "checkpoint_task_id": "8d143b7cc6564bbf90fc6bc972bd1566",
        "checkpoint_name": "epoch=15-step=8384",
        "weights_only": True,
    },
    "trainer": {
        "init": {},
    },
    "config": {
        "data_dir": "./data/hms/train_eegs",
        "train_ann": "./data/processed/two_stage_prob_cross_val/train_few_0.csv",
        "val_ann": "./data/processed/two_stage_prob_cross_val/val_few_0.csv",
        "sample_rate": 200.0,
        "freq_resolution": 0.5,
        "bandpass_low": 0.5,
        "bandpass_high": 70.0,
        "learning_rate": 3 * 1e-4,
        "weight_decay": 0.02,
        "num_workers": 6,
        "batch_size": 128,
        "patience": 10,
        "milestones": [20],
        "gamma": 0.2,
        "monitor": "loss/validate",
    },
}
