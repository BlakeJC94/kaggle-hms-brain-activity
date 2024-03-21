hparams = {
    "task": {
        "init": {
            "project_name": "HMS",
        },
        # "parent_task_id": "xxx",
    },
    "checkpoint": {
        "checkpoint_task_id": None,
        # "checkpoint_name": "last",
        # "weights_only": False,
    },
    "trainer": {
        "init": {
            # "devices": [1],
        },
    },
    "config": {
        "data_dir": "./data/hms/train_eegs",
        "train_ann": "./data/processed/two_stage_prob_cross_val/train_many.csv",
        "val_ann": "./data/processed/two_stage_prob_cross_val/val_many.csv",
        "sample_rate": 200.0,
        "freq_res": 0.5,
        "bandpass_low": 0.5,
        "bandpass_high": 70.0,
        "learning_rate": 4 * 1e-3,
        "weight_decay": 0.01,
        "num_workers": 10,
        "batch_size": 128,
        "patience": 20,
        "milestones": [20],
        "gamma": 0.2,
        "monitor": "loss/validate",
    },
}
