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
        "train_ann": "./data/processed/multiclass_one_hot/train_0.csv",
        "val_ann": "./data/processed/multiclass_one_hot/val_0.csv",
        "sample_rate": 200.0,
        "bandpass_low": 0.3,
        "bandpass_high": 75.0,
        "learning_rate": 3 * 1e-3,
        "weight_decay": 0.01,
        "num_workers": 3,
        "batch_size": 256,
        "patience": 20,
        "milestones": [20],
        "gamma": 0.2,
        "monitor": "loss/validate",
    },
}
