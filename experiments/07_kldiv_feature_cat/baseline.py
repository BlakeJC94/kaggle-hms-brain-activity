hparams = {
    "task": {
        "init": {
            "project_name": "HMS",
        },
    },
    "checkpoint": {
        "checkpoint_task_id": None,
        # "checkpoint_name": "last",
        # "weights_only": False,
    },
    "trainer": {
        "init": {},
    },
    "config": {
        "seizure_weights": ...,
        "pdrda_weights": ...,
        "data_dir": "./data/hms/train_eegs",
        "train_ann": "./data/processed/patient_split/train_patient_ids.csv",
        "val_ann": "./data/processed/patient_split/val_patient_ids.csv",
        "sample_rate": 200.0,
        "freq_resolution": 0.5,
        "bandpass_low": 0.5,
        "bandpass_high": 70.0,
        "learning_rate": 5 * 1e-4,
        "weight_decay": 0.02,
        "learning_rate_decay_epochs": 7,
        "learning_rate_min": 5 * 1e-6,
        "num_workers": 10,
        "batch_size": 16,
        "patience": 9,
        "monitor": "loss/validate",
    },
}
