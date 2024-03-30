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
        # "seizure_weights": "./artifacts/05_seizure_classification-baseline-v1-7a6a785852194b58b19971f61c7b5977/train/model_weights/epoch=3-step=5284.ckpt",
        # "pdrda_weights": "./artifacts/06_pdrda_classification-baseline_low-v4-e568535bf4274c94a00c2b78ba4c9dc2/train/model_weights/epoch=2-step=15852.ckpt",
        "data_dir": "./data/hms/train_eegs",
        "train_ann": "./data/processed/patient_split/train_patient_ids.csv",
        "val_ann": "./data/processed/patient_split/val_patient_ids.csv",
        "sample_rate": 200.0,
        "freq_resolution": 0.5,
        "bandpass_low": 0.5,
        "bandpass_high": 70.0,
        "learning_rate": 1.5 * 1e-3,
        "weight_decay": 0.02,
        "learning_rate_decay_epochs": 7,
        "learning_rate_min": 5 * 1e-6,
        "num_workers": 10,
        "batch_size": 32,
        "patience": 9,
        "monitor": "loss/validate",
    },
}
