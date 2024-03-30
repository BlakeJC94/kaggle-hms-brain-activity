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
        "seizure_weights": "artifacts/09_seizure_classification_small-baseline_high-v1-21f4276830c543e78ad02e42c3fe34ab/train/model_weights/epoch=10-step=29062.ckpt",
        "pdrda_weights": "artifacts/10_pdrda_classification_small-baseline_high-v1-ce0c819ddee1481496dd0e75a92de45d/train/model_weights/epoch=4-step=13210.ckpt",
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
