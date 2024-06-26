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
        "sample_rate": 200.0,
        "bandpass_low": 0.3,
        "bandpass_high": 75.0,
        "learning_rate": 1.5 * 1e-3,
        "weight_decay": 0.01,
        "num_workers": 10,
        "batch_size": 64,
        "patience": 20,
        "milestones": [5, 8],
        "gamma": 0.2,
        "monitor": "loss/validate",
    },
}
