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
            "enable_progress_bar": False,
        },
    },
    "config": {
        "sample_rate": 200.0,
        "bandpass_low": 0.3,
        "bandpass_high": 45.0,
        "learning_rate": 3 * 1e-2,
        "weight_decay": 0.01,
        "num_workers": 12,
        "batch_size": 256,
        "patience": 20,
        "milestones": [20],
        "gamma": 0.2,
        "monitor": "loss/validate",
    },
}
