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
        "learning_rate": 3 * 1e-3,
        "weight_decay": 0.01,
        "num_workers": 12,
        "batch_size": 256,
        "patience": 20,
        "milestones": [20],
        "gamma": 0.2,
        "monitor": "loss/validate",
    },
}

# Smoke test
if __name__ == "__main__":
    from pathlib import Path
    exec((Path(__file__).parent / "__init__.py").open().read())
    conf = config(hparams)
    model = conf.model

    import torch
    n_timesteps = 50 * 200
    x = torch.rand(hparams["config"]["batch_size"], 20, n_timesteps)
    y = torch.rand(hparams["config"]["batch_size"], 6)

    out = model(x)
    try:
        loss = model.loss_function(out, y)
    except:
        breakpoint()

    print("Pass!")
