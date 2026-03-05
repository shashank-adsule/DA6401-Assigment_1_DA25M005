import os, sys
import numpy as np
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data   import prepare_data
from train        import train             # reuse the core train() function


# ── Your W&B project details ──────────────────────────────────────────────
WANDB_PROJECT = "da6401-assignment1"
WANDB_ENTITY  = None    # ← PUT YOUR W&B USERNAME HERE e.g. "john_doe"
                        #   Leave None to use your default entity

DATASET       = "mnist" # Change to "fashion_mnist" for Q2.10


# ─────────────────────────────────────────────────────────────────────────
#  Sweep Configuration
#  method: "bayes" uses Bayesian optimisation to pick configs intelligently
#          (much better than "random" or "grid" for 100 runs)
# ─────────────────────────────────────────────────────────────────────────

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "val_acc",
        "goal": "maximize"
    },
    "parameters": {
        # ── Fixed (not swept) ──────────────────────────────────────
        "dataset":      {"value": DATASET},
        "epochs":       {"value": 10},
        "loss":         {"value": "cross_entropy"},

        # ── Swept hyperparameters ──────────────────────────────────
        "batch_size": {
            "values": [32, 64, 128]
        },
        "optimizer": {
            "values": ["sgd", "momentum", "nag", "rmsprop"]
        },
        "learning_rate": {
            # log_uniform_values gives Bayes better coverage over orders of magnitude
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1
        },
        "weight_decay": {
            "values": [0.0, 0.0005, 0.005]
        },
        "num_layers": {
            "values": [2, 3, 4, 5]
        },
        "hidden_size": {
            # Single integer — will be expanded to [hidden_size] * num_layers in train()
            "values": [32, 64, 128]
        },
        "activation": {
            "values": ["relu", "sigmoid", "tanh"]
        },
        "weight_init": {
            "values": ["random", "xavier"]
        },
    }
}


# ─────────────────────────────────────────────────────────────────────────
#  Sweep agent function
#  W&B calls this function for each trial.
#  It reads wandb.config (filled by W&B) and trains one model.
# ─────────────────────────────────────────────────────────────────────────

# Load data once outside the sweep function to avoid reloading every trial
print(f"Pre-loading {DATASET}...")
DATA = prepare_data(DATASET, val_fraction=0.1)
print("Data ready.\n")


def sweep_run():
    """Called once per sweep trial by wandb.agent()."""

    # wandb.init() with no config — W&B fills wandb.config automatically
    run = wandb.init()
    cfg = dict(wandb.config)   # convert to plain dict

    # Add fields that train() needs but aren't swept
    cfg.setdefault("save_dir",     "./models")
    cfg.setdefault("val_fraction", 0.1)

    # Expand hidden_size: e.g. hidden_size=128, num_layers=3 → [128,128,128]
    if isinstance(cfg["hidden_size"], int):
        cfg["hidden_size"] = [cfg["hidden_size"]] * cfg["num_layers"]

    # Rename the run with a readable label
    run.name = (
        f"{cfg['optimizer']}_lr{cfg['learning_rate']:.4f}"
        f"_nl{cfg['num_layers']}x{cfg['hidden_size'][0]}"
        f"_{cfg['activation']}"
    )

    # Train — use_wandb=True so metrics are logged each epoch
    train(cfg, DATA, use_wandb=True)

    wandb.finish()


# ─────────────────────────────────────────────────────────────────────────
#  Launch sweep
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create the sweep on W&B servers
    sweep_id = wandb.sweep(
        sweep    = SWEEP_CONFIG,
        project  = WANDB_PROJECT,
        entity   = WANDB_ENTITY,
    )
    print(f"\nSweep created!")
    print(f"  Sweep ID : {sweep_id}")
    print(f"  View at  : https://wandb.ai/{WANDB_ENTITY or 'YOUR_USERNAME'}/{WANDB_PROJECT}/sweeps/{sweep_id}")
    print(f"\nStarting agent (will run 100 trials)...\n")

    # Start the agent — runs 100 trials sequentially
    # To run in parallel: open another terminal and run:
    #   wandb agent <entity>/<project>/<sweep_id>
    wandb.agent(
        sweep_id = sweep_id,
        function = sweep_run,
        project  = WANDB_PROJECT,
        entity   = WANDB_ENTITY,
        count    = 100,   # total trials for this agent
    )