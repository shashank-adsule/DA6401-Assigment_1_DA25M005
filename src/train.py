import argparse
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from ann.optimizers     import get_optimizer
from utils.data        import prepare_data, get_batches
from utils.metric      import precision_recall_f1, print_report


# ─────────────────────────────────────────────────────────────────
#  Argument Parser
# ─────────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-d",    "--dataset",       default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",    "--epochs",         type=int,   default=[10,25,50,100][0])
    parser.add_argument("-b",    "--batch_size",     type=int,   default=64)
    parser.add_argument("-l",    "--loss",           default="cross_entropy",
                        choices=["cross_entropy", "mse"])
    parser.add_argument("-o",    "--optimizer",      default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr",   "--learning_rate",  type=float, default=0.001)
    parser.add_argument("-wd",   "--weight_decay",   type=float, default=0.0)
    parser.add_argument("-nhl",  "--num_layers",     type=int,   default=3)
    parser.add_argument("-sz",   "--hidden_size",    nargs="+",  type=int,
                        default=[128, 128, 128])
    parser.add_argument("-a",    "--activation",     default="relu",
                        choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-wi",   "--weight_init",    default="xavier",
                        choices=["random", "xavier", "zeros"])
    parser.add_argument("-w_p",  "--wandb_project",  type=str,
                        default="da6401-assignment1",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (username or team). Leave blank to use default.")
    parser.add_argument("--no_wandb",     action="store_true",
                        help="Disable W&B logging (useful for quick local tests)")
    parser.add_argument("--save_dir",     default=["./models"][0])
    parser.add_argument("--val_fraction", type=float, default=0.1)

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
#  Core training function
#  Accepts a plain dict so it can be called from sweep_agent too
# ─────────────────────────────────────────────────────────────────

def train(config, data, use_wandb=False):
    """
    Train one model run with the given config dict.

    Parameters
    ----------
    config    : dict  – all hyperparameters
    data      : dict  – output of prepare_data()
    use_wandb : bool  – whether to log metrics to W&B

    Returns
    -------
    trained NeuralNetwork model
    """

    # Resolve hidden layer sizes
    # If user passes -sz 128 (single value) and -nhl 3, expand to [128, 128, 128]
    hidden = config["hidden_size"]
    if isinstance(hidden, int):
        hidden = [hidden]
    if len(hidden) == 1:
        hidden = hidden * config["num_layers"]

    # Build model
    model = NeuralNetwork(
        input_size   = 784,
        hidden_sizes = hidden,
        output_size  = 10,
        activation   = config["activation"],
        loss         = config["loss"],
        weight_init  = config["weight_init"],
        weight_decay = config["weight_decay"],
    )

    optimizer = get_optimizer(config["optimizer"], lr=config["learning_rate"])

    best_val_f1 = -1.0
    os.makedirs(config["save_dir"], exist_ok=True)
    save_path   = os.path.join(config["save_dir"],"best_model.npy")
    config_path = os.path.join(config["save_dir"],"best_config.json")
    # save_path   = os.path.join(config["save_dir"],"models", "best_model.npy")
    # config_path = os.path.join(config["save_dir"],"configs", "best_config.json")

    for epoch in range(config["epochs"]):

        # ── Training loop ─────────────────────────────────────────
        train_losses = []
        train_correct = 0
        train_total   = 0

        for x_batch, y_batch in get_batches(
            data["x_train"], data["y_train"],
            batch_size = config["batch_size"],
            shuffle    = True,
        ):
            logits = model.forward(x_batch)
            loss   = model.compute_loss(y_batch, logits)
            train_losses.append(loss)

            # Track training accuracy
            preds = np.argmax(logits, axis=1)
            train_correct += np.sum(preds == np.argmax(y_batch, axis=1))
            train_total   += len(preds)

            model.backward(y_batch, logits)
            optimizer.step(model.layers)

        train_loss = float(np.mean(train_losses))
        train_acc  = train_correct / train_total

        # ── Validation ────────────────────────────────────────────
        val_logits  = model.forward(data["x_val"])
        val_preds   = np.argmax(val_logits, axis=1)
        val_metrics = precision_recall_f1(data["y_val_int"], val_preds)

        # Compute val loss too (for W&B curves)
        val_loss = float(model.compute_loss(data["y_val"], val_logits))

        print(
            f"Epoch {epoch+1:>3}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} | val_acc={val_metrics['accuracy']*100:.2f}% | "
            f"val_f1={val_metrics['f1']*100:.2f}%"
        )

        # ── Log to W&B ────────────────────────────────────────────
        if use_wandb:
            import wandb
            wandb.log({
                "epoch":      epoch + 1,
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_metrics["accuracy"],
                "val_f1":     val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall":    val_metrics["recall"],
            })

        # ── Save best model ───────────────────────────────────────
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            model.save(save_path)
            best_config = {
                "input_size":    784,
                "output_size":   10,
                "hidden_size":   hidden,
                "num_layers":    len(hidden),
                "activation":    config["activation"],
                "loss":          config["loss"],
                "weight_init":   config["weight_init"],
                "weight_decay":  config["weight_decay"],
                "optimizer":     config["optimizer"],
                "learning_rate": config["learning_rate"],
                "batch_size":    config["batch_size"],
                "epochs":        config["epochs"],
                "dataset":       config["dataset"],
            }
            with open(config_path, "w") as f:
                json.dump(best_config, f, indent=2)

    print(f"\nBest validation F1: {best_val_f1*100:.2f}%")

    # ── Final test evaluation ─────────────────────────────────────
    test_logits = model.forward(data["x_test"])
    test_preds  = np.argmax(test_logits, axis=1)
    test_metrics = precision_recall_f1(data["y_test_int"], test_preds)

    if use_wandb:
        import wandb
        wandb.log({
            "test_acc":       test_metrics["accuracy"],
            "test_f1":        test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall":    test_metrics["recall"],
        })

    return model, test_metrics


# ─────────────────────────────────────────────────────────────────
#  Main entry point (direct CLI run)
# ─────────────────────────────────────────────────────────────────

def main():
    args   = parse_arguments()
    config = vars(args)

    use_wandb = not config.pop("no_wandb")

    # ── Initialise W&B run ────────────────────────────────────────
    if use_wandb:
        import wandb
        wandb.init(
            project = config["wandb_project"],
            entity  = config.get("wandb_entity"),   # None → uses your default entity
            config  = {k: v for k, v in config.items()
                       if k not in ("save_dir", "val_fraction", "wandb_project", "wandb_entity")},
            name    = (f"{config['optimizer']}_lr{config['learning_rate']}"
                       f"_hl{config['num_layers']}x{config['hidden_size'][0]}"
                       f"_{config['activation']}"),
        )

    print(f"Loading {config['dataset']}...")
    data = prepare_data(config["dataset"], val_fraction=config["val_fraction"])

    model, test_metrics = train(config, data, use_wandb=use_wandb)

    print("\nTest set evaluation:")
    print_report(test_metrics)

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()