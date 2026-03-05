import argparse
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from ann.optimizers     import get_optimizer
from utils.data         import prepare_data, get_batches
from utils.metric       import precision_recall_f1, print_report


# ─────────────────────────────────────────────────────────────────
#  Argument Parser  —  flags match professor's spec EXACTLY
# ─────────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required W&B args
    parser.add_argument("-wp",   "--wandb_project",  type=str,
                        help="W&B project name")
    parser.add_argument("-we",   "--wandb_entity",   type=str,
                        help="W&B entity (username or team)")
    # parser.add_argument("-wp",   "--wandb_project",  type=str,   required=True,
    #                     help="W&B project name")
    # parser.add_argument("-we",   "--wandb_entity",   type=str,   required=True,
    #                     help="W&B entity (username or team)")

    # Dataset / training
    parser.add_argument("-d",    "--dataset",        type=str,   default="fashion_mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",    "--epochs",         type=int,   default=10)
    parser.add_argument("-b",    "--batch_size",     type=int,   default=64)
    parser.add_argument("-l",    "--loss",           type=str,   default="cross_entropy",
                        choices=["cross_entropy", "mean_squared_error"])
    parser.add_argument("-o",    "--optimizer",      type=str,   default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr",   "--learning_rate",  type=float, default=0.001)

    # Optimizer hyperparams
    parser.add_argument("-m",    "--momentum",       type=float, default=0.9)
    parser.add_argument("-beta", "--beta",           type=float, default=0.9)
    parser.add_argument("-beta1","--beta1",          type=float, default=0.9)
    parser.add_argument("-beta2","--beta2",          type=float, default=0.999)
    parser.add_argument("-eps",  "--epsilon",        type=float, default=1e-8)

    # Regularisation & init
    parser.add_argument("-w_d",  "--weight_decay",   type=float, default=0.0)
    parser.add_argument("-w_i",  "--weight_init",    type=str,   default="Xavier",
                        choices=["random", "Xavier"])

    # Architecture
    parser.add_argument("-nhl",  "--num_layers",     type=int,   default=3)
    parser.add_argument("-sz",   "--hidden_size",    type=int,   default=128)
    parser.add_argument("-a",    "--activation",     type=str,   default="relu",
                        choices=["sigmoid", "tanh", "relu"])

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
#  Core training function
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
    (model, test_metrics)
    """

    # hidden_size is now a single int → expand to list of num_layers
    hidden_size = config["hidden_size"]
    num_layers  = config["num_layers"]
    hidden = [hidden_size] * num_layers

    # Map loss name: gradescope uses "mean_squared_error", internal uses "mse"
    loss_map = {"cross_entropy": "cross_entropy", "mean_squared_error": "mse"}
    loss_key = loss_map.get(config["loss"], config["loss"])

    # Map weight_init: spec uses "Xavier" (capital X), internal uses "xavier"
    weight_init = config["weight_init"].lower()

    # Build model
    model = NeuralNetwork(
        input_size   = 784,
        hidden_sizes = hidden,
        output_size  = 10,
        activation   = config["activation"],
        loss         = loss_key,
        weight_init  = weight_init,
        weight_decay = config["weight_decay"],
    )

    # Build optimizer with all relevant hyperparams
    opt_name = config["optimizer"]
    if opt_name == "sgd":
        optimizer = get_optimizer(opt_name, lr=config["learning_rate"])
    elif opt_name in ("momentum", "nag"):
        optimizer = get_optimizer(opt_name, lr=config["learning_rate"],
                                  beta=config["momentum"])
    elif opt_name == "rmsprop":
        optimizer = get_optimizer(opt_name, lr=config["learning_rate"],
                                  beta=config["beta"], eps=config["epsilon"])
    elif opt_name in ("adam", "nadam"):
        optimizer = get_optimizer(opt_name, lr=config["learning_rate"],
                                  beta1=config["beta1"], beta2=config["beta2"],
                                  eps=config["epsilon"])
    else:
        optimizer = get_optimizer(opt_name, lr=config["learning_rate"])

    save_dir    = config.get("save_dir", "./models")
    os.makedirs(save_dir, exist_ok=True)
    save_path   = os.path.join(save_dir, "best_model.npy")
    config_path = os.path.join(save_dir, "best_config.json")

    best_val_f1 = -1.0

    for epoch in range(config["epochs"]):

        # ── Training loop ─────────────────────────────────────────
        train_losses  = []
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
        val_loss    = float(model.compute_loss(data["y_val"], val_logits))

        print(
            f"Epoch {epoch+1:>3}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} | val_acc={val_metrics['accuracy']*100:.2f}% | "
            f"val_f1={val_metrics['f1']*100:.2f}%"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "epoch":         epoch + 1,
                "train_loss":    train_loss,
                "train_acc":     train_acc,
                "val_loss":      val_loss,
                "val_acc":       val_metrics["accuracy"],
                "val_f1":        val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall":    val_metrics["recall"],
            })

        # ── Save best model ───────────────────────────────────────
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            model.save(save_path)
            best_cfg = {
                "input_size":    784,
                "output_size":   10,
                "hidden_size":   hidden,
                "num_layers":    len(hidden),
                "activation":    config["activation"],
                "loss":          loss_key,
                "weight_init":   weight_init,
                "weight_decay":  config["weight_decay"],
                "optimizer":     config["optimizer"],
                "learning_rate": config["learning_rate"],
                "batch_size":    config["batch_size"],
                "epochs":        config["epochs"],
                "dataset":       config["dataset"],
            }
            with open(config_path, "w") as f:
                json.dump(best_cfg, f, indent=2)

    print(f"\nBest validation F1: {best_val_f1*100:.2f}%")

    # ── Final test evaluation ─────────────────────────────────────
    test_logits  = model.forward(data["x_test"])
    test_preds   = np.argmax(test_logits, axis=1)
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
#  Main entry point
# ─────────────────────────────────────────────────────────────────

def main():
    args   = parse_arguments()
    config = vars(args)

    import wandb
    wandb.init(
        project = config["wandb_project"],
        entity  = config["wandb_entity"],
        config  = {k: v for k, v in config.items()
                   if k not in ("wandb_project", "wandb_entity")},
        name    = (f"{config['optimizer']}_lr{config['learning_rate']}"
                   f"_hl{config['num_layers']}x{config['hidden_size']}"
                   f"_{config['activation']}"),
    )

    config["save_dir"]     = "./models"
    config["val_fraction"] = 0.1

    print(f"Loading {config['dataset']}...")
    data = prepare_data(config["dataset"], val_fraction=config["val_fraction"])

    model, test_metrics = train(config, data, use_wandb=True)

    print("\nTest set evaluation:")
    print_report(test_metrics)

    wandb.finish()


if __name__ == "__main__":
    main()