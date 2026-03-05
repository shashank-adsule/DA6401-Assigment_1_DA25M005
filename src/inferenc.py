import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import MLP
from utils.data        import prepare_data
from utils.metric      import precision_recall_f1, confusion_matrix, print_report


# ─────────────────────────────────────────────
#  Argument Parser
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference with a saved MLP model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # p.add_argument("--weights",  type=str,default=r"D:\code\repo\M.tech\sem2\DL\assignments\project1\assets\models\best_model.npy", required=True,
    p.add_argument("--weights",  type=str,default=r"D:\code\repo\M.tech\sem2\DL\assignments\project1\assets\models\best_model.npy",
                   help="Path to .npy weights file (e.g. best_model.npy)")
    p.add_argument("--config",   type=str,default=r"D:\code\repo\M.tech\sem2\DL\assignments\project1\assets\configs\best_config.json",
    # p.add_argument("--config",   type=str,default=r"D:\code\repo\M.tech\sem2\DL\assignments\project1\assets\configs\best_config.json", required=True,
                   help="Path to model config JSON (e.g. best_config.json)")
    p.add_argument("-d", "--dataset", type=str, default="mnist",
                   choices=["mnist","fashion_mnist"],
                   help="Dataset to evaluate on")
    p.add_argument("--split",    type=str, default="test",
                   choices=["train","val","test"],
                   help="Which data split to evaluate")
    p.add_argument("--output",   type=str, default=None,
                   help="Optional: path to save metrics as JSON")
    p.add_argument("--image",    type=str, default=None,
                   help="Optional: path to a single .npy image (784,) for single prediction")
    p.add_argument("--wandb",    action="store_true",
                   help="Log results to W&B")

    return p.parse_args()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load model from config + weights ─────────────────────────
    print(f"📦 Loading model from {args.config}...")
    model = MLP.from_config(args.config)
    model.load(args.weights)
    print(model)

    # ── Single image mode ─────────────────────────────────────────
    if args.image is not None:
        img = np.load(args.image)
        if img.ndim > 1:
            img = img.flatten()
        img = img.reshape(1, -1).astype(np.float64)
        if img.max() > 1.0:
            img /= 255.0

        logits = model.forward(img)
        # Apply softmax for human-readable confidence scores (display only)
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        pred  = int(np.argmax(probs))
        conf  = float(probs[0, pred])

        FASHION_CLASSES = [
            "T-shirt","Trouser","Pullover","Dress","Coat",
            "Sandal","Shirt","Sneaker","Bag","Ankle boot"
        ]
        label = FASHION_CLASSES[pred] if "fashion" in args.dataset else str(pred)

        print(f"\n🔍 Prediction: {label}  (confidence: {conf*100:.1f}%)")
        print(f"   Full probability distribution:")
        for i, p in enumerate(probs[0]):
            bar = "█" * int(p * 30)
            print(f"   {i:>2}: {p*100:>5.1f}%  {bar}")
        return

    # ── Full dataset evaluation ───────────────────────────────────
    print(f"\n📂 Loading {args.dataset} ({args.split} split)...")
    data = prepare_data(args.dataset, val_fraction=0.1)

    # Select split
    split_map = {
        "train": (data["x_train"], data["y_train_int"]),
        "val":   (data["x_val"],   data["y_val_int"]),
        "test":  (data["x_test"],  data["y_test_int"]),
    }
    x_eval, y_true = split_map[args.split]
    print(f"   Samples: {len(x_eval)}")

    # ── Run inference in batches (avoids OOM on large sets) ──────
    BATCH = 512
    all_preds = []
    for start in range(0, len(x_eval), BATCH):
        x_batch  = x_eval[start : start + BATCH]
        logits   = model.forward(x_batch)      # returns logits — argmax same as on probs
        preds    = np.argmax(logits, axis=1)
        all_preds.append(preds)

    y_pred = np.concatenate(all_preds)

    # ── Compute metrics ───────────────────────────────────────────
    MNIST_CLASSES   = [str(i) for i in range(10)]
    FASHION_CLASSES = [
        "T-shirt","Trouser","Pullover","Dress","Coat",
        "Sandal","Shirt","Sneaker","Bag","Ankle boot"
    ]
    class_names = FASHION_CLASSES if "fashion" in args.dataset else MNIST_CLASSES

    metrics = precision_recall_f1(y_true, y_pred, num_classes=10)

    print(f"\n{'═'*50}")
    print(f"  EVALUATION RESULTS  ({args.dataset.upper()} / {args.split})")
    print_report(metrics, class_names)

    # ── Confusion matrix ─────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, num_classes=10)
    print(f"\nConfusion Matrix:")
    print(f"{'':>12}", end="")
    for name in class_names:
        print(f"{name[:6]:>8}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:<10}", end="")
        for val in row:
            print(f"{val:>8}", end="")
        print()

    # ── Save metrics to JSON if requested ────────────────────────
    output_metrics = {
        "dataset":   args.dataset,
        "split":     args.split,
        "accuracy":  metrics["accuracy"],
        "precision": metrics["precision"],
        "recall":    metrics["recall"],
        "f1":        metrics["f1"],
        "per_class": {
            class_names[i]: {
                "precision": float(metrics["per_class_precision"][i]),
                "recall":    float(metrics["per_class_recall"][i]),
                "f1":        float(metrics["per_class_f1"][i]),
            }
            for i in range(10)
        }
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_metrics, f, indent=2)
        print(f"\n💾 Metrics saved to {args.output}")

    # ── W&B logging ──────────────────────────────────────────────
    if args.wandb:
        import wandb
        wandb.init(project="da6401-assignment1", job_type="inference")
        wandb.log({
            "test_accuracy":  metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall":    metrics["recall"],
            "test_f1":        metrics["f1"],
        })
        # Log confusion matrix as W&B artifact
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true.tolist(),
                preds=y_pred.tolist(),
                class_names=class_names
            )
        })
        wandb.finish()

    return output_metrics


if __name__ == "__main__":
    main()