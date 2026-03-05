"""
utils/metrics.py
----------------
Evaluation metrics required by inference.py:
  Accuracy, Precision, Recall, F1-score (macro-averaged)

WHY MACRO AVERAGE?
  We compute precision/recall per class, then average across classes.
  This treats each class equally regardless of sample count.
  (Micro-average would weight by class frequency.)

All implemented from scratch with NumPy — no sklearn needed for core metrics.
(sklearn is only used for the confusion matrix in experiments.)
"""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of correct predictions.

    Parameters
    ----------
    y_true : shape (N,) — integer class labels
    y_pred : shape (N,) — integer predicted labels

    Returns
    -------
    float in [0, 1]
    """
    return np.mean(y_true == y_pred)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10):
    """
    Compute per-class and macro-averaged Precision, Recall, F1.

    DEFINITIONS (per class c):
      TP_c = correctly predicted as class c
      FP_c = wrongly predicted as class c (it's actually something else)
      FN_c = actually class c but predicted as something else

      Precision_c = TP_c / (TP_c + FP_c)   ← of all predicted c, how many are right?
      Recall_c    = TP_c / (TP_c + FN_c)   ← of all actual c, how many did we catch?
      F1_c        = 2 * P_c * R_c / (P_c + R_c)   ← harmonic mean

    Macro average = mean over all classes.

    Parameters
    ----------
    y_true      : shape (N,) — true integer labels
    y_pred      : shape (N,) — predicted integer labels
    num_classes : int

    Returns
    -------
    dict with keys: precision, recall, f1, per_class_precision,
                    per_class_recall, per_class_f1
    """
    precision_per_class = np.zeros(num_classes)
    recall_per_class    = np.zeros(num_classes)
    f1_per_class        = np.zeros(num_classes)

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision_per_class[c] = p
        recall_per_class[c]    = r
        f1_per_class[c]        = f

    return {
        "accuracy":           accuracy(y_true, y_pred),
        "precision":          float(np.mean(precision_per_class)),   # macro
        "recall":             float(np.mean(recall_per_class)),      # macro
        "f1":                 float(np.mean(f1_per_class)),          # macro
        "per_class_precision": precision_per_class,
        "per_class_recall":    recall_per_class,
        "per_class_f1":        f1_per_class,
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10):
    """
    Build a confusion matrix.

    confusion[i, j] = number of samples with true label i predicted as j.
    Diagonal = correct predictions, off-diagonal = errors.

    Returns
    -------
    np.ndarray shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_report(metrics: dict, class_names=None):
    """Pretty-print the metrics dict from precision_recall_f1()."""
    print(f"\n{'─'*50}")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%  (macro)")
    print(f"  Recall:    {metrics['recall']*100:.2f}%  (macro)")
    print(f"  F1-score:  {metrics['f1']*100:.2f}%  (macro)")
    print(f"{'─'*50}")

    if class_names is None:
        class_names = [str(i) for i in range(len(metrics["per_class_f1"]))]

    print(f"\n  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*52}")
    for i, name in enumerate(class_names):
        p = metrics["per_class_precision"][i]
        r = metrics["per_class_recall"][i]
        f = metrics["per_class_f1"][i]
        print(f"  {name:<20} {p*100:>9.1f}% {r*100:>9.1f}% {f*100:>9.1f}%")


# ─────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0)

    # Simulate predictions — mostly correct with some errors
    y_true = np.random.randint(0, 10, 1000)
    # Inject ~80% accuracy
    y_pred = y_true.copy()
    noise_idx = np.random.choice(1000, 200, replace=False)
    y_pred[noise_idx] = np.random.randint(0, 10, 200)

    metrics = precision_recall_f1(y_true, y_pred)
    print_report(metrics)

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion matrix shape: {cm.shape}")
    print(f"Diagonal sum (correct): {np.trace(cm)}")
    print(f"Total samples:          {cm.sum()}")
    print("\n✅ Metrics module ready")