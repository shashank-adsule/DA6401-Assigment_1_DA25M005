"""
utils/data.py
-------------
Data loading, preprocessing, and splitting for MNIST / Fashion-MNIST.

IMPORTANT RULES FROM ASSIGNMENT:
  - Training and test datasets must be STRICTLY isolated
  - Validation split must come from training data only (NOT test set)
  - Split must be random
  - We use keras.datasets (permitted) instead of torchvision
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist   # permitted by assignment


# ─────────────────────────────────────────────
#  Load Dataset
# ─────────────────────────────────────────────

def load_dataset(name: str):
    """
    Load MNIST or Fashion-MNIST.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
      x: float64, shape (N, 784), values in [0, 1]
      y: int,     shape (N,),     values in {0..9}
    """
    name = name.lower().replace("-", "_")

    if name in ("mnist",):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif name in ("fashion_mnist", "fashion-mnist"):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose: mnist, fashion_mnist")

    # ── Normalize to [0, 1] ──
    x_train = x_train.astype(np.float64) / 255.0
    x_test  = x_test.astype(np.float64)  / 255.0

    # ── Flatten 28×28 → 784 ──
    x_train = x_train.reshape(len(x_train), -1)   # (60000, 784)
    x_test  = x_test.reshape(len(x_test),  -1)    # (10000, 784)

    return (x_train, y_train), (x_test, y_test)


# ─────────────────────────────────────────────
#  Train / Validation Split
# ─────────────────────────────────────────────

def train_val_split(x, y, val_fraction=0.1, seed=42):
    """
    Splits training data into train + validation sets.

    WHY NOT USE TEST SET AS VALIDATION?
      Test set is only touched ONCE at the very end to report final metrics.
      Using it during training leaks information → inflated accuracy scores.
      Instead, carve out val_fraction (default 10%) from training data.

    Parameters
    ----------
    x            : shape (N, 784)
    y            : shape (N,)  — integer labels
    val_fraction : float in (0, 1)  — fraction to use as validation
    seed         : int — for reproducibility

    Returns
    -------
    (x_train, y_train), (x_val, y_val)
    """
    np.random.seed(seed)
    N = len(x)
    indices = np.random.permutation(N)

    val_size   = int(N * val_fraction)
    val_idx    = indices[:val_size]
    train_idx  = indices[val_size:]

    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx])


# ─────────────────────────────────────────────
#  One-Hot Encoding
# ─────────────────────────────────────────────

def one_hot(y, num_classes=10):
    """
    Converts integer labels to one-hot encoded matrix.

    y: shape (N,)        — integer labels e.g. [3, 7, 2, ...]
    returns: shape (N, C) — one-hot matrix
    """
    N = len(y)
    ohe = np.zeros((N, num_classes))
    ohe[np.arange(N), y] = 1.0
    return ohe


# ─────────────────────────────────────────────
#  Mini-Batch Generator
# ─────────────────────────────────────────────

def get_batches(x, y_ohe, batch_size: int, shuffle: bool = True):
    """
    Yields (x_batch, y_batch) tuples for one epoch.

    Parameters
    ----------
    x         : shape (N, 784)
    y_ohe     : shape (N, C)   — one-hot encoded labels
    batch_size: int
    shuffle   : bool — shuffle before each epoch

    Yields
    ------
    (x_batch, y_batch) where x_batch.shape = (batch_size, 784)
    """
    N = len(x)
    indices = np.random.permutation(N) if shuffle else np.arange(N)

    for start in range(0, N, batch_size):
        idx = indices[start : start + batch_size]
        yield x[idx], y_ohe[idx]


# ─────────────────────────────────────────────
#  Full Pipeline Helper
# ─────────────────────────────────────────────

def prepare_data(dataset: str, val_fraction: float = 0.1, seed: int = 42):
    """
    One-call helper: load → split → one-hot encode.

    Returns
    -------
    dict with keys:
      x_train, y_train         — training inputs & one-hot labels
      x_val,   y_val           — validation inputs & one-hot labels
      x_test,  y_test          — test inputs & one-hot labels
      y_train_int, y_val_int, y_test_int  — integer labels (for metrics)
    """
    (x_all_train, y_all_train), (x_test, y_test_int) = load_dataset(dataset)

    (x_train, y_train_int), (x_val, y_val_int) = train_val_split(
        x_all_train, y_all_train, val_fraction=val_fraction, seed=seed
    )

    return {
        "x_train":     x_train,
        "y_train":     one_hot(y_train_int),
        "y_train_int": y_train_int,

        "x_val":       x_val,
        "y_val":       one_hot(y_val_int),
        "y_val_int":   y_val_int,

        "x_test":      x_test,
        "y_test":      one_hot(y_test_int),
        "y_test_int":  y_test_int,
    }


# ─────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading MNIST...")
    data = prepare_data("mnist")

    print(f"  x_train:     {data['x_train'].shape}   values [{data['x_train'].min():.1f}, {data['x_train'].max():.1f}]")
    print(f"  y_train:     {data['y_train'].shape}   (one-hot)")
    print(f"  x_val:       {data['x_val'].shape}")
    print(f"  x_test:      {data['x_test'].shape}")
    print(f"\nSample labels (first 10 train): {data['y_train_int'][:10]}")

    # Test batch generator
    batches = list(get_batches(data["x_train"], data["y_train"], batch_size=64))
    print(f"\nBatches per epoch: {len(batches)}")
    print(f"Batch shape: x={batches[0][0].shape}, y={batches[0][1].shape}")
    print("\n✅ Data pipeline ready")