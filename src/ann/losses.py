import numpy as np


def softmax(z):
    """Numerically stable row-wise softmax."""
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def backward(self, y_true, y_pred):
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)


class CrossEntropyLoss(Loss):
    """
    Softmax + Cross-entropy loss.

    IMPORTANT: y_pred is now LOGITS (raw linear output), NOT probabilities.
    Softmax is applied here internally.

    forward : L = -mean( sum_k( y_k * log(softmax(logits)_k) ) )
    backward: dL/d(logits) = softmax(logits) - y_true   (clean closed form)
    """

    def forward(self, y_true, y_pred):
        # y_pred = logits → convert to probabilities
        probs = softmax(y_pred)
        probs = np.clip(probs, 1e-9, 1 - 1e-9)
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(probs)) / m

    def backward(self, y_true, y_pred):
        # Gradient: (probs - y_true) / m  — /m lives here, NOT in layers
        probs = softmax(y_pred)
        m = y_true.shape[0]
        return (probs - y_true) / m


class MSELoss(Loss):
    """
    Softmax + MSE loss.

    IMPORTANT: y_pred is LOGITS. Softmax applied internally.

    forward : L = mean( sum_k( (softmax(logits)_k - y_k)^2 ) )
    backward: via chain rule through softmax Jacobian
    """

    def forward(self, y_true, y_pred):
        probs = softmax(y_pred)
        m = y_true.shape[0]
        return np.sum((probs - y_true) ** 2) / m

    def backward(self, y_true, y_pred):
        probs = softmax(y_pred)
        m = y_true.shape[0]
        # dL/d(probs) / m — /m lives here, NOT in layers
        dL_dp = 2 * (probs - y_true) / m
        # Chain rule through softmax Jacobian
        dot = np.sum(dL_dp * probs, axis=1, keepdims=True)
        return probs * (dL_dp - dot)

LOSSES = {
    "cross_entropy": CrossEntropyLoss,
    "ce": CrossEntropyLoss,
    "mse": MSELoss,
}

def get_loss(name):
    name = name.lower()
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'")
    return LOSSES[name]()

if __name__ == "__main__":
    np.random.seed(42)
    # Simulate batch of 4, 3 classes
    y_true = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]], dtype=float)
    y_pred = np.array([[0.8,0.1,0.1],[0.2,0.7,0.1],[0.1,0.2,0.7],[0.3,0.4,0.3]], dtype=float)

    ce = CrossEntropyLoss()
    mse = MSELoss()

    print("=== Cross Entropy ===")
    print(f"  Loss:     {ce.forward(y_true, y_pred):.4f}")
    print(f"  Gradient: {ce.backward(y_true, y_pred)}")

    print("\n=== MSE ===")
    print(f"  Loss:     {mse.forward(y_true, y_pred):.4f}")
    print(f"  Gradient: {mse.backward(y_true, y_pred)}")

    # print("\nNOTE: CE loss is lower because predictions are fairly confident & correct.")
