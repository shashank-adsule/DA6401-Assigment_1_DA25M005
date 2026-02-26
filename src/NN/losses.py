import numpy as np

class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def backward(self, y_true, y_pred):
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)

class CrossEntropyLoss(Loss):
    """
    Cross-entropy loss for multi-class classification.
    Assumes y_true is one-hot and y_pred is softmax output.
    """

    def forward(self, y_true, y_pred):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        return -np.sum(y_true * np.log(y_pred)) / m

    def backward(self, y_true, y_pred):
        m = y_true.shape[0]
        return (y_pred - y_true) / m

class MSELoss(Loss):
    """
    Mean squared error loss.
    """

    def forward(self, y_true, y_pred):
        m = y_true.shape[0]
        return np.sum((y_true - y_pred) ** 2) / m

    def backward(self, y_true, y_pred):
        m = y_true.shape[0]

        # dL/dy_pred
        dL_dy = 2 * (y_pred - y_true) / m

        # softmax jacobian trick
        dot = np.sum(dL_dy * y_pred, axis=1, keepdims=True)
        return y_pred * (dL_dy - dot)

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

    print("\nNOTE: CE loss is lower because predictions are fairly confident & correct.")