import numpy as np
from .layers import Linear
from .losses import get_loss


class MLP:
    """
    Modular Multi-Layer Perceptron built using Linear layers.
    """

    def __init__(self,
                 input_size: int,
                 hidden_sizes: list,
                 output_size: int,
                 activation: str = "relu",
                 loss: str = "cross_entropy",
                 weight_init: str = "xavier",
                 weight_decay: float = 0.0):

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.layers = []
        self.loss_fn = get_loss(loss)

        # Build hidden layers
        prev_size = input_size
        for h in hidden_sizes:
            self.layers.append(
                Linear(prev_size,
                       h,
                       activation=activation,
                       init=weight_init,
                       weight_decay=weight_decay)
            )
            prev_size = h

        # Output layer uses linear activation → returns raw logits
        # Softmax is applied INSIDE the loss functions (CrossEntropyLoss / MSELoss)
        # This matches the updated spec (27-02-2026): "model must return logits"
        self.layers.append(
            Linear(prev_size,
                   output_size,
                   activation="linear",
                   init=weight_init,
                   weight_decay=weight_decay)
        )

    # Forward
    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # Compute Loss
    def compute_loss(self, y_true, y_pred):
        return self.loss_fn.forward(y_true, y_pred)

    # Backward
    def backward(self, y_true, y_pred):
        """
        Performs full backpropagation.
        Does NOT update weights.
        Optimizer must handle updates.
        """
        delta = self.loss_fn.backward(y_true, y_pred)

        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    # Utilities
    def get_parameters(self):
        """Returns list of (W, b) tuples for optimizer."""
        return [layer.get_params() for layer in self.layers]

    def get_weights(self):
        """Return all weights as a dict for np.save()."""
        weights = {}
        for i, l in enumerate(self.layers):
            weights[f"layer_{i}_W"] = l.W.copy()
            weights[f"layer_{i}_b"] = l.b.copy()
        return weights

    def set_weights(self, weights):
        """Load weights from a dict (as saved by get_weights)."""
        for i, layer in enumerate(self.layers):
            layer.W = weights[f"layer_{i}_W"]
            layer.b = weights[f"layer_{i}_b"]

    def save(self, path):
        """Save weights to a .npy file."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.save(path, self.get_weights())
        print(f"  Saved model to {path}")

    def load(self, path):
        """Load weights from a .npy file."""
        data = np.load(path, allow_pickle=True).item()
        self.set_weights(data)
        print(f"  Loaded weights from {path}")

    @classmethod
    def from_config(cls, config_path):
        """Instantiate an MLP from a best_config.json file."""
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        return cls(
            input_size   = cfg.get("input_size", 784),
            hidden_sizes = cfg["hidden_size"],
            output_size  = cfg.get("output_size", 10),
            activation   = cfg.get("activation", "relu"),
            loss         = cfg.get("loss", "cross_entropy"),
            weight_init  = cfg.get("weight_init", "xavier"),
            weight_decay = cfg.get("weight_decay", 0.0),
        )

    def __repr__(self):
        sizes = []
        for layer in self.layers:
            sizes.append(layer.n_out)

        return f"MLP(layers={sizes})"

if __name__ == "__main__":
    np.random.seed(42)

    # Simulated batch
    batch_size = 5
    input_dim = 20
    num_classes = 4

    # Random input
    X = np.random.randn(batch_size, input_dim)

    # Random one-hot labels
    y = np.zeros((batch_size, num_classes))
    random_classes = np.random.randint(0, num_classes, size=batch_size)
    y[np.arange(batch_size), random_classes] = 1

    # Create model
    model = MLP(
        input_size=input_dim,
        hidden_sizes=[16, 8],
        output_size=num_classes,
        activation="relu",
        loss="cross_entropy",
        weight_init="xavier"
    )

    # Forward pass
    y_pred = model.forward(X)
    loss = model.compute_loss(y, y_pred)

    print("Forward pass:")
    print("Output shape:", y_pred.shape)
    print("Prob sums (should be 1):", y_pred.sum(axis=1))
    print("Loss:", loss)

    # Backward pass
    model.backward(y, y_pred)

    print("\nGradient check:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}:")
        print("  grad_W shape:", layer.grad_W.shape)
        print("  grad_b shape:", layer.grad_b.shape)

    print("\nTest run completed successfully.")