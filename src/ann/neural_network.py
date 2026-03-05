import numpy as np
from .layers import Linear
from .losses import get_loss


class NeuralNetwork:
    """
    Modular Multi-Layer Perceptron built using Linear layers.

    Can be instantiated in two ways:
      1. NeuralNetwork(args)                    ← autograder style (args Namespace)
      2. NeuralNetwork(input_size=784, ...)     ← explicit kwargs style
    """

    def __init__(self,
                 input_size=784,
                 hidden_sizes=None,
                 output_size: int = 10,
                 activation: str = "relu",
                 loss: str = "cross_entropy",
                 weight_init: str = "xavier",
                 weight_decay: float = 0.0,
                 # Extra params accepted from args Namespace (ignored if not needed)
                 num_layers: int = None,
                 hidden_size: int = None,
                 **kwargs):   # absorb any extra Namespace fields silently

        # ── Handle args Namespace passed as first positional argument ──
        # Autograder may call: NeuralNetwork(args) where args is argparse.Namespace
        import argparse
        if isinstance(input_size, argparse.Namespace):
            args = input_size
            # Unpack all fields from the Namespace
            hidden_size  = getattr(args, "hidden_size",  128)
            num_layers   = getattr(args, "num_layers",   3)
            hidden_sizes = [hidden_size] * num_layers
            output_size  = getattr(args, "output_size",  10)
            input_size   = getattr(args, "input_size",   784)
            activation   = getattr(args, "activation",   "relu")
            loss         = getattr(args, "loss",         "cross_entropy")
            weight_init  = getattr(args, "weight_init",  "xavier")
            weight_decay = getattr(args, "weight_decay", 0.0)

        # ── Handle hidden_size (int) + num_layers instead of hidden_sizes (list) ──
        if hidden_sizes is None:
            if hidden_size is not None and num_layers is not None:
                hidden_sizes = [hidden_size] * num_layers
            elif hidden_size is not None:
                hidden_sizes = [hidden_size]
            else:
                hidden_sizes = [128]

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        # ── Normalise weight_init casing ──
        # Spec uses "Xavier" (capital X), internal uses "xavier"
        weight_init = weight_init.lower()

        # ── Normalise loss name ──
        # Spec uses "mean_squared_error", internal uses "mse"
        loss_map = {"mean_squared_error": "mse"}
        loss = loss_map.get(loss, loss)

        self.layers   = []
        self.loss_fn  = get_loss(loss)

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

        # Output layer (linear activation → raw logits)
        self.layers.append(
            Linear(prev_size,
                   output_size,
                   activation="linear",
                   init=weight_init,
                   weight_decay=weight_decay)
        )

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def compute_loss(self, y_true, y_pred):
        return self.loss_fn.forward(y_true, y_pred)

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns (grad_W, grad_b) as object arrays where index 0 = LAST layer,
        index 1 = second-to-last, etc.  — matches professor's spec exactly.
        """
        delta = self.loss_fn.backward(y_true, y_pred)

        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # Store as object arrays (index 0 = last layer)
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def get_parameters(self):
        return [layer.get_params() for layer in self.layers]

    def get_weights(self):
        """
        Returns dict with keys W0, b0, W1, b1, ...
        where 0 = first (input) layer — matches professor's spec exactly.
        """
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        """
        Loads weights from dict with keys W0, b0, W1, b1, ...
        Matches professor's spec exactly.
        """
        # Handle numpy 0-d wrapped object (from np.load)
        if isinstance(weight_dict, np.ndarray) and weight_dict.ndim == 0:
            weight_dict = weight_dict.item()

        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.save(path, self.get_weights())
        print(f"  Saved model to {path}")

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()
        self.set_weights(data)
        print(f"  Loaded weights from {path}")

    @classmethod
    def from_config(cls, config_path):
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
        sizes = [layer.n_out for layer in self.layers]
        return f"NeuralNetwork(layers={sizes})"
