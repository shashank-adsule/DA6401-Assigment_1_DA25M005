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
        delta = self.loss_fn.backward(y_true, y_pred)
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def get_parameters(self):
        return [layer.get_params() for layer in self.layers]

    def get_weights(self):
        weights = {}
        for i, l in enumerate(self.layers):
            weights[f"layer_{i}_W"] = l.W.copy()
            weights[f"layer_{i}_b"] = l.b.copy()
        return weights

    def set_weights(self, weights):
        for i, layer in enumerate(self.layers):
            layer.W = weights[f"layer_{i}_W"]
            layer.b = weights[f"layer_{i}_b"]

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.save(path, self.get_weights())
        print(f"  Saved model to {path}")

    def load(self, path):
        data = np.load(path, allow_pickle=True).item()
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