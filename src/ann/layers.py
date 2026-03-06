import numpy as np
from .activation import get_activation


def init_weights(method: str, n_in: int, n_out: int):
    method = method.lower()

    if method == "random":
        W = np.random.randn(n_in, n_out) * 0.01
    elif method == "xavier":
        std = np.sqrt(1.0 / n_in)
        W = np.random.randn(n_in, n_out) * std
    elif method == "zeros":
        W = np.zeros((n_in, n_out))
    else:
        raise ValueError("Initialization must be 'random', 'xavier', or 'zeros'")

    b = np.zeros((1, n_out))
    return W, b


class Linear:
    """
    Fully connected layer: z = a_prev @ W + b
    """

    def __init__(self, n_in, n_out,
                 activation="relu",
                 init="random",
                 weight_decay=0.0):

        self.n_in = n_in
        self.n_out = n_out
        self.weight_decay = weight_decay

        # Output layer uses "linear" (no activation) — returns raw logits
        # is_output flag: skip activation gradient multiplication in backward
        self.is_output = (activation == "linear")

        self.W, self.b = init_weights(init, n_in, n_out)

        # For linear (output) layer we don't need an activation object
        self.activation = None if self.is_output else get_activation(activation)

        # Cache for backprop
        self.a_prev = None
        self.z = None

        # Gradients — stored here so the autograder can read them directly
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, a_prev):
        self.a_prev = a_prev
        self.z = a_prev @ self.W + self.b
        # Linear output layer: return z directly (logits)
        if self.is_output:
            return self.z
        return self.activation.forward(self.z)

    # def backward(self, delta):
    #     # For hidden layers: multiply upstream delta by activation derivative
    #     # For output (linear) layer: activation grad = 1, so delta passes through unchanged
    #     if not self.is_output:
    #         delta = delta * self.activation.backward(self.z)

    #     # Parameter gradients — /m is handled by loss.backward, not here
    #     self.grad_W = self.a_prev.T @ delta
    #     self.grad_b = np.sum(delta, axis=0, keepdims=True)

    #     # L2 regularization (only for weights, not biases)
    #     if self.weight_decay > 0:
    #         self.grad_W += self.weight_decay * self.W

    #     # Gradient to propagate to the previous layer
    #     delta_prev = delta @ self.W.T
    #     return delta_prev
    def backward(self, delta):

        if not self.is_output:
            delta = delta * self.activation.backward(self.z)

        self.grad_W = self.a_prev.T @ delta
        self.grad_b = np.sum(delta, axis=0, keepdims=True)

        # Correct L2 regularization
        if self.weight_decay > 0:
            m = self.a_prev.shape[0]
            self.grad_W += (self.weight_decay / m) * self.W

        delta_prev = delta @ self.W.T

        return delta_prev
    
    def get_params(self):
        return self.W, self.b

    def set_params(self, W, b):
        self.W = W
        self.b = b

    def __repr__(self):
        act_name = "linear" if self.is_output else self.activation.__class__.__name__
        return (f"Linear({self.n_in} → {self.n_out}, "
                f"act={act_name}, "
                f"wd={self.weight_decay})")

# NOTE: to the module file on system use `python -m NN.layers` instead of `python src/NN/layers.py from project root`

if __name__ == "__main__":
    np.random.seed(0)

    # Simulate: input layer → hidden → output
    batch_size = 4
    x = np.random.randn(batch_size, 8)       # 8 input features
    y = np.zeros((batch_size, 3))            # 3 classes, one-hot
    y[np.arange(batch_size), [0,1,2,0]] = 1

    # Layers
    from .losses import CrossEntropyLoss
    

    hidden = Linear(8, 16, activation="relu",    init="xavier")
    output = Linear(16, 3, activation="linear", init="xavier")
    loss_fn = CrossEntropyLoss()

    # Forward
    a1 = hidden.forward(x)
    a2 = output.forward(a1)

    print("=== Forward ===")
    print(f" hidden output shape: {a1.shape}")
    print(f" output probs shape:  {a2.shape}")
    print(f" probs sum:           {a2.sum(axis=1)}  ← should all be 1.0")
    print(f" loss:                {loss_fn(y, a2):.4f}")

    # Backward
    delta = loss_fn.backward(y, a2)       # δ for output layer
    delta = output.backward(delta)         # δ for hidden layer
    delta = hidden.backward(delta)         # δ for input (not used further)

    print("\n=== Backward ===")
    print(f"hidden.grad_W shape: {hidden.grad_W.shape}  ← (8, 16)")
    print(f"hidden.grad_b shape: {hidden.grad_b.shape}  ← (1, 16)")
    print(f"output.grad_W shape: {output.grad_W.shape}  ← (16, 3)")
    print(f"output.grad_b shape: {output.grad_b.shape}  ← (1, 3)")
