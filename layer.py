from activations import ActivationFunction, Softmax
from typing import Tuple
import numpy as np

class Layer():
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        activation: ActivationFunction,
        dropout_rate: float = 0.0 
    ):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.training = True

        # Glorot/Xavier initialization
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        self.W = np.random.normal(loc=0.0, scale=scale, size=(fan_in, fan_out))
        self.b = np.zeros(fan_out)  # Initialize biases to zero
        self.delta = None
        
        self.dropout_mask = None  # To store the dropout mask
        self.outputs = None

    def forward(self, h: np.ndarray) -> np.ndarray:
        assert len(h.shape) == 1, f"Expected 1D array, got {len(h.shape)}D array"
        assert h.shape[0] == self.fan_in, f"Expected input size {self.fan_in}, got {h.shape[0]}"

        self.z = np.dot(h, self.W) + self.b
        if self.training and isinstance(self.activation, Softmax):
            pre_activation = self.z
        else:
            pre_activation = self.activation.forward(self.z)
        
        # Apply dropout if enabled and during training
        if self.training and self.dropout_rate > 0.0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=pre_activation.shape)
            self.outputs = pre_activation * self.dropout_mask / (1 - self.dropout_rate)
        else:
            self.outputs = pre_activation
        return self.outputs

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert delta is not None, "No delta term found"
        assert len(h.shape) == 1, f"Expected 1D array, got {len(h.shape)}D array"
        assert h.shape[0] == self.fan_in, f"Expected input size {self.fan_in}, got {h.shape[0]}"
		
        # Backpropagate dropout mask if it was used during forward pass
        if self.training and self.dropout_rate > 0.0 and self.dropout_mask is not None:
            delta = delta * self.dropout_mask / (1 - self.dropout_rate)
        
        if self.training and isinstance(self.activation, Softmax):
            dz = delta
        else:
            dz = delta * self.activation.derivative(self.z)
        
        dW = np.outer(h, dz)
        db = dz
      
        self.delta = np.dot(self.W, dz)
      
        return (dW, db)
    
    def __repr__(self):
        other_args = [
            "dropout_rate=" + str(self.dropout_rate) if self.dropout_rate != 0.0 else ""
        ]
        other_args_str = ", " + ", ".join([arg for arg in other_args if arg]) if "".join(other_args) != "" else ""
        activation_str = self.activation.__class__.__name__ + "()"
        
        return f"Layer(fan_in={self.fan_in}, fan_out={self.fan_out}, activation={activation_str}{other_args_str})"


class LayerBatchNormed(Layer):
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        activation: ActivationFunction,
        dropout_rate: float = 0.0,
        use_batchnorm: bool = False,      # New flag
        bn_momentum: float = 0.9          # Momentum for running stats
    ):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.training = True

        # Glorot/Xavier initialization
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        self.W = np.random.normal(loc=0.0, scale=scale, size=(fan_in, fan_out))
        self.b = np.zeros(fan_out)
        self.delta = None

        # Dropout mask
        self.dropout_mask = None  
        self.outputs = None

        # BatchNorm parameters
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.gamma = np.ones(fan_out)
            self.beta = np.zeros(fan_out)
            # Running mean and variance for inference
            self.running_mean = np.zeros(fan_out)
            self.running_var = np.zeros(fan_out)
            self.bn_momentum = bn_momentum
            # To store intermediate variables for backprop (if needed)
            self.bn_cache = {}

    def forward(self, h: np.ndarray) -> np.ndarray:
        assert len(h.shape) == 1, f"Expected 1D array, got {len(h.shape)}D array"
        assert h.shape[0] == self.fan_in, f"Expected input size {self.fan_in}, got {h.shape[0]}"

        # Compute pre-activation
        self.z = np.dot(h, self.W) + self.b

        # Apply batch normalization if enabled
        if self.use_batchnorm:
            epsilon = 1e-8
            if self.training:
                batch_mean = self.z.mean(axis=0)
                batch_var = self.z.var(axis=0)
                # Cache variables needed for potential backward pass
                self.bn_cache = {
                    "z": self.z,
                    "mean": batch_mean,
                    "var": batch_var,
                }
                # Update running statistics
                self.running_mean = self.bn_momentum * self.running_mean + (1 - self.bn_momentum) * batch_mean
                self.running_var = self.bn_momentum * self.running_var + (1 - self.bn_momentum) * batch_var
                # Normalize
                z_norm = (self.z - batch_mean) / np.sqrt(batch_var + epsilon)
            else:
                # Use running stats for inference
                z_norm = (self.z - self.running_mean) / np.sqrt(self.running_var + epsilon)
            # Scale and shift
            self.z = self.gamma * z_norm + self.beta

        # Apply activation
        if self.training and isinstance(self.activation, Softmax):
            pre_activation = self.z
        else:
            pre_activation = self.activation.forward(self.z)
        
        # Apply dropout if enabled during training
        if self.training and self.dropout_rate > 0.0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=pre_activation.shape)
            self.outputs = pre_activation * self.dropout_mask / (1 - self.dropout_rate)
        else:
            self.outputs = pre_activation

        return self.outputs

    # def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     # Note: Backward pass will need to incorporate the batch norm derivative if enabled.
    #     # For brevity, the standard computation is provided without BN backprop.
    #     assert delta is not None, "No delta term found"
    #     assert len(h.shape) == 1, f"Expected 1D array, got {len(h.shape)}D array"
    #     assert h.shape[0] == self.fan_in, f"Expected input size {self.fan_in}, got {h.shape[0]}"

    #     # If dropout was applied, propagate the mask
    #     if self.training and self.dropout_rate > 0.0 and self.dropout_mask is not None:
    #         delta = delta * self.dropout_mask / (1 - self.dropout_rate)
        
    #     if self.training and isinstance(self.activation, Softmax):
    #         dz = delta
    #     else:
    #         dz = delta * self.activation.derivative(self.z)

    #     dW = np.outer(h, dz)
    #     db = dz

    #     self.delta = np.dot(self.W, dz)

    #     return (dW, db)
    
    
    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Ensure delta and input dimensions are valid.
        assert delta is not None, "No delta term found"
        assert len(h.shape) == 1, f"Expected 1D array, got {len(h.shape)}D array"
        assert h.shape[0] == self.fan_in, f"Expected input size {self.fan_in}, got {h.shape[0]}"

        # Propagate through dropout if it was applied.
        if self.training and self.dropout_rate > 0.0 and self.dropout_mask is not None:
            delta = delta * self.dropout_mask / (1 - self.dropout_rate)
        
        # Compute the derivative through the activation:
        if self.training and isinstance(self.activation, Softmax):
            dz = delta
        else:
            dz = delta * self.activation.derivative(self.z)
        
        # If batch normalization was used, backpropagate through it.
        if self.use_batchnorm:
            epsilon = 1e-8
            # Retrieve cached values from forward pass.
            z_bn = self.bn_cache["z"]
            mean = self.bn_cache["mean"]
            var = self.bn_cache["var"]
            inv_std = 1 / np.sqrt(var + epsilon)
            # Compute normalized z
            z_norm = (z_bn - mean) * inv_std
            # (Optional) Compute gradients for gamma and beta:
            # dgamma = dz * z_norm
            # dbeta  = dz
            # Backprop through the scaling and shifting:
            dz_bn = dz * self.gamma
            # Let N be the number of features (fan_out)
            N = self.fan_out
            # Standard BN backward propagation formula:
            dz = (dz_bn - np.mean(dz_bn) - z_norm * np.mean(dz_bn * z_norm)) * inv_std

        # Compute gradients for weights and biases.
        dW = np.outer(h, dz)
        db = dz

        # Store the delta for further propagation.
        self.delta = np.dot(self.W, dz)

        return (dW, db)
    
    
    def __repr__(self):
        other_args = [
            "dropout_rate=" + str(self.dropout_rate) if self.dropout_rate != 0.0 else "",
            "use_batchnorm=" + str(self.use_batchnorm) if self.use_batchnorm else "",
            "bn_momentum=" + str(self.bn_momentum) if self.use_batchnorm else ""
        ]
        other_args_str = ", " + ", ".join([arg for arg in other_args if arg]) if "".join(other_args) != "" else ""
        activation_str = self.activation.__class__.__name__
        
        return f"Layer(fan_in={self.fan_in}, fan_out={self.fan_out}, activation={activation_str}{other_args_str})"
