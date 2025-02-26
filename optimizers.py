from typing import List, Tuple
import numpy as np

class Optimizer():
	def update(
     	self,
       	dW: List[np.ndarray], 
        db: List[np.ndarray],
        params_W: List[np.ndarray],
        params_b: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
		"""
		Taking in the latest computed gradient, return the new differences to make.
  		"""
		raise NotImplementedError("Subclasses must implement this method")


class RMSProp(Optimizer):
    def __init__(
        self,
        learning_rate: float,
        decay: float = 0.9,
        epsilon: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.cache_W = None
        self.cache_b = None
  
    def update(
		self,
		dW: List[np.ndarray], 
        db: List[np.ndarray],
        **kwargs
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute RMSProp update differences for weights and biases.
        """
        # Initialize caches if they are None
        if self.cache_W is None:
            self.cache_W = [np.zeros_like(dw) for dw in dW]
        if self.cache_b is None:
            self.cache_b = [np.zeros_like(db_elem) for db_elem in db]
        
        updates_W = []
        updates_b = []
        
        # Process weight gradients
        for i, grad in enumerate(dW):
            # Update cache for weights
            self.cache_W[i] = self.decay * self.cache_W[i] + (1 - self.decay) * np.square(grad)
            # Compute adaptive learning rate for weights
            adaptive_lr = self.learning_rate / (np.sqrt(self.cache_W[i]) + self.epsilon)
            # Compute the update difference (delta)
            update = - adaptive_lr * grad
            updates_W.append(update)
            
        # Process bias gradients
        for i, grad in enumerate(db):
            # Update cache for biases
            self.cache_b[i] = self.decay * self.cache_b[i] + (1 - self.decay) * np.square(grad)
            # Compute adaptive learning rate for biases
            adaptive_lr = self.learning_rate / (np.sqrt(self.cache_b[i]) + self.epsilon)
            # Compute the update difference (delta)
            update = - adaptive_lr * grad
            updates_b.append(update)
            
        return updates_W, updates_b
    
class AdamW2017(Optimizer):
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m_W = None  # First-moment estimate for weights
        self.v_W = None  # Second-moment estimate for weights
        self.m_b = None  # First-moment estimate for biases
        self.v_b = None  # Second-moment estimate for biases
        
        self.t = 0  # time step

    def update(
        self,
        dW: List[np.ndarray],
        db: List[np.ndarray],
        params_W: List[np.ndarray],
        params_b: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute AdamW update differences for weights and biases.

        Args:
            dW: List of gradients for the weights.
            db: List of gradients for the biases.
            params_W: List of current weight arrays.
            params_b: List of current bias arrays.

        Returns:
            A tuple containing:
              - List of update differences for weights.
              - List of update differences for biases.
        """
        # Initialize moment estimates if needed
        if self.m_W is None:
            self.m_W = [np.zeros_like(dw) for dw in dW]
        if self.v_W is None:
            self.v_W = [np.zeros_like(dw) for dw in dW]
        if self.m_b is None:
            self.m_b = [np.zeros_like(db_elem) for db_elem in db]
        if self.v_b is None:
            self.v_b = [np.zeros_like(db_elem) for db_elem in db]
        
        self.t += 1
        updates_W = []
        updates_b = []
        
        # Update weights
        for i, grad in enumerate(dW):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * grad
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * np.square(grad)
            m_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
            
            # Compute the AdamW update difference:
            # delta = -[lr * (m_hat / (sqrt(v_hat)+epsilon) + weight_decay * param)]
            delta = - (self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                       + self.learning_rate * self.weight_decay * params_W[i])
            updates_W.append(delta)
        
        # Update biases
        for i, grad in enumerate(db):
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * np.square(grad)
            m_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
            
            delta = - (self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                       + self.learning_rate * self.weight_decay * params_b[i])
            updates_b.append(delta)
        
        return updates_W, updates_b