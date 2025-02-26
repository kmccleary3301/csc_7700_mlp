import numpy as np

class ActivationFunction:
	def __init__(self):
		pass

	def forward(self, x: np.ndarray) -> np.ndarray:
		raise NotImplementedError("Subclasses must implement this method")

	def derivative(self, x: np.ndarray) -> np.ndarray:
		raise NotImplementedError("Subclasses must implement this method")

class SoftPlus(ActivationFunction):
	def forward(self, x: np.ndarray, beta: float = 1) -> np.ndarray:
		"""
		Original func is f(x) = (1/b)log(1 + exp(bx))
		"""
		v = np.log(1 + np.exp(x if beta == 1 else x * beta))
		return v if beta == 1 else v / beta

	def derivative(self, x: np.ndarray, beta: float = 1) -> np.ndarray:
		"""
		Original func is f(x) = (1/b)log(1 + exp(bx))
		Derivative of f(x) = (1/b) * (1 / (1 + exp(bx))) * exp(bx) * b
			= (1 / (1 + exp(-bx)))
		"""

		return 1 / (1 + np.exp(-x if beta == 1 else -x * beta))

class Tanh(ActivationFunction):
	def forward(self, x: np.ndarray) -> np.ndarray:
		v = np.exp(2*x)
		return (v - 1) / (v + 1)

	def derivative(self, x: np.ndarray) -> np.ndarray:
		v_1 = np.exp(x)
		v_2 = 1/v_1
		cosh_val = (v_1 + v_2) / 2
		
		return 1 / (cosh_val**2)

class Mish(ActivationFunction):
	def __init__(self):
		self.tanh = Tanh()
		self.softplus = SoftPlus()
  
	def forward(self, x: np.ndarray) -> np.ndarray:
		return x * self.tanh.forward(self.softplus.forward(x))

	def derivative(self, x: np.ndarray) -> np.ndarray:
		s = self.softplus.forward(x)
		s_d = self.softplus.derivative(x)
		return self.tanh.forward(s) + x * s_d * self.tanh.derivative(s)


class Sigmoid(ActivationFunction):
	def forward(self, x: np.ndarray) -> np.ndarray:
		return 1 / (1 + np.exp(-x))

	def derivative(self, x: np.ndarray, is_output: bool = False) -> np.ndarray:
		v = self.forward(x) if not is_output else x
		return v * (1 - v)


class ReLU(ActivationFunction):
	def forward(self, x: np.ndarray) -> np.ndarray:
		return np.maximum(0, x)

	def derivative(self, x: np.ndarray) -> np.ndarray: # No need to check if it's an output
		return (x > 0).astype(float)

class Softmax(ActivationFunction):
	def forward(self, x: np.ndarray) -> np.ndarray:
		exps = np.exp(x - np.max(x))
		return exps / np.sum(exps, axis=0)
    
	def derivative(self, x: np.ndarray, is_output: bool = False) -> np.ndarray:
		"""
		Given either an input vector x, or an output vector y, 
		compute the derivative of the softmax function.
		"""
		softmax_vector = self.forward(x) if not is_output else x
		
		return np.diag(softmax_vector) - np.outer(softmax_vector, softmax_vector)


class Linear(ActivationFunction):
	def forward(self, x: np.ndarray) -> np.ndarray:
		return x

	def derivative(self, x: np.ndarray) -> np.ndarray:
		return np.ones_like(x)