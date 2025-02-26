import numpy as np


class LossFunction():
	def __init__(self):
		pass
	
	def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
		raise NotImplementedError("Subclasses must implement this method")

	def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
		"""
		Taken with respect to y_pred
  		"""
		raise NotImplementedError("Subclasses must implement this method")


class SquaredError(LossFunction):
	def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
		return 0.5 * np.sum((y_true - y_pred)**2)

	def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
		return y_pred - y_true

class CrossEntropy(LossFunction):
    """
    On the issue of NaNs:
    The issue is likely numerical instability. In the CrossEntropy implementation:
	
	* 	You're taking the logarithm of y_pred without any safeguard. If any value in y_pred is 0 
 		(or very close to 0), np.log(y_pred) will return -inf, and eventually computations produce NaN.
	
	* 	Similarly, the derivative uses division by y_pred. If any element of y_pred is 0, 
 		it results in a division by zero, leading to NaNs.
	
 	You can fix this by clipping y_pred using a small epsilon value before computing the logarithm or division.
    """
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return - y_true / y_pred
