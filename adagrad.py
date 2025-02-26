"""
Functional implementation of Adagrad optimizer.
"""

import numpy as np

def adagrad(
        params: np.ndarray,
        grads: np.ndarray,
        learning_rate: float = 0.01,
        epsilon: float = 1e-8,
        state: dict = None
    ) -> tuple[list, dict]:
    """
    Adagrad optimizer.
    
    Args:
    params (np.ndarray): List of parameter arrays to update.
    grads (np.ndarray): List of gradient arrays corresponding to the parameters.
    learning_rate (float): Learning rate for the optimizer.
    epsilon (float): Small constant to prevent division by zero.
    state (dict): Dictionary to store the state of the optimizer (cache).
    
    Returns:
    tuple of list and dict: Updated parameters and state of the optimizer.
        updated_params (np.ndarray): List of updated parameter arrays.
        state (dict): Updated state of the optimizer.
    
    Example:
    ```python
    updated_params, state = adagrad(params, grads, learning_rate, epsilon, state)
    ```
    """

    if state is None:
        state = {'cache': [np.zeros_like(p) for p in params]}

    cache = state['cache']
    updated_params = []

    for i, (param, grad) in enumerate(zip(params, grads)):
        cache[i] += grad ** 2
        param_update = learning_rate * grad / (np.sqrt(cache[i]) + epsilon)
        updated_param = param - param_update
        updated_params.append(updated_param)

    state['cache'] = cache

    return updated_params, state
