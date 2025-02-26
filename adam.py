"""
Functional implementation of Adam optimizer.
"""

import numpy as np


def adam(
    params: np.ndarray,
    grads: np.ndarray,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    state: dict = None,
    t: int = 1,
) -> tuple[list, dict]:
    """
    Adam optimizer.

    Args:
    params (np.ndarray): List of parameter arrays to update.
    grads (np.ndarray): List of gradient arrays corresponding to the parameters.
    learning_rate (float): Learning rate for the optimizer.
    beta1 (float): Exponential decay rate for the first moment estimates.
    beta2 (float): Exponential decay rate for the second moment estimates.
    epsilon (float): Small constant to prevent division by zero.
    state (dict): Dictionary to store the state of the optimizer (m and v).
    t (int): Time step (iteration number).

    Returns:
    tuple of list and dict: Updated parameters and state of the optimizer.
        updated_params (list): List of updated parameter arrays.
        state (dict): Updated state of the optimizer.

    Example:
    ```python
    updated_params, state = adam(params, grads, learning_rate, beta1, beta2, epsilon, state, t)
    ```
    """

    if state is None:
        state = {
            "m": [np.zeros_like(p) for p in params],
            "v": [np.zeros_like(p) for p in params],
        }

    m = state["m"]
    v = state["v"]

    updated_params = []

    for i, (param, grad) in enumerate(zip(params, grads)):
        m[i] = beta1 * m[i] + (1 - beta1) * grad
        v[i] = beta2 * v[i] + (1 - beta2) * (grad ** 2)

        m_hat = m[i] / (1 - beta1**t)
        v_hat = v[i] / (1 - beta2**t)

        param_update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        updated_param = param - param_update
        updated_params.append(updated_param)

    state["m"] = m
    state["v"] = v

    return updated_params, state
