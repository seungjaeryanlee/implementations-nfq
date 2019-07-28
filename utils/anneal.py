"""Useful miscellaneous functions."""
from typing import Callable


def get_linear_anneal_func(
    start_value: float, end_value: float, end_steps: int
) -> Callable:
    """Create a linear annealing function.

    Parameters
    ----------
    start_value : float
        Initial value for linear annealing.
    end_value : float
        Terminal value for linear annealing.
    end_steps : int
        Number of steps to anneal value.

    Returns
    -------
    linear_anneal_func : Callable
        A function that returns annealed value given a step index.

    """

    def linear_anneal_func(x):
        assert x >= 0
        return (end_value - start_value) * min(x, end_steps) / end_steps + start_value

    return linear_anneal_func
