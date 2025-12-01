from typing import Callable

from numpy import ndarray


class Problem():
    """Classe per rappresentare un problema di ottimizzazione."""
    def __init__(self, function:Callable, n_var:int, lb:list[float]|ndarray, ub:list[float]|ndarray):
        if len(lb) != n_var or len(ub) != n_var:
            raise ValueError("Lb and Ub length must be equal to n_var")
        self.function = function
        self.n_var = n_var
        self.lb = lb
        self.ub = ub