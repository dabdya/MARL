import numpy as np


def generate_communication_matrix(n: int):
    m = np.random.randint(0, high=2, size=(n, n))
    np.fill_diagonal(m, 1)
    return m
