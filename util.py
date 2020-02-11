import numpy as np


def bl_noise(shape, frequencies=None, weights=None):
    """Band-limited "smooth" noise"""
    grid = np.meshgrid(*[np.linspace(0, 2*np.pi, s) for s in shape])
    result = 0
    frequencies = frequencies or range(1, 11)
    weights = weights or [1.0 / f for f in frequencies]
    for f, w in zip(frequencies, weights):
        for _ in range(5):
            d = 0
            u = 2*np.random.rand(len(grid))-1
            u /= np.linalg.norm(u)
            for g, c in zip(grid, u):
                d += g*c
            result += np.sin(f*d + np.random.rand()*2*np.pi) * np.random.rand() * w
    return result
