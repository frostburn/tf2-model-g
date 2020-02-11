import numpy as np


def bl_noise(shape, octaves=10):
    """Band-limited "smooth" noise"""
    grid = np.meshgrid(*[np.linspace(0, 2*np.pi, s) for s in shape])
    result = 0
    for n in range(1, octaves+1):
        for _ in range(5):
            d = 0
            u = 2*np.random.rand(len(grid))-1
            u /= np.linalg.norm(u)
            for g, c in zip(grid, u):
                d += g*c
            result += np.sin(n*d + np.random.rand()*2*np.pi) * np.random.rand() / n
    return result
