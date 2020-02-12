import numpy as np
import tensorflow as tf


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


@tf.function
def laplacian1d(y, dx):
    f = tf.signal.fft(tf.cast(y, 'complex128'))
    l = f.shape[-1]
    ratio = (2*np.pi/(dx*l))**2
    f *= tf.cast((l//2 - tf.abs(tf.range(l) - l//2))**2, 'complex128')
    return tf.cast(tf.math.real(tf.signal.ifft(f)), 'float64') * ratio


if __name__ == '__main__':
    import pylab
    x = np.linspace(-4, 3, 100)
    dx = x[1] - x[0]
    x = tf.constant(x)
    y = tf.exp(-x*x)
    nabla2_y = (2 - 4*x*x) * tf.exp(-x*x)
    pylab.plot(x.numpy(), laplacian1d(y, dx=dx).numpy())
    pylab.plot(x.numpy(), nabla2_y)
    pylab.show()
