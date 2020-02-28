import numpy as np
import tensorflow as tf


def bl_noise_generic(shape, frequencies=None, weights=None):
    """Band-limited "smooth" noise"""
    frequencies = list(frequencies or range(1, 11))
    weights = weights or [1.0 / f for f in frequencies]
    if True:
        grid = np.meshgrid(*[np.linspace(0, 2*np.pi, s) for s in shape])
        result = 0
        for f, w in zip(frequencies, weights):
            for _ in range(5):
                d = 0
                u = 2*np.random.rand(len(grid))-1
                u /= np.linalg.norm(u)
                for g, c in zip(grid, u):
                    d += g*c
                result += np.sin(f*d + np.random.rand()*2*np.pi) * np.random.rand() * w
        return result


def bl_noise(shape, limit=11, weight=None):
    weight = weight or (lambda f: 1.0 / (f + 1) - (f == 0))

    components = np.random.randn(*shape) + np.random.randn(*shape) * 1j
    dims = len(shape)
    if dims == 1:
        l = shape[0]
        freqs = l//2 - abs(np.arange(l) - l//2)
        components *= weight(freqs) * (freqs <= limit)
    elif dims == 2:
        l = shape[0]
        x = l//2 - abs(np.arange(l) - l//2)
        l = shape[1]
        y = l//2 - abs(np.arange(l) - l//2)
        x, y = np.meshgrid(x, y)
        freqs = np.sqrt(x*x + y*y)
    elif dims == 3:
        l = shape[0]
        x = l//2 - abs(np.arange(l) - l//2)
        l = shape[1]
        y = l//2 - abs(np.arange(l) - l//2)
        l = shape[2]
        z = l//2 - abs(np.arange(l) - l//2)
        x, y, z = np.meshgrid(x, y, z)
        freqs = np.sqrt(x*x + y*y + z*z)
    if dims <= 3:
        components *= weight(freqs) * (freqs <= limit)
        return np.real(np.fft.fftn(components, shape))
    else:
        return bl_noise_generic(shape)  # TODO: Pass in the freqs and weights

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
    x, y = np.meshgrid(x, x)

    pylab.imshow(bl_noise(x.shape))
    pylab.show()


def l2_location(field, *axis):
    weight = field**2
    # TODO: Normalize
    return np.array([(a * weight).sum() for a in axis])
