import tensorflow as tf
import numpy as np
import util


class HeatEquation(object):
    """
    Simple integrator to test spectral methods
    """
    def __init__(self, u, dx):
        self.dx = dx
        self.dt = 0.1 * dx
        self.u = tf.constant(u)

        def integrator(u):
            f = tf.signal.fft(tf.cast(u, 'complex128'))
            l = f.shape[-1]
            ratio = (2*np.pi/(self.dx*l))**2
            delta = self.dt * ratio
            f *= tf.cast(tf.exp(-delta * tf.cast((l//2 - tf.abs(tf.range(l) - l//2))**2, 'float64')), 'complex128')
            return tf.cast(tf.math.real(tf.signal.ifft(f)), 'float64')

        self.integrator = tf.function(integrator)

    def step(self):
        self.u = self.integrator(self.u)

    def numpy(self):
        return self.u.numpy()


if __name__ == '__main__':
    from util import bl_noise
    import pylab
    from matplotlib.animation import FuncAnimation

    x = np.linspace(-5, 5, 256)
    dx = x[1] - x[0]
    u = np.exp(-x*x) * bl_noise(x.shape)

    heat_equation = HeatEquation(u, dx)

    u = heat_equation.numpy()
    plots = pylab.plot(x, u)

    pylab.ylim(-1.0, 1.0)

    def update(frame):
        heat_equation.step()
        v = heat_equation.numpy()
        plots[0].set_ydata(v)
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()
