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

        l = u.shape[-1]
        ratio = 2*np.pi / (l*self.dx)
        delta = self.dt * ratio**2
        omega2 = (l//2 - tf.abs(tf.range(l) - l//2))**2
        dims = len(u.shape)
        if dims == 1:
            decay = tf.exp(-delta * tf.cast(omega2, 'complex128'))
            def integrator(u):
                f = tf.signal.fft(tf.cast(u, 'complex128'))
                f *= decay
                return tf.cast(tf.math.real(tf.signal.ifft(f)), 'float64')
        elif dims == 2:
            omega2_x, omega2_y = tf.meshgrid(omega2, omega2)
            decay = tf.exp(-delta * tf.cast(omega2_x + omega2_y, 'complex128'))
            def integrator(u):
                f = tf.signal.fft2d(tf.cast(u, 'complex128'))
                f *= decay
                return tf.cast(tf.math.real(tf.signal.ifft2d(f)), 'float64')
        else:
            raise ValueError('Only up to 2D supported')

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
    x, y = np.meshgrid(x, x)
    # u = np.exp(-x*x) * bl_noise(x.shape)
    u = np.exp(-x*x -y*y) * bl_noise(x.shape)

    heat_equation = HeatEquation(u, dx)

    u = heat_equation.numpy()
    plots = [pylab.imshow(u)]
    # plots = pylab.plot(x, u)
    # pylab.ylim(-1.0, 1.0)

    def update(frame):
        heat_equation.step()
        u = heat_equation.numpy()
        plots[0].set_data(u)
        # plots[0].set_ydata(u)
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()
