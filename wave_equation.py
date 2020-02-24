import tensorflow as tf
import numpy as np
import util


class WaveEquation(object):
    """
    Simple integrator to test spectral methods
    """
    def __init__(self, u, dx, zeta=0):
        self.dx = dx
        self.dt = 0.5 * dx

        l = u.shape[-1]
        if any(s != l for s in u.shape):
            raise ValueError('Only square grids supported')

        ratio = 2*np.pi / (l*self.dx)
        delta = self.dt * ratio**2
        omega = l//2 - tf.abs(tf.range(l) - l//2)
        dims = len(u.shape)
        if dims == 1:
            self.fft = tf.signal.fft
            self.ifft = tf.signal.ifft

            decay = tf.exp(-zeta * delta * tf.cast(omega**2, 'complex128'))
            positive_gyre = tf.exp(1j * delta * tf.cast(omega, 'complex128'))
            negative_gyre = tf.exp(-1j * delta * tf.cast(omega, 'complex128'))
        elif dims == 2:
            self.fft = tf.signal.fft2d
            self.ifft = tf.signal.ifft2d

            omega_x, omega_y = tf.meshgrid(omega, omega)
            omega2 = tf.cast(omega_x**2 + omega_y**2, 'complex128')
            decay = tf.exp(-zeta * delta * omega2)
            positive_gyre = tf.exp(1j * delta * tf.sqrt(omega2))
            negative_gyre = tf.exp(-1j * delta * tf.sqrt(omega2))
        elif dims == 3:
            self.fft = tf.signal.fft3d
            self.ifft = tf.signal.ifft3d

            omega_x, omega_y, omega_z = tf.meshgrid(omega, omega, omega)
            omega2 = tf.cast(omega_x**2 + omega_y**2 + omega_z**2, 'complex128')
            decay = tf.exp(-zeta * delta * omega2)
            positive_gyre = tf.exp(1j * delta * tf.sqrt(omega2))
            negative_gyre = tf.exp(-1j * delta * tf.sqrt(omega2))
        else:
            raise ValueError('Only up to 3D supported')

        self.positive_v = self.fft(tf.constant(u, 'complex128')) * 0.5
        self.negative_v = self.fft(tf.constant(u, 'complex128')) * 0.5

        def integrator(positive_v, negative_v):
            positive_v *= positive_gyre
            positive_v *= decay
            negative_v *= negative_gyre
            negative_v *= decay
            return positive_v, negative_v

        self.integrator = tf.function(integrator)

    def step(self):
        self.positive_v, self.negative_v = self.integrator(self.positive_v, self.negative_v)

    def numpy(self):
        u = self.ifft(self.positive_v) + self.ifft(self.negative_v)
        return np.real(u.numpy())


if __name__ == '__main__':
    from util import bl_noise
    import pylab
    from matplotlib.animation import FuncAnimation

    x = np.linspace(-4, 4, 128)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)
    # u = np.exp(-x*x) # * bl_noise(x.shape)
    u = np.exp(-x*x - y*y)

    wave_equation = WaveEquation(u, dx, zeta=0.01)

    u = wave_equation.numpy()
    plots = [pylab.imshow(u, vmin=-0.25, vmax=0.25, cmap='cividis')]
    # plots = pylab.plot(x, u)
    # pylab.ylim(-1.0, 1.0)

    def update(frame):
        wave_equation.step()
        u = wave_equation.numpy()
        plots[0].set_data(u)
        # plots[0].set_ydata(u)
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=10)
    pylab.show()
