import tensorflow as tf
import numpy as np

DEFAULT_PARAMS = {
    "A": 1.0,
    "B": 2.5,
    "k4": 1.0,
    "Dx": 0.01,
    "Dy": 0.2,
    "cx": 0.5,
    "cy": 0.1,
}


class WaveBrusselerator(object):
    """
    Brusselerator with travelling wave dynamics in addition to diffusion
    """

    # We use a scheme where the concentration is broken into positive and negative "gyrating" components.
    # This so that we get stable simulation of the wave equation portion.
    def __init__(self, concentration_X, concentration_Y, dx, params=None, fixed_point_iterations=3):
        if concentration_X.shape != concentration_Y.shape:
            raise ValueError("Concentration shapes must match")
        self.dx = dx
        self.dt = dx*0.1

        self.params = params or DEFAULT_PARAMS
        self.fixed_point_iterations = fixed_point_iterations

        l = concentration_X.shape[-1]
        if any(s != l for s in concentration_X.shape):
            raise ValueError('Only square grids supported')

        ratio = 2*np.pi / (l*self.dx)
        delta = self.dt * ratio**2
        omega = l//2 - tf.abs(tf.range(l) - l//2)
        dims = len(concentration_X.shape)
        if dims == 1:
            self.fft = tf.signal.fft
            self.ifft = tf.signal.ifft

            omega2 = tf.cast(omega**2, 'complex128')
        elif dims == 2:
            self.fft = tf.signal.fft2d
            self.ifft = tf.signal.ifft2d

            omega_x, omega_y = tf.meshgrid(omega, omega)
            omega2 = tf.cast(omega_x**2 + omega_y**2, 'complex128')
        elif dims == 3:
            self.fft = tf.signal.fft3d
            self.ifft = tf.signal.ifft3d

            omega_x, omega_y, omega_z = tf.meshgrid(omega, omega, omega)
            omega2 = tf.cast(omega_x**2 + omega_y**2 + omega_z**2, 'complex128')
        else:
            raise ValueError('Only up to 3D supported')

        decay_X = tf.exp(-self.params['Dx'] * delta * omega2)
        decay_Y = tf.exp(-self.params['Dy'] * delta * omega2)
        positive_gyre_X = tf.exp(self.params['cx']* 1j * delta * tf.sqrt(omega2))
        negative_gyre_X = tf.exp(self.params['cx']* -1j * delta * tf.sqrt(omega2))
        positive_gyre_Y = tf.exp(self.params['cy']* 1j * delta * tf.sqrt(omega2))
        negative_gyre_Y = tf.exp(self.params['cy']* -1j * delta * tf.sqrt(omega2))

        self.positive_X = tf.constant(concentration_X, 'complex128') * 0.5
        self.negative_X = tf.constant(concentration_X, 'complex128') * 0.5
        self.positive_Y = tf.constant(concentration_Y, 'complex128') * 0.5
        self.negative_Y = tf.constant(concentration_Y, 'complex128') * 0.5

        def wave_diffusion_integrator(positive_X, negative_X, positive_Y, negative_Y):
            positive_X = self.fft(positive_X)
            negative_X = self.fft(negative_X)
            positive_Y = self.fft(positive_Y)
            negative_Y = self.fft(negative_Y)

            positive_X *= positive_gyre_X
            positive_X *= decay_X
            negative_X *= negative_gyre_X
            negative_X *= decay_X
            positive_Y *= positive_gyre_Y
            positive_Y *= decay_Y
            negative_Y *= negative_gyre_Y
            negative_Y *= decay_Y
            return self.ifft(positive_X), self.ifft(negative_X), self.ifft(positive_Y), self.ifft(negative_Y)

        # This is another way to partition the interactions between the positive and negative gyrations, but it's not working for some reason.
        # def reaction_integrator(positive_X, negative_X, positive_Y, negative_Y):
        #     new_pos_X = positive_X
        #     new_neg_X = negative_X
        #     new_pos_Y = positive_Y
        #     new_neg_Y = negative_Y
        #     for _ in range(self.fixed_point_iterations):
        #         X = new_pos_X + new_neg_X
        #         Y = new_pos_Y + new_neg_Y

        #         v_pos_X = self.params['A'] - (self.params['k4'] + self.params['B']) * new_pos_X + (new_pos_X**2 + new_pos_X*new_neg_X) * Y
        #         v_neg_X = self.params['A'] - (self.params['k4'] + self.params['B']) * new_neg_X + (new_neg_X**2 + new_pos_X*new_neg_X) * Y

        #         v_pos_Y = -X**2 * new_pos_Y + self.params['B'] * X
        #         v_neg_Y = -X**2 * new_neg_Y + self.params['B'] * X

        #         new_pos_X = positive_X + self.dt*v_pos_X
        #         new_neg_X = negative_X + self.dt*v_neg_X
        #         new_pos_Y = positive_Y + self.dt*v_pos_Y
        #         new_neg_Y = negative_Y + self.dt*v_neg_Y
        #     positive_X = new_pos_X
        #     negative_X = new_neg_X
        #     positive_Y = new_pos_Y
        #     negative_Y = new_neg_Y
        #     return positive_X, negative_X, positive_Y, negative_Y

        def reaction_integrator(positive_X, negative_X, positive_Y, negative_Y):
            new_pos_X = positive_X
            new_neg_X = negative_X
            new_pos_Y = positive_Y
            new_neg_Y = negative_Y
            for _ in range(self.fixed_point_iterations):
                new_X = new_pos_X + new_neg_X
                new_Y = new_pos_Y + new_neg_Y
                xy_flow = new_X*new_X*new_Y - self.params["B"]*new_X
                v_X = self.params["A"] + xy_flow - self.params["k4"] * new_X
                v_Y = -xy_flow
                new_pos_X = positive_X + 0.5*self.dt*v_X
                new_neg_X = negative_X + 0.5*self.dt*v_X
                new_pos_Y = positive_Y + 0.5*self.dt*v_Y
                new_neg_Y = negative_Y + 0.5*self.dt*v_Y
            positive_X = new_pos_X
            negative_X = new_neg_X
            positive_Y = new_pos_Y
            negative_Y = new_neg_Y
            return positive_X, negative_X, positive_Y, negative_Y

        self.wave_diffusion_integrator = tf.function(wave_diffusion_integrator)
        self.reaction_integrator = tf.function(reaction_integrator)

    def step(self):
        values = self.wave_diffusion_integrator(self.positive_X, self.negative_X, self.positive_Y, self.negative_Y)
        values = self.reaction_integrator(*values)
        self.positive_X, self.negative_X, self.positive_Y, self.negative_Y = values

    def numpy(self):
        X = self.positive_X + self.negative_X
        Y = self.positive_Y + self.negative_Y
        return np.real(X.numpy()), np.real(Y.numpy())


if __name__ == '__main__':
    from util import bl_noise
    import pylab
    from matplotlib.animation import FuncAnimation

    x = np.linspace(-5, 5, 128)
    dx = x[1] - x[0]
    x, y, z = np.meshgrid(x, x, x)

    r2 = x*x+y*y+z*z
    brusselerator = WaveBrusselerator(1.5 + np.exp(-0.2*r2)*bl_noise(x.shape), 1.5 + np.exp(-0.3*r2), dx)
    X, Y = brusselerator.numpy()
    plots = []
    plots.extend(pylab.plot(z[0,0], X[64,64]))
    plots.extend(pylab.plot(z[0,0], Y[64,64]))
    pylab.ylim(0, 4)

    def update(frame):
        brusselerator.step()
        X, Y = brusselerator.numpy()
        plots[0].set_ydata(X[64, 64])
        plots[-1].set_ydata(Y[64, 64])
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(1000), init_func=lambda: plots, blit=True, repeat=True, interval=10)
    pylab.show()

    X, Y = brusselerator.numpy()
    pylab.imshow(X[64])
    pylab.show()
