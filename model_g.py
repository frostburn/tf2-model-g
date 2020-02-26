import tensorflow as tf
import numpy as np
from integrators.model_g import polynomial_order_4_centered as reaction_integrator

DEFAULT_PARAMS = {
    "A": 3.42,
    "B": 13.5,
    "k2": 1.0,
    "k-2": 0.1,
    "k5": 0.9,
    "Dx": 1.0,
    "Dy": 2.0,
}


class ModelG(object):
    """
    Model G Reaction Diffusion system
    """

    def __init__(self, concentration_G, concentration_X, concentration_Y, dx, params=None, fixed_point_iterations=3, source_functions=None):
        if concentration_X.shape != concentration_Y.shape or concentration_X.shape != concentration_G.shape:
            raise ValueError("Concentration shapes must match")
        self.dx = dx
        self.dt = 0.1 * dx
        self.t = 0

        self.concentration_G = tf.constant(concentration_G, dtype="float64")
        self.concentration_X = tf.constant(concentration_X, dtype="float64")
        self.concentration_Y = tf.constant(concentration_Y, dtype="float64")
        self.params = params or DEFAULT_PARAMS
        self.fixed_point_iterations = fixed_point_iterations
        self.source_functions = source_functions or {}

        l = concentration_X.shape[-1]
        if any(s != l for s in concentration_X.shape):
            raise ValueError('Only square grids supported')

        ratio = 2*np.pi / (l*self.dx)
        delta = self.dt * ratio**2
        omega2 = (l//2 - tf.abs(tf.range(l) - l//2))**2
        dims = len(concentration_X.shape)
        if dims == 1:
            gamma = -delta * tf.cast(omega2, 'complex128')
            decay_G = tf.exp(gamma)
            decay_X = tf.exp(self.params['Dx'] * gamma)
            decay_Y = tf.exp(self.params['Dy'] * gamma)
            def diffusion_integrator(*cons):
                result = []
                for decay, concentration in zip([decay_G, decay_X, decay_Y], cons):
                    f = tf.signal.fft(tf.cast(concentration, 'complex128'))
                    f *= decay
                    result.append(tf.cast(tf.math.real(tf.signal.ifft(f)), 'float64'))
                return result
        elif dims == 2:
            omega2_x, omega2_y = tf.meshgrid(omega2, omega2)
            gamma = -delta * tf.cast(omega2_x + omega2_y, 'complex128')
            decay_G = tf.exp(gamma)
            decay_X = tf.exp(self.params['Dx'] * gamma)
            decay_Y = tf.exp(self.params['Dy'] * gamma)
            def diffusion_integrator(*cons):
                result = []
                for decay, concentration in zip([decay_G, decay_X, decay_Y], cons):
                    f = tf.signal.fft2d(tf.cast(concentration, 'complex128'))
                    f *= decay
                    result.append(tf.cast(tf.math.real(tf.signal.ifft2d(f)), 'float64'))
                return result
        elif dims == 3:
            omega2_x, omega2_y, omega2_z = tf.meshgrid(omega2, omega2, omega2)
            gamma = -delta * tf.cast(omega2_x + omega2_y + omega2_z, 'complex128')
            decay_G = tf.exp(gamma)
            decay_X = tf.exp(self.params['Dx'] * gamma)
            decay_Y = tf.exp(self.params['Dy'] * gamma)
            def diffusion_integrator(*cons):
                result = []
                for decay, concentration in zip([decay_G, decay_X, decay_Y], cons):
                    f = tf.signal.fft3d(tf.cast(concentration, 'complex128'))
                    f *= decay
                    result.append(tf.cast(tf.math.real(tf.signal.ifft3d(f)), 'float64'))
                return result
        else:
            raise ValueError('Only up to 3D supported')

        reaction_integrator_curried = lambda con_G, con_X, con_Y: reaction_integrator(
            con_G, con_X, con_Y,
            self.dt, self.params['A'], self.params['B'], self.params['k2'], self.params['k-2'], self.params['k5']
        )

        self.diffusion_integrator = tf.function(diffusion_integrator)
        self.reaction_integrator = tf.function(reaction_integrator_curried)

    def step(self):
        values = self.diffusion_integrator(self.concentration_G, self.concentration_X, self.concentration_Y)
        self.concentration_G, self.concentration_X, self.concentration_Y = values
        values = self.reaction_integrator(self.concentration_G, self.concentration_X, self.concentration_Y)
        self.concentration_G, self.concentration_X, self.concentration_Y = values
        zero = lambda t: 0
        source_G = self.source_functions.get('G', zero)(self.t)
        source_X = self.source_functions.get('X', zero)(self.t)
        source_Y = self.source_functions.get('Y', zero)(self.t)
        self.concentration_G += self.dt * source_G
        self.concentration_X += self.dt * source_X
        self.concentration_Y += self.dt * source_Y
        self.t += self.dt

    def numpy(self):
        return (
            self.concentration_G.numpy(),
            self.concentration_X.numpy(),
            self.concentration_Y.numpy()
        )
