import tensorflow as tf
import numpy as np
from pde_solver import PDESolver
from integrators.model_g import polynomial_order_4_centered as reaction_integrator
from integrators.model_g import steady_state

DEFAULT_PARAMS = {
    "A": 3.42,
    "B": 13.5,
    "k2": 1.0,
    "k-2": 0.1,
    "k5": 0.9,
    "D_G": 1.0,
    "D_X": 1.0,
    "D_Y": 2.0,
}


class ModelG(PDESolver):
    """
    Model G Reaction Diffusion system
    """

    def __init__(self, concentration_G, concentration_X, concentration_Y, dx, dt=None, params=None, source_functions=None):
        if dt is None:
            dt = 0.1 * dx

        if dt > 0.5 * dx:
            warnings.warn("Time increment {} too large for simulation stability with grid constant {}".format(dt, dx))

        super().__init__(dx, dt, concentration_G.shape)

        if concentration_X.shape != concentration_Y.shape or concentration_X.shape != concentration_G.shape:
            raise ValueError("Concentration shapes must match")

        self.params = params or DEFAULT_PARAMS
        self.source_functions = source_functions or {}

        self.concentration_G = tf.constant(concentration_G, dtype="float64")
        self.concentration_X = tf.constant(concentration_X, dtype="float64")
        self.concentration_Y = tf.constant(concentration_Y, dtype="float64")

        if self.dims == 1:
            omega2 = self.omega_x**2
        elif self.dims == 2:
            omega_x, omega_y = self.omega_x, self.omega_y
            omega2 = omega_x**2 + omega_y**2
        elif self.dims == 3:
            omega_x, omega_y, omega_z = self.omega_x, self.omega_y, self.omega_z
            omega2 = omega_x**2 + omega_y**2 + omega_z**2
        else:
            raise ValueError('Only up to 3D supported')

        delta = tf.cast(-omega2 * self.dt, 'complex128')
        decay_G = tf.exp(self.params['D_G'] * delta)
        decay_X = tf.exp(self.params['D_X'] * delta)
        decay_Y = tf.exp(self.params['D_Y'] * delta)
        def diffusion_integrator(*cons):
            result = []
            for decay, concentration in zip([decay_G, decay_X, decay_Y], cons):
                f = self.fft(tf.cast(concentration, 'complex128'))
                f *= decay
                result.append(tf.cast(tf.math.real(self.ifft(f)), 'float64'))
            return result

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
