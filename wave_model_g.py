import tensorflow as tf
import numpy as np
from integrators.model_g import polynomial_order_4_centered as reaction_integrator


DEFAULT_PARAMS = {
    "A": 3.42,
    "B": 13.5,
    "k2": 1.0,
    "k-2": 0.1,
    "k5": 0.9,
    "Dg": 1.0,
    "Dx": 1.0,
    "Dy": 2.0,
    "cg": 0.1,
    "cx": 0.1,
    "cy": 0.1,
}


class WaveModelG(object):
    """
    Model G Reaction system with travelling wave dynamics in addition to diffusion
    """

    # We use a scheme where the concentration is broken into positive and negative "gyrating" components.
    # This so that we get stable simulation of the wave equation portion.
    def __init__(self, concentration_G, concentration_X, concentration_Y, dx, params=None, source_functions=None):
        if concentration_X.shape != concentration_Y.shape or concentration_X.shape != concentration_G.shape:
            raise ValueError("Concentration shapes must match")
        self.dx = dx
        self.dt = dx*0.1
        self.t = 0

        self.params = params or DEFAULT_PARAMS
        self.source_functions = source_functions or {}

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

        decay_G = tf.exp(-self.params['Dg'] * delta * omega2)
        decay_X = tf.exp(-self.params['Dx'] * delta * omega2)
        decay_Y = tf.exp(-self.params['Dy'] * delta * omega2)
        positive_gyre_G = tf.exp(self.params['cg']* +1j * delta * tf.sqrt(omega2)) * decay_G
        negative_gyre_G = tf.exp(self.params['cg']* -1j * delta * tf.sqrt(omega2)) * decay_G
        positive_gyre_X = tf.exp(self.params['cx']* +1j * delta * tf.sqrt(omega2)) * decay_X
        negative_gyre_X = tf.exp(self.params['cx']* -1j * delta * tf.sqrt(omega2)) * decay_X
        positive_gyre_Y = tf.exp(self.params['cy']* +1j * delta * tf.sqrt(omega2)) * decay_Y
        negative_gyre_Y = tf.exp(self.params['cy']* -1j * delta * tf.sqrt(omega2)) * decay_Y

        self.positive_G = tf.constant(concentration_G, 'complex128') * 0.5
        self.negative_G = tf.constant(concentration_G, 'complex128') * 0.5
        self.positive_X = tf.constant(concentration_X, 'complex128') * 0.5
        self.negative_X = tf.constant(concentration_X, 'complex128') * 0.5
        self.positive_Y = tf.constant(concentration_Y, 'complex128') * 0.5
        self.negative_Y = tf.constant(concentration_Y, 'complex128') * 0.5

        def wave_diffusion_integrator(positive_G, negative_G, positive_X, negative_X, positive_Y, negative_Y):
            positive_G = self.fft(positive_G)
            negative_G = self.fft(negative_G)
            positive_X = self.fft(positive_X)
            negative_X = self.fft(negative_X)
            positive_Y = self.fft(positive_Y)
            negative_Y = self.fft(negative_Y)

            positive_G *= positive_gyre_G
            negative_G *= negative_gyre_G
            positive_X *= positive_gyre_X
            negative_X *= negative_gyre_X
            positive_Y *= positive_gyre_Y
            negative_Y *= negative_gyre_Y
            return self.ifft(positive_G), self.ifft(negative_G), self.ifft(positive_X), self.ifft(negative_X), self.ifft(positive_Y), self.ifft(negative_Y)

        # def reaction_integrator(positive_G, negative_G, positive_X, negative_X, positive_Y, negative_Y):
        #     new_pos_G = positive_G
        #     new_neg_G = negative_G
        #     new_pos_X = positive_X
        #     new_neg_X = negative_X
        #     new_pos_Y = positive_Y
        #     new_neg_Y = negative_Y
        #     for _ in range(self.fixed_point_iterations):
        #         new_G = new_pos_G + new_neg_G
        #         new_X = new_pos_X + new_neg_X
        #         new_Y = new_pos_Y + new_neg_Y

        #         gx_flow = self.params["k-2"]*new_X - self.params["k2"]*new_G
        #         xy_flow = new_X*new_X*new_Y - self.params["B"]*new_X
        #         v_G = self.params["A"] + gx_flow
        #         v_X = xy_flow - gx_flow - self.params["k5"] * new_X
        #         v_Y = -xy_flow

        #         new_pos_G = positive_G + 0.5*self.dt*v_G
        #         new_neg_G = negative_G + 0.5*self.dt*v_G
        #         new_pos_X = positive_X + 0.5*self.dt*v_X
        #         new_neg_X = negative_X + 0.5*self.dt*v_X
        #         new_pos_Y = positive_Y + 0.5*self.dt*v_Y
        #         new_neg_Y = negative_Y + 0.5*self.dt*v_Y
        #     return new_pos_G, new_neg_G, new_pos_X, new_neg_X, new_pos_Y, new_neg_Y

        def reaction_integrator_curried(positive_G, negative_G, positive_X, negative_X, positive_Y, negative_Y):
            con_G = positive_G + negative_G
            con_X = positive_X + negative_X
            con_Y = positive_Y + negative_Y
            new_G, new_X, new_Y = reaction_integrator(
                con_G, con_X, con_Y,
                self.dt, self.params['A'], self.params['B'], self.params['k2'], self.params['k-2'], self.params['k5']
            )
            positive_G += 0.5*(new_G - con_G)
            negative_G += 0.5*(new_G - con_G)
            positive_X += 0.5*(new_X - con_X)
            negative_X += 0.5*(new_X - con_X)
            positive_Y += 0.5*(new_Y - con_Y)
            negative_Y += 0.5*(new_Y - con_Y)
            return positive_G, negative_G, positive_X, negative_X, positive_Y, negative_Y

        self.wave_diffusion_integrator = tf.function(wave_diffusion_integrator)
        self.reaction_integrator = tf.function(reaction_integrator_curried)

    def step(self):
        values = self.wave_diffusion_integrator(self.positive_G, self.negative_G, self.positive_X, self.negative_X, self.positive_Y, self.negative_Y)
        values = self.reaction_integrator(*values)
        self.positive_G, self.negative_G, self.positive_X, self.negative_X, self.positive_Y, self.negative_Y = values
        zero = lambda t: 0
        source_G = self.source_functions.get('G', zero)(self.t)
        source_X = self.source_functions.get('X', zero)(self.t)
        source_Y = self.source_functions.get('Y', zero)(self.t)
        self.positive_G += 0.5*self.dt * source_G
        self.negative_G += 0.5*self.dt * source_G
        self.positive_X += 0.5*self.dt * source_X
        self.negative_X += 0.5*self.dt * source_X
        self.positive_Y += 0.5*self.dt * source_Y
        self.negative_Y += 0.5*self.dt * source_Y
        self.t += self.dt

    def numpy(self):
        G = self.positive_G + self.negative_G
        X = self.positive_X + self.negative_X
        Y = self.positive_Y + self.negative_Y
        return np.real(G.numpy()), np.real(X.numpy()), np.real(Y.numpy())
