import tensorflow as tf
import numpy as np

DEFAULT_PARAMS = {
    "A": 3.42,
    "B": 13.5,
    "k2": 1.0,
    "k-2": 0.1,
    "k5": 0.9,
    "Dx": 1.0,
    "Dy": 2.0,
}


def integrate_model_g_centered(a0, b0, c0, t, A, B, k2, k_2, k5):
    """
    Integrate 0-dimensional Model G starting from a known position `t` time units ahead.
    The system has been "centered" so that the origin becomes a fixed point.
    The coefficients have been derived using a computer algebra system by developing the solution into a series with respect to t around G,X,Y == a0,b0,c0.
    """
    a1 = -a0*k2 + b0*k_2
    c1 = -b0**2*c0 - B*b0**2*k5/A - B*b0 - 2*A*b0*c0/k5 - A**2*c0/k5**2
    b1 = -c1 + a0*k2 - b0*k5 - b0*k_2
    a2 = -1/2*a1*k2 + 1/2*b1*k_2
    c2 = -b0*b1*c0 - 1/2*b0**2*c1 - B*b0*b1*k5/A - 1/2*B*b1 - A*b1*c0/k5 - A*b0*c1/k5 - 1/2*A**2*c1/k5**2
    b2 = -c2 + 1/2*a1*k2 - 1/2*b1*k5 - 1/2*b1*k_2
    a3 = -1/3*a2*k2 + 1/3*b2*k_2
    c3 = -1/3*b1**2*c0 - 2/3*b0*b2*c0 - 2/3*b0*b1*c1 - 1/3*b0**2*c2 - 1/3*B*b1**2*k5/A - 2/3*B*b0*b2*k5/A - 1/3*B*b2 - 2/3*A*b2*c0/k5 - 2/3*A*b1*c1/k5 - 2/3*A*b0*c2/k5 - 1/3*A**2*c2/k5**2
    b3 = -c3 + 1/3*a2*k2 - 1/3*b2*k5 - 1/3*b2*k_2
    a4 = -1/4*a3*k2 + 1/4*b3*k_2
    c4 = -1/2*b1*b2*c0 - 1/2*b0*b3*c0 - 1/4*b1**2*c1 - 1/2*b0*b2*c1 - 1/2*b0*b1*c2 - 1/4*b0**2*c3 - 1/2*B*b1*b2*k5/A - 1/2*B*b0*b3*k5/A - 1/4*B*b3 - 1/2*A*b3*c0/k5 - 1/2*A*b2*c1/k5 - 1/2*A*b1*c2/k5 - 1/2*A*b0*c3/k5 - 1/4*A**2*c3/k5**2
    b4 = -c4 + 1/4*a3*k2 - 1/4*b3*k5 - 1/4*b3*k_2

    return (
        a0 + t * (a1 + t * (a2 + t * (a3 + t*a4))),
        b0 + t * (b1 + t * (b2 + t * (b3 + t*b4))),
        c0 + t * (c1 + t * (c2 + t * (c3 + t*c4))),
    )


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

        reaction_integrator = lambda con_G, con_X, con_Y: integrate_model_g_centered(
            con_G, con_X, con_Y,
            self.dt, self.params['A'], self.params['B'], self.params['k2'], self.params['k-2'], self.params['k5']
        )

        self.diffusion_integrator = tf.function(diffusion_integrator)
        self.reaction_integrator = tf.function(reaction_integrator)

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
