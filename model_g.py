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


def steady_state(params=DEFAULT_PARAMS):
    G0 = params["A"]*(params["k5"] + params["k-2"])/(params["k2"]*params["k5"])
    X0 = params["A"]/params["k5"]
    Y0 = params["B"]*params["k5"]/params["A"]
    return G0, X0, Y0


class ModelG(object):
    """
    Model G Reaction Diffusion system
    """

    def __init__(self, concentration_G, concentration_X, concentration_Y, dx, params=None, fixed_point_iterations=3):
        if concentration_X.shape != concentration_Y.shape or concentration_X.shape != concentration_G.shape:
            raise ValueError("Concentration shapes must match")
        self.dx = dx
        self.dt = 0.1 * dx

        self.concentration_G = tf.constant(concentration_G, dtype="float64")
        self.concentration_X = tf.constant(concentration_X, dtype="float64")
        self.concentration_Y = tf.constant(concentration_Y, dtype="float64")
        self.params = params or DEFAULT_PARAMS
        self.fixed_point_iterations = fixed_point_iterations

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

        def reaction_integrator(con_G, con_X, con_Y):
            new_G = con_G
            new_X = con_X
            new_Y = con_Y
            for _ in range(self.fixed_point_iterations):
                gx_flow = self.params["k-2"]*new_X - self.params["k2"]*new_G
                xy_flow = new_X*new_X*new_Y - self.params["B"]*new_X
                v_G = self.params["A"] + gx_flow
                v_X = xy_flow - gx_flow - self.params["k5"] * new_X
                v_Y = -xy_flow
                new_G = con_G + self.dt*v_G
                new_X = con_X + self.dt*v_X
                new_Y = con_Y + self.dt*v_Y
            con_G = new_G
            con_X = new_X
            con_Y = new_Y
            return con_G, con_X, con_Y

        self.diffusion_integrator = tf.function(diffusion_integrator)
        self.reaction_integrator = tf.function(reaction_integrator)

    def step(self):
        values = self.diffusion_integrator(self.concentration_G, self.concentration_X, self.concentration_Y)
        self.concentration_G, self.concentration_X, self.concentration_Y = values
        values = self.reaction_integrator(self.concentration_G, self.concentration_X, self.concentration_Y)
        self.concentration_G, self.concentration_X, self.concentration_Y = values

    def numpy(self):
        return (
            self.concentration_G.numpy(),
            self.concentration_X.numpy(),
            self.concentration_Y.numpy()
        )


if __name__ == '__main__':
    from util import bl_noise
    import pylab
    from matplotlib.animation import FuncAnimation

    x = np.linspace(-16, 16, 128)
    dx = x[1] - x[0]
    x, y, z = np.meshgrid(x, x, x)

    G0, X0, Y0 = steady_state()

    r2 = x*x+y*y+z*z
    model_g = ModelG(
        G0 - np.exp(-0.1*r2)*1.0,
        X0 - np.exp(-r2)*0.01,
        Y0 + np.exp(-r2)*0.01 + bl_noise(x.shape)*0.02,
        dx,
    )
    G, X, Y = model_g.numpy()
    plots = []
    plots.extend(pylab.plot(z[0,0], (G - G0)[64,64]))
    plots.extend(pylab.plot(z[0,0], (X - X0)[64,64]))
    plots.extend(pylab.plot(z[0,0], (Y - Y0)[64,64]))
    x_scale = 1.0
    y_scale = 0.05
    # s = slice(60, 80, 2)
    # for cross_section, mu in zip(G[s]-G0, np.linspace(0, 1, 10)):
    #     plots.extend(pylab.plot(x[0], cross_section, c=(0.25, 0.8, 0.6, 1.0-mu*0.8)))
    # for cross_section, mu in zip((X[s]-X0)*x_scale, np.linspace(0, 1, 10)):
    #     plots.extend(pylab.plot(x[0], cross_section, c=(0.5, 0.25, 0.6, 1.0-mu*0.8)))
    # for cross_section, mu in zip((Y[s]-Y0)*y_scale, np.linspace(0, 1, 10)):
    #     plots.extend(pylab.plot(x[0], cross_section, c=(0.7, 0.3, 0.1, 1.0-mu*0.8)))
    pylab.ylim(-0.4, 0.4)

    def update(frame):
        for _ in range(5):
            model_g.step()
        G, X, Y = model_g.numpy()
        plots[0].set_ydata((G - G0)[64,64])
        plots[1].set_ydata(((X - X0)*x_scale)[64,64])
        plots[2].set_ydata(((Y - Y0)*y_scale)[64,64])
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()

    G, X, Y = model_g.numpy()
    pylab.imshow(Y[64])
    pylab.show()
