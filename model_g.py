import tensorflow as tf
import numpy as np
from integrator import Integrator

DEFAULT_PARAMS = {
    "A": 3.0,
    "B": 12.4,
    "k2": 0.9,
    "k_2": 0.1,
    "k5": 0.9,
    "Dx": 1.0,
    "Dy": 8.0,
}


def steady_state(params=DEFAULT_PARAMS):
    G0 = params["A"]*(params["k5"] + params["k_2"])/(params["k2"]*params["k5"])
    X0 = params["A"]/params["k5"]
    Y0 = params["B"]*params["k5"]/params["A"]
    return G0, X0, Y0


class ModelG(Integrator):
    """
    Model G Reaction Diffusion system
    """

    def __init__(self, concentration_G, concentration_X, concentration_Y, dx, params=None, fixed_point_iterations=3, iterations=2):
        if concentration_X.shape != concentration_Y.shape:
            raise ValueError("Concentration shapes must match")
        super().__init__(dx=dx, dt=dx*0.03, dims=len(concentration_X.shape))

        self.nshape = concentration_X.shape
        self.cshape = [1] + list(concentration_X.shape) + [1]
        self.concentration_G = tf.reshape(tf.constant(concentration_G, dtype="float64"), self.cshape)
        self.concentration_X = tf.reshape(tf.constant(concentration_X, dtype="float64"), self.cshape)
        self.concentration_Y = tf.reshape(tf.constant(concentration_Y, dtype="float64"), self.cshape)
        self.params = params or DEFAULT_PARAMS
        self.fixed_point_iterations = fixed_point_iterations
        self.iterations = iterations

        def laplacian(concentration):
            return self.conv(
                tf.pad(concentration, self.pads, 'REFLECT'),
                self.kernel, self.strides, 'VALID'
            )

        def integrator(con_G, con_X, con_Y):
            for _ in range(self.iterations):
                new_G = con_G
                new_X = con_X
                new_Y = con_Y
                for _ in range(self.fixed_point_iterations):
                    gx_flow = self.params["k_2"]*new_X - self.params["k2"]*new_G
                    xy_flow = new_X*new_X*new_Y - self.params["B"]*new_X
                    v_G = self.params["A"] + gx_flow + laplacian(new_G)
                    v_X = xy_flow - gx_flow - self.params["k5"] * new_X + self.params["Dx"] * laplacian(new_X)
                    v_Y = -xy_flow + self.params["Dy"] * laplacian(new_Y)
                    new_G = con_G + self.dt*v_G
                    new_X = con_X + self.dt*v_X
                    new_Y = con_Y + self.dt*v_Y
                con_G = new_G
                con_X = new_X
                con_Y = new_Y
            return con_G, con_X, con_Y

        self.integrator = tf.function(integrator)

    def step(self):
        values = self.integrator(self.concentration_G, self.concentration_X, self.concentration_Y)
        self.concentration_G, self.concentration_X, self.concentration_Y = values

    def numpy(self):
        return (
            self.concentration_G.numpy().reshape(self.nshape),
            self.concentration_X.numpy().reshape(self.nshape),
            self.concentration_Y.numpy().reshape(self.nshape)
        )


if __name__ == '__main__':
    from util import bl_noise
    import pylab
    from matplotlib.animation import FuncAnimation

    x = np.linspace(-8, 8, 512)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)
    z = 0

    G0, X0, Y0 = steady_state()

    r2 = x*x+y*y+z*z
    model_g = ModelG(
        G0 - np.exp(-0.1*r2)*1.5,
        X0 + np.exp(-0.2*r2)*bl_noise(x.shape, range(4, 12))*0.2,
        Y0 + np.exp(-0.3*r2)*bl_noise(x.shape)*0.1, dx,
        iterations=10
    )
    G, X, Y = model_g.numpy()
    plots = []
    x_scale = 1.0
    y_scale = 1.0
    s = slice(60, 80, 2)
    for cross_section, mu in zip(G[s]-G0, np.linspace(0, 1, 10)):
        plots.extend(pylab.plot(x[0], cross_section, c=(0.25, 0.8, 0.6, 1.0-mu*0.8)))
    for cross_section, mu in zip((X[s]-X0)*x_scale, np.linspace(0, 1, 10)):
        plots.extend(pylab.plot(x[0], cross_section, c=(0.5, 0.25, 0.6, 1.0-mu*0.8)))
    for cross_section, mu in zip((Y[s]-Y0)*y_scale, np.linspace(0, 1, 10)):
        plots.extend(pylab.plot(x[0], cross_section, c=(0.7, 0.3, 0.1, 1.0-mu*0.8)))
    pylab.ylim(-2, 2)

    def update(frame):
        model_g.step()
        G, X, Y = model_g.numpy()
        for plot, section in zip(plots, G[s]-G0):
            plot.set_ydata(section)
        for plot, section in zip(plots[10:], (X[s]-X0)*x_scale):
            plot.set_ydata(section)
        for plot, section in zip(plots[20:], (Y[s]-Y0)*y_scale):
            plot.set_ydata(section)
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()
