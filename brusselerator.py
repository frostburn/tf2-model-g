import tensorflow as tf
import numpy as np

DEFAULT_PARAMS = {
    "A": 1.0,
    "B": 2.0,
    "k4": 1.0,
    "Dy": 8.0,
}


class Brusselerator(object):
    """
    Brusselerator is a simpler version of Model G that is either globally critical or sub-critical.

    Used as a sanity check to make sure that Model G produces the same behaviour when the G concentration is held constant
    """

    def __init__(self, concentration_X, concentration_Y, dx, params=None, fixed_point_iterations=3, iterations=2):
        if concentration_X.shape != concentration_Y.shape:
            raise ValueError("Concentration shapes must match")
        dims = len(concentration_X.shape)
        if dims == 1:
            kernel = [1, -2, 1]
            self.conv = tf.nn.conv1d
        elif dims == 2:
            kernel = [
                0, 1, 0,
                1, -4, 1,
                0, 1, 0,
            ]
            self.conv = tf.nn.conv2d
        elif dims == 3:
            kernel = [
                0, 0, 0,
                0, 1, 0,
                0, 0, 0,

                0, 1, 0,
                1, -6, 1,
                0, 1, 0,

                0, 0, 0,
                0, 1, 0,
                0, 0, 0,
            ]
            self.conv = tf.nn.conv3d
        else:
            raise ValueError("Only up to 3D supported")

        self.dx = dx
        self.kernel = tf.reshape(tf.constant(np.array(kernel) * dx*dx, dtype="float64"), [3]*dims + [1, 1])
        self.strides = [1] * (dims+2)
        self.pads = tf.constant([[0, 0]] + [[1, 1]]*dims + [[0, 0]])

        self.nshape = concentration_X.shape
        self.cshape = [1] + list(concentration_X.shape) + [1]
        self.concentration_X = tf.reshape(tf.constant(concentration_X, dtype="float64"), self.cshape)
        self.concentration_Y = tf.reshape(tf.constant(concentration_Y, dtype="float64"), self.cshape)
        self.dt = 0.1 * dx
        self.params = params or DEFAULT_PARAMS
        self.fixed_point_iterations = fixed_point_iterations
        self.iterations = iterations

        def laplacian(concentration):
            return self.conv(
                tf.pad(concentration, self.pads, 'REFLECT'),
                self.kernel, self.strides, 'VALID'
            )

        def integrator(con_X, con_Y):
            for _ in range(self.iterations):
                new_X = con_X
                new_Y = con_Y
                for _ in range(self.fixed_point_iterations):
                    xy_flow = new_X*new_X*new_Y - self.params["B"]*new_X
                    v_X = self.params["A"] + xy_flow - self.params["k4"] * new_X + laplacian(new_X)
                    v_Y = -xy_flow + self.params["Dy"] * laplacian(new_Y)
                    new_X = con_X + self.dt*v_X
                    new_Y = con_Y + self.dt*v_Y
                con_X = new_X
                con_Y = new_Y
            return con_X, con_Y

        self.integrator = tf.function(integrator)

    def step(self):
        self.concentration_X, self.concentration_Y = self.integrator(self.concentration_X, self.concentration_Y)

    def numpy(self):
        return self.concentration_X.numpy().reshape(self.nshape), self.concentration_Y.numpy().reshape(self.nshape)


if __name__ == '__main__':
    from util import bl_noise
    import pylab
    from matplotlib.animation import FuncAnimation

    x = np.linspace(-5, 5, 128)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)
    z = 0

    r2 = x*x+y*y+z*z
    brusselerator = Brusselerator(1.5 + np.exp(-0.2*r2)*bl_noise(x.shape), np.exp(-0.3*r2), dx, iterations=5)
    X, Y = brusselerator.numpy()
    plots = []
    for cross_section, mu in zip(X[60:80:2], np.linspace(0, 1, 10)):
        plots.extend(pylab.plot(x[0], cross_section, c=(0.5, 0.25, 0.6, 1.0-mu*0.8)))
    for cross_section, mu in zip(Y[60:80:2], np.linspace(0, 1, 10)):
        plots.extend(pylab.plot(x[0], cross_section, c=(0.7, 0.3, 0.1, 1.0-mu*0.8)))
    pylab.ylim(0.1, 2.9)

    def update(frame):
        brusselerator.step()
        X, Y = brusselerator.numpy()
        for plot, section in zip(plots, X[60:80:2]):
            plot.set_ydata(section)
        for plot, section in zip(plots[10:], Y[60:80:2]):
            plot.set_ydata(section)
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()
