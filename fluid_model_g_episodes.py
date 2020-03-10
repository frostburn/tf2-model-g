import numpy as np
from util import bl_noise, l2_location
from mpl_toolkits import mplot3d
import pylab
from matplotlib.animation import FuncAnimation
from fluid_model_g import FluidModelG


def nucleation_and_motion_in_G_gradient_2D():
    params = {
        "A": 3.42,
        "B": 13.5,
        "k2": 1.0,
        "k-2": 0.1,
        "k5": 0.9,
        "D_G": 1.0,
        "D_X": 1.0,
        "D_Y": 1.95,
        "density_G": 2.0,
        "density_X": 1.0,
        "density_Y": 1.5,
        "base-density": 35.0,
        "viscosity": 0.6,
        "speed-of-sound": 0.4,
    }

    x = np.linspace(-16, 16, 128)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)

    def source_G(t):
        center = np.exp(-0.5*(t-5)**2) * 10
        gradient = (1+np.tanh(t-40)) * 0.0005
        # print("t = {}\tcenter potential = {}\tx-gradient = {}".format(t, center, gradient))
        return -np.exp(-0.5*(x*x+y*y))* center + (x+8) * gradient

    source_functions = {
        'G': source_G,
    }

    flow = [0*x, 0*x]

    r2 = x*x+y*y
    fluid_model_g = FluidModelG(
        -np.exp(-0.1*r2)*0,
        -np.exp(-r2)*0.01*0,
        np.exp(-r2)*0.01*0,
        flow,
        dx,
        dt=0.02*dx,
        params=params,
        source_functions=source_functions,
    )

    times = []
    locs = []

    def get_data():
        G, X, Y, (u, v) = fluid_model_g.numpy()

        loc = l2_location(X, x, y)
        times.append(fluid_model_g.t)
        locs.append(loc[0])
        print("t={}\tL2 location: {}".format(fluid_model_g.t, tuple(loc)))

        x_scale = 0.1
        y_scale = 0.1
        return (
            G[64],
            X[64] * x_scale,
            Y[64] * y_scale,
            u[64],
            v[64],
        )

    G, X, Y, u, v = get_data()
    plots = []
    plots.extend(pylab.plot(x[0], G))
    plots.extend(pylab.plot(x[0], X))
    plots.extend(pylab.plot(x[0], Y))
    plots.extend(pylab.plot(x[0], u))
    plots.extend(pylab.plot(x[0], v))
    pylab.ylim(-0.1, 0.1)

    def update(frame):
        for _ in range(20):
            fluid_model_g.step()
        G, X, Y, u, v = get_data()
        plots[0].set_ydata(G)
        plots[1].set_ydata(X)
        plots[2].set_ydata(Y)
        plots[3].set_ydata(u)
        plots[4].set_ydata(v)
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()

    G, X, Y, (u, v) = fluid_model_g.numpy()
    pylab.imshow(X)
    pylab.show()

    pylab.plot(times, locs)
    pylab.show()


if __name__ == '__main__':
    nucleation_and_motion_in_G_gradient_2D()
