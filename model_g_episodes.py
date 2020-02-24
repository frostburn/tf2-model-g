import numpy as np
from util import bl_noise
import pylab
from matplotlib.animation import FuncAnimation
from model_g import ModelG, steady_state

def self_stabilizing_soliton_2D():
    # TODO: Needs some parameter fiddling
    params = {
        "A": 3.42,
        "B": 13.5,
        "k2": 1.0,
        "k-2": 0.1,
        "k5": 0.9,
        "Dx": 1.0,
        "Dy": 2.0,
    }

    x = np.linspace(-16, 16, 256)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)

    G0, X0, Y0 = steady_state(params)

    r2 = x*x+y*y
    model_g = ModelG(
        G0 - np.exp(-0.1*r2)*1.0,
        X0 - np.exp(-r2)*0.01,
        Y0 + np.exp(-r2)*0.01 + bl_noise(x.shape)*0.02,
        dx,
        params,
    )

    def get_data():
        G, X, Y = model_g.numpy()
        x_scale = 0.5
        y_scale = 0.04
        return (
            G[64] - G0,
            (X[64] - X0) * x_scale,
            (Y[64] - Y0) * y_scale,
        )

    G, X, Y = get_data()
    plots = []
    plots.extend(pylab.plot(x[0], G))
    plots.extend(pylab.plot(x[0], X))
    plots.extend(pylab.plot(x[0], Y))
    pylab.ylim(-0.1, 0.1)

    def update(frame):
        for _ in range(5):
            model_g.step()
        G, X, Y = get_data()
        plots[0].set_ydata(G)
        plots[1].set_ydata(X)
        plots[2].set_ydata(Y)
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()

    G, X, Y = model_g.numpy()
    pylab.imshow(Y)
    pylab.show()


def self_stabilizing_soliton_3D():
    params = {
        "A": 3.42,
        "B": 13.5,
        "k2": 1.0,
        "k-2": 0.1,
        "k5": 0.9,
        "Dx": 1.0,
        "Dy": 2.0,
    }

    x = np.linspace(-16, 16, 128)
    dx = x[1] - x[0]
    x, y, z = np.meshgrid(x, x, x)

    G0, X0, Y0 = steady_state(params)

    r2 = x*x+y*y+z*z
    model_g = ModelG(
        G0 - np.exp(-0.1*r2)*1.0,
        X0 - np.exp(-r2)*0.01,
        Y0 + np.exp(-r2)*0.01 + bl_noise(x.shape)*0.02,
        dx,
        params,
    )

    def get_data():
        G, X, Y = model_g.numpy()
        x_scale = 0.5
        y_scale = 0.04
        return (
            G[64, 64] - G0,
            (X[64, 64] - X0) * x_scale,
            (Y[64, 64] - Y0) * y_scale,
        )

    G, X, Y = get_data()
    plots = []
    plots.extend(pylab.plot(z[0,0], G))
    plots.extend(pylab.plot(z[0,0], X))
    plots.extend(pylab.plot(z[0,0], Y))
    pylab.ylim(-0.4, 0.4)

    def update(frame):
        for _ in range(5):
            model_g.step()
        G, X, Y = get_data()
        plots[0].set_ydata(G)
        plots[1].set_ydata(X)
        plots[2].set_ydata(Y)
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()

    G, X, Y = model_g.numpy()
    pylab.imshow(Y[64])
    pylab.show()


if __name__ == '__main__':
    self_stabilizing_soliton_2D()
    # self_stabilizing_soliton_3D()
