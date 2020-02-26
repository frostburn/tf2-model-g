import numpy as np
from util import bl_noise
import pylab
from matplotlib.animation import FuncAnimation
from model_g import ModelG

def self_stabilizing_soliton_2D():
    params = {
        "A": 4.2,
        "B": 18,
        "k2": 1.0,
        "k-2": 0.2,
        "k5": 0.9,
        "Dx": 1.0,
        "Dy": 2.0,
    }

    x = np.linspace(-16, 16, 256)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)

    r2 = x*x+y*y
    model_g = ModelG(
        -np.exp(-0.1*r2)*1.0,
        np.exp(-r2)*0.01,
        np.exp(-r2)*0.01 + bl_noise(x.shape)*0.02,
        dx,
        params,
    )

    def get_data():
        G, X, Y = model_g.numpy()
        x_scale = 0.2
        y_scale = 0.1
        return (
            G[64],
            X[64] * x_scale,
            Y[64] * y_scale,
        )

    G, X, Y = get_data()
    plots = []
    plots.extend(pylab.plot(x[0], G))
    plots.extend(pylab.plot(x[0], X))
    plots.extend(pylab.plot(x[0], Y))
    pylab.ylim(-0.03, 0.03)

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

    r2 = x*x+y*y+z*z
    model_g = ModelG(
        -np.exp(-0.1*r2)*1.0,
        -np.exp(-r2)*0.01,
        np.exp(-r2)*0.01 + bl_noise(x.shape)*0.02,
        dx,
        params,
    )

    def get_data():
        G, X, Y = model_g.numpy()
        x_scale = 0.5
        y_scale = 0.04
        return (
            G[64, 64],
            X[64, 64] * x_scale,
            Y[64, 64] * y_scale,
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


# XXX: Nucleates, but the "movement" is just more catastrophic nucleation
def nucleation_and_motion_in_G_gradient_1D():
    params = {
        "A": 3.611,
        "B": 13.8,
        "k2": 1.0,
        "k-2": 0.2,
        "k5": 0.9,
        "Dx": 1.0,
        "Dy": 2.225,
    }

    x = np.linspace(-32, 32, 256)
    dx = x[1] - x[0]

    def source_G(t):
        print(t)
        return -np.exp(-0.5*x*x)*np.exp(-0.1*(t-5)**2) * 2.5 + x*0.00015 * (1+np.tanh(t-50))

    source_functions = {
        'G': source_G
    }

    r2 = x*x
    model_g = ModelG(
        np.exp(-0.1*r2)*0,
        np.exp(-r2)*0.01*0,
        np.exp(-r2)*0.01*0,
        dx,
        params,
        source_functions=source_functions,
    )

    def get_data():
        G, X, Y = model_g.numpy()
        x_scale = 0.1
        y_scale = 0.1
        return (
            G,
            X * x_scale,
            Y * y_scale,
        )

    G, X, Y = get_data()
    plots = []
    plots.extend(pylab.plot(x, G))
    plots.extend(pylab.plot(x, X))
    plots.extend(pylab.plot(x, Y))
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


# XXX: Nucleates, but doesn't move
def nucleation_and_motion_in_G_gradient_2D():
    params = {
        "A": 3.42,
        "B": 13.5,
        "k2": 1.0,
        "k-2": 0.1,
        "k5": 0.9,
        "Dx": 1.0,
        "Dy": 1.95,
    }

    x = np.linspace(-16, 16, 128)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)

    def source_G(t):
        print(t)
        return -np.exp(-0.5*(x*x+y*y))*np.exp(-0.1*(t-5)**2) * 2 + x*0.0005 * (1+np.tanh(t-20))

    source_functions = {
        'G': source_G,
    }

    r2 = x*x+y*y
    model_g = ModelG(
        -np.exp(-0.1*r2)*0,
        -np.exp(-r2)*0.01*0,
        np.exp(-r2)*0.01*0,
        dx,
        params,
        source_functions=source_functions,
    )

    def get_data():
        G, X, Y = model_g.numpy()
        x_scale = 0.1
        y_scale = 0.1
        return (
            G[64],
            X[64] * x_scale,
            Y[64] * y_scale,
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
    pylab.imshow(X)
    pylab.show()


if __name__ == '__main__':
    # nucleation_and_motion_in_G_gradient_1D()
    nucleation_and_motion_in_G_gradient_2D()
    # self_stabilizing_soliton_2D()
    # self_stabilizing_soliton_3D()
