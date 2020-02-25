import numpy as np
from util import bl_noise
import pylab
from matplotlib.animation import FuncAnimation
from wave_model_g import WaveModelG, steady_state


def morphing_hex_grid_2D():
    params = {
        "A": 3.42,
        "B": 13.4,
        "k2": 1.0,
        "k-2": 0.1,
        "k5": 0.9,
        "Dg": 0.01,
        "Dx": 0.92,
        "Dy": 2.1,
        "cg": 0.9,
        "cx": 0.3,
        "cy": 0.15,
    }

    x = np.linspace(-16, 16, 128)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)

    G0, X0, Y0 = steady_state(params)

    r2 = x*x+y*y
    model_g = WaveModelG(
        G0 - np.exp(-0.1*r2)*1.0 + bl_noise(x.shape)*0.01,
        X0 - np.exp(-r2)*0.01 + bl_noise(x.shape)*0.01,
        Y0 + np.exp(-r2)*0.01 + bl_noise(x.shape)*0.02,
        dx,
        params,
    )

    def get_data():
        G, X, Y = model_g.numpy()
        x_scale = 0.02
        y_scale = 0.02
        return G, X, Y
        return (
            G[64] - G0,
            (X[64] - X0) * x_scale,
            (Y[64] - Y0) * y_scale,
        )

    G, X, Y = get_data()
    plots = []
    plots.append(pylab.imshow(G, vmin=G0-0.1, vmax=G0+0.1))
    # plots.extend(pylab.plot(x[0], G, label="G"))
    # plots.extend(pylab.plot(x[0], X, label="X"))
    # plots.extend(pylab.plot(x[0], Y, label="Y"))
    # pylab.legend()
    # pylab.ylim(-0.1, 0.1)

    def update(frame):
        for _ in range(5):
            model_g.step()
        G, X, Y = get_data()
        plots[0].set_data(G)
        # plots[0].set_ydata(G)
        # plots[1].set_ydata(X)
        # plots[2].set_ydata(Y)
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()

    G, X, Y = model_g.numpy()
    pylab.imshow(Y)
    pylab.show()


# XXX: Nucleates, but doesn't move
def nucleation_and_motion_in_G_gradient_2D():
    params = {
        "A": 3.42,
        "B": 13.5,
        "k2": 1.0,
        "k-2": 0.1,
        "k5": 0.9,
        "Dg": 0.3,
        "Dx": 1.0,
        "Dy": 2.0,
        "cg": 2.0,
        "cx": 1.0,
        "cy": 1.0,
    }

    x = np.linspace(-16, 16, 128)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)

    G0, X0, Y0 = steady_state(params)

    def source_G(t):
        print(t)
        return -np.exp(-0.5*(x*x+y*y))*np.exp(-0.1*(t-5)**2) * 2 + x*0.0005 * (1+np.tanh(t-20))

    source_functions = {
        'G': source_G,
    }

    r2 = x*x+y*y
    model_g = WaveModelG(
        G0 - np.exp(-0.1*r2)*0,
        X0 - np.exp(-r2)*0.01*0,
        Y0 + np.exp(-r2)*0.01*0,
        dx,
        params,
        source_functions=source_functions,
    )

    def get_data():
        G, X, Y = model_g.numpy()
        x_scale = 0.1
        y_scale = 0.1
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
    pylab.imshow(X)
    pylab.show()


if __name__ == '__main__':
    nucleation_and_motion_in_G_gradient_2D()
    # morphing_hex_grid_2D()
