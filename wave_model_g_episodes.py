import numpy as np
from util import bl_noise, l2_location
import pylab
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
from wave_model_g import WaveModelG


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

    r2 = x*x+y*y
    model_g = WaveModelG(
        np.exp(-0.1*r2)*1.0 + bl_noise(x.shape)*0.01,
        np.exp(-r2)*0.01 + bl_noise(x.shape)*0.01,
        np.exp(-r2)*0.01 + bl_noise(x.shape)*0.02,
        dx,
        params,
    )

    def get_data():
        G, X, Y = model_g.numpy()
        x_scale = 0.02
        y_scale = 0.02
        return G, X, Y
        return (
            G[64],
            X[64] * x_scale,
            Y[64] * y_scale,
        )

    G, X, Y = get_data()
    plots = []
    plots.append(pylab.imshow(G, vmin=-0.1, vmax=0.1))
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


def nucleation_and_motion_in_G_gradient_2D():
    params = {
        "A": 3.42,
        "B": 13.3,
        "k2": 1.0,
        "k-2": 0.11,
        "k5": 0.9,
        "Dg": 0.5,
        "Dx": 1.0,
        "Dy": 2.0,
        "cg": 1.0,
        "cx": 1.0,
        "cy": 1.0,
    }

    N = 256
    x = np.linspace(-32, 32, N)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)

    def source_G(t):
        center = np.exp(-0.5*(t-3)**2) * 5
        gradient = (1+np.tanh(t-40)) * 0.0005
        # print("t = {}\tcenter potential = {}\tx-gradient = {}".format(t, center, gradient))
        return -np.exp(-0.5*(x*x+y*y))* center + (0.2*x+8) * gradient

    source_functions = {
        'G': source_G,
    }

    r2 = x*x+y*y
    model_g = WaveModelG(
        np.exp(-0.1*r2)*0,
        np.exp(-r2)*0.01*0,
        np.exp(-r2)*0.01*0,
        dx,
        dt=0.025*dx,
        params=params,
        source_functions=source_functions,
    )

    times = []
    locs = []

    def get_data():
        G, X, Y = model_g.numpy()

        loc = l2_location(X, x, y)
        times.append(model_g.t)
        locs.append(loc[0])
        print("t={}\tL2 location: {}".format(model_g.t, tuple(loc)))

        x_scale = 0.1
        y_scale = 0.1
        return (
            G[N//2],
            X[N//2] * x_scale,
            Y[N//2] * y_scale,
        )

    G, X, Y = get_data()
    plots = []
    plots.extend(pylab.plot(x[0], G))
    plots.extend(pylab.plot(x[0], X))
    plots.extend(pylab.plot(x[0], Y))
    pylab.ylim(-0.1, 0.1)

    def update(frame):
        for _ in range(10):
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

    pylab.plot(times, locs)
    pylab.show()


def random_2D():
    r = np.random.randn
    params = {
        "A": 2 + r()*0.1,
        "B": 10 + r(),
        "k2": 1.0 + 0.1*r(),
        "k-2": 0.1 + 0.01*r(),
        "k5": 0.9 + 0.1*r(),
        "Dg": 0.05 + 0.01*r(),
        "Dx": 0.05 + 0.01*r(),
        "Dy": 0.1 + 0.01*r(),
        "cg": 1.0 + 0.1*r(),
        "cx": 1.5 + 0.1*r(),
        "cy": 2.0 + 0.1*r(),
    }
    print(params)

    x = np.linspace(-16, 16, 256)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)

    def source_G(t):
        center = np.exp(-0.5*(t-10)**2) * 10
        gradient = (1+np.tanh(t-20)) * 0.05
        print("t = {}\tcenter potential = {}\tx-gradient = {}".format(t, center, gradient))
        return -np.exp(-0.1*(x*x+y*y)) * center + x * gradient

    source_functions = {
        'G': source_G,
    }

    model_g = WaveModelG(
        bl_noise(x.shape)*0.01,
        bl_noise(x.shape)*0.01,
        bl_noise(x.shape)*0.01,
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
    pylab.ylim(-0.5, 0.5)

    def update(frame):
        model_g.step()
        G, X, Y = get_data()
        plots[0].set_ydata(G)
        plots[1].set_ydata(X)
        plots[2].set_ydata(Y)
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()

    G, X, Y = model_g.numpy()
    plots = [pylab.imshow(X)]

    def update(frame):
        model_g.step()
        G, X, Y = model_g.numpy()
        plots[0].set_data(X)
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()


def random_3D():
    r = np.random.randn
    params = {
        "A": 2 + r()*0.1,
        "B": 8.5 + r(),
        "k2": 0.6 + 0.1*r(),
        "k-2": 0.1 + 0.01*r(),
        "k5": 0.7 + 0.1*r(),
        "Dg": 0.1 + 0.01*r(),
        "Dx": 0.1 + 0.01*r(),
        "Dy": 0.2 + 0.01*r(),
        "cg": 0.2 + 0.05*r(),
        "cx": 0.5 + 0.1*r(),
        "cy": 0.6 + 0.1*r(),
    }
    print(params)

    x = np.linspace(-16, 16, 128)
    dx = x[1] - x[0]
    x, y, z = np.meshgrid(x, x, x)

    def source_G(t):
        center = np.exp(-0.5*(t-20)**2) * 10
        gradient = (1+np.tanh(t-40)) * 0.0005
        print("t = {}\tcenter potential = {}\tx-gradient = {}".format(t, center, gradient))
        return -np.exp(-0.5*(x*x+y*y+z*z)) * center + x * gradient

    source_functions = {
        'G': source_G,
    }

    model_g = WaveModelG(
        bl_noise(x.shape)*0.01,
        bl_noise(x.shape)*0.01,
        bl_noise(x.shape)*0.01,
        dx,
        params,
        source_functions=source_functions,
    )

    def get_data():
        G, X, Y = model_g.numpy()
        x_scale = 0.1
        y_scale = 0.1
        return (
            G[0, 0],
            X[0, 0] * x_scale,
            Y[0, 0] * y_scale,
        )

    G, X, Y = get_data()
    plots = []
    plots.extend(pylab.plot(z[0,0], G))
    plots.extend(pylab.plot(z[0,0], X))
    plots.extend(pylab.plot(z[0,0], Y))
    pylab.ylim(-0.5, 0.5)

    def update(frame):
        model_g.step()
        G, X, Y = get_data()
        plots[0].set_ydata(G)
        plots[1].set_ydata(X)
        plots[2].set_ydata(Y)
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()

    G, X, Y = model_g.numpy()
    plots = [pylab.imshow(X[0])]

    def update(frame):
        model_g.step()
        G, X, Y = model_g.numpy()
        plots[0].set_data(X[0])
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()

    fig = pylab.figure()
    ax = fig.add_subplot(111, projection='3d')


    G, X, Y = model_g.numpy()
    m = X.max() - (X.max() - X.min()) * 0.2
    points = []
    for _ in range(1000000):
        px = np.random.randint(x.shape[0])
        py = np.random.randint(y.shape[1])
        pz = np.random.randint(z.shape[2])

        c = X[px, py, pz]

        if c > m:
            points.append((px, py, pz, c))
            if len(points) > 10000:
                break

    xs, ys, zs, cs = zip(*points)
    ax.scatter3D(xs, ys, zs, c=cs)
    pylab.show()


if __name__ == '__main__':
    # random_3D()
    nucleation_and_motion_in_G_gradient_2D()
    # morphing_hex_grid_2D()
