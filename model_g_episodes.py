import numpy as np
from util import bl_noise
from mpl_toolkits import mplot3d
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


def random_2D():
    r = np.random.randn
    params = {
        "A": 2 + r()*0.1,
        "B": 10 + r(),
        "k2": 1.0 + 0.1*r(),
        "k-2": 0.1 + 0.01*r(),
        "k5": 0.9 + 0.1*r(),
        "Dx": 1.0 + 0.1*r(),
        "Dy": 2.0 + 0.1*r(),
    }
    print(params)

    x = np.linspace(-16, 16, 256)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)

    def source_G(t):
        center = np.exp(-0.5*(t-20)**2) * 10
        gradient = (1+np.tanh(t-40)) * 0.0005
        print("t = {}\tcenter potential = {}\tx-gradient = {}".format(t, center, gradient))
        return -np.exp(-0.5*(x*x+y*y)) * center + x * gradient

    source_functions = {
        'G': source_G,
    }

    model_g = ModelG(
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
        "B": 10 + r(),
        "k2": 1.0 + 0.1*r(),
        "k-2": 0.1 + 0.01*r(),
        "k5": 0.9 + 0.1*r(),
        "Dx": 1.0 + 0.1*r(),
        "Dy": 2.0 + 0.1*r(),
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

    model_g = ModelG(
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
            G[64, 64],
            X[64, 64] * x_scale,
            Y[64, 64] * y_scale,
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
    plots = [pylab.imshow(X[64])]

    def update(frame):
        model_g.step()
        G, X, Y = model_g.numpy()
        plots[0].set_data(X[64])
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()

    fig = pylab.figure()
    ax = fig.add_subplot(111, projection='3d')


    G, X, Y = model_g.numpy()
    m = X.max() - (X.max() - X.min()) * 0.3
    points = []
    for _ in range(1000000):
        px = np.random.randint(x.shape[0])
        py = np.random.randint(y.shape[1])
        pz = np.random.randint(z.shape[2])

        c = X[px, py, pz]

        if c > m:
            points.append((px, py, pz, c))
            if len(points) > 20000:
                break

    xs, ys, zs, cs = zip(*points)
    ax.scatter3D(xs, ys, zs, c=cs)
    pylab.show()


if __name__ == '__main__':
    random_3D()
    # nucleation_and_motion_in_G_gradient_1D()
    # nucleation_and_motion_in_G_gradient_2D()
    # self_stabilizing_soliton_2D()
    # self_stabilizing_soliton_3D()
