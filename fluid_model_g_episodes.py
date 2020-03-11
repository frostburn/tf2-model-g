import numpy as np
from util import bl_noise, l2_location
from mpl_toolkits import mplot3d
import pylab
from matplotlib.animation import FuncAnimation
from fluid_model_g import FluidModelG


def nucleation_and_motion_in_G_gradient_2D(N=128, R=16):
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
        "viscosity": 0.4,
        "speed-of-sound": 1.0,
    }

    x = np.linspace(-R, R, N)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)

    def source_G(t):
        center = np.exp(-0.5*(t-5)**2) * 10
        gradient = (1+np.tanh(t-40)) * 0.0005
        return -np.exp(-0.5*(x*x+y*y))* center + (x+8) * gradient

    source_functions = {
        'G': source_G,
    }

    flow = [0*x, 0*x]

    fluid_model_g = FluidModelG(
        x*0,
        x*0,
        x*0,
        flow,
        dx,
        dt=0.25*dx,
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
            G[N//2],
            X[N//2] * x_scale,
            Y[N//2] * y_scale,
            u[N//2],
            v[N//2],
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


def nucleation_3D(animated=False, N=128, R=20):
    params = {
        "A": 3.4,
        "B": 13.5,
        "k2": 1.0,
        "k-2": 0.1,
        "k5": 0.9,
        "D_G": 1.0,
        "D_X": 1.0,
        "D_Y": 1.95,
        "density_G": 1.0,
        "density_X": 0.0002,
        "density_Y": 0.043,
        "base-density": 9.0,
        "viscosity": 0.4,
        "speed-of-sound": 1.0,
    }

    x = np.linspace(-R, R, N)
    dx = x[1] - x[0]
    x, y, z = np.meshgrid(x, x, x)


    def source_G(t):
        center = np.exp(-0.5*(t-5)**2) * 10
        return -np.exp(-0.5*(x*x+y*y+z*z)) * center

    source_functions = {
        'G': source_G,
    }

    # We need some noise to break spherical symmetry
    noise_scale = 1e-4
    G = bl_noise(x.shape) * noise_scale
    X = bl_noise(x.shape) * noise_scale
    Y = bl_noise(x.shape) * noise_scale
    flow = [
        bl_noise(x.shape) * noise_scale,
        bl_noise(x.shape) * noise_scale,
        bl_noise(x.shape) * noise_scale
    ]

    dt = 0.2*dx
    fluid_model_g = FluidModelG(
        G, X, Y,
        flow,
        dx,
        dt=dt,
        params=params,
        source_functions=source_functions,
    )

    if animated:
        def get_data():
            G, X, Y, (u, v, w) = fluid_model_g.numpy()

            x_scale = 0.1
            y_scale = 0.1
            return (
                G[N//2,N//2],
                X[N//2,N//2] * x_scale,
                Y[N//2,N//2] * y_scale,
                u[N//2,N//2],
                v[N//2,N//2],
                w[N//2,N//2],
            )

        G, X, Y, u, v, w = get_data()
        plots = []
        plots.extend(pylab.plot(z[0,0], G))
        plots.extend(pylab.plot(z[0,0], X))
        plots.extend(pylab.plot(z[0,0], Y))
        plots.extend(pylab.plot(z[0,0], u))
        plots.extend(pylab.plot(z[0,0], v))
        pylab.ylim(-0.1, 0.1)

        def update(frame):
            fluid_model_g.step()
            G, X, Y, u, v, w = get_data()
            print(fluid_model_g.t, abs(G).max(), abs(u).max())
            plots[0].set_ydata(G)
            plots[1].set_ydata(X)
            plots[2].set_ydata(Y)
            plots[3].set_ydata(u)
            plots[4].set_ydata(v)
            plots[4].set_ydata(w)
            return plots

        FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
        pylab.show()
    else:
        from datetime import datetime
        from pathlib import Path
        start = datetime.now()

        num_steps = int(1.0/dt)
        print("Starting simulation {} steps at a time".format(num_steps))

        path = Path("/tmp/model_g")
        path.mkdir(exist_ok=True)
        while True:
            for _ in range(num_steps):
                fluid_model_g.step()
            print("Saving a snapshot into {}".format(path))
            G, X, Y, (u, v, w) = fluid_model_g.numpy()
            np.save(path / 'G.npy', G)
            np.save(path / 'X.npy', X)
            np.save(path / 'Y.npy', Y)
            np.save(path / 'u.npy', u)
            np.save(path / 'v.npy', v)
            np.save(path / 'w.npy', w)
            print("Saved everything")

            wall_clock_time = (datetime.now() - start).total_seconds()
            print("t={}, wall clock time={} s, efficiency={}".format(fluid_model_g.t, wall_clock_time, fluid_model_g.t / wall_clock_time))
            print("max|G|={}, max|u|={}".format(abs(G).max(), abs(u).max()))




if __name__ == '__main__':
    nucleation_3D()
    # nucleation_and_motion_in_G_gradient_2D()
