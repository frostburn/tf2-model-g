from pylab import *
from integrators.model_g import polynomial_order_4_centered as reaction_integrator
from model_g import ModelG
from util import bl_noise


def get_trajectories(A, B, k2, k_2, k5):
    G, X, Y = randn(3, 10)
    Gs = [G]
    Xs = [X]
    Ys = [Y]

    for _ in range(1000):
        G, X, Y = reaction_integrator(G, X, Y, 0.02, A, B, k2, k_2, k5)
        Gs.append(G)
        Xs.append(X)
        Ys.append(Y)

    return array([Gs, Xs, Ys])


def random_0D_trajectories():
    fig, axs = subplots(2, 3)
    for ax_ in axs:
        for ax in ax_:
            while True:
                Gs, Xs, Ys = get_trajectories(rand()*40, rand()*40, rand()*2, rand()*2, rand()*2)
                if abs(Xs).max() < 100:
                    break
            ax.plot(Xs, Ys)
    show()


def random_1D_fields():
    x = linspace(-20, 20, 512)
    dx = x[1] - x[0]
    noise_scale = 1.0

    fig, axs = subplots(2, 3)
    for ax_ in axs:
        for ax in ax_:
            while True:
                params = {
                    "A": rand() * 20,
                    "B": rand() * 20,
                    "k2": rand() * 2,
                    "k-2": rand() * 2,
                    "k5": rand() * 2,
                    "Dx": rand() * 4,
                    "Dy": rand() * 4,
                }
                model_g = ModelG(
                    bl_noise(x.shape) * noise_scale,
                    bl_noise(x.shape) * noise_scale,
                    bl_noise(x.shape) * noise_scale,
                    dx,
                    params=params
                )

                while model_g.t < 20:
                    model_g.step()
                    if rand() < 0.05:
                        G, X, Y = model_g.numpy()
                        if abs(X).max() >= 100:
                            break

                G, X, Y = model_g.numpy()
                if abs(X).max() < 100:
                    break
                print("Rejected", params)

            print("Done", params)

            G /= abs(G).max()
            X /= abs(X).max()
            Y /= abs(Y).max()

            ax.plot(x, G)
            ax.plot(x, X)
            ax.plot(x, Y)
    show()


def random_2D_fields():
    x = linspace(-20, 20, 512)
    dx = x[1] - x[0]
    x, y = meshgrid(x, x)
    noise_scale = 1.0

    fig, axs = subplots(2, 3)
    for ax_ in axs:
        for ax in ax_:
            while True:
                params = {
                    "A": rand() * 20,
                    "B": rand() * 20,
                    "k2": rand() * 2,
                    "k-2": rand() * 2,
                    "k5": rand() * 2,
                    "Dx": rand() * 4,
                    "Dy": rand() * 4,
                }
                model_g = ModelG(
                    bl_noise(x.shape) * noise_scale,
                    bl_noise(x.shape) * noise_scale,
                    bl_noise(x.shape) * noise_scale,
                    dx,
                    params=params
                )

                while model_g.t < 3:
                    model_g.step()
                    if rand() < 0.05:
                        G, X, Y = model_g.numpy()
                        if abs(X).max() >= 100:
                            break

                G, X, Y = model_g.numpy()
                if abs(X).max() < 100:
                    break
                print("...Rejected", params)

            print("Done", params)

            G /= abs(G).max()
            X /= abs(X).max()
            Y /= abs(Y).max()

            ax.imshow(X, extent=(-20, 20, -20, 20))
    show()

if __name__ == '__main__':
    random_2D_fields()
    # random_1D_fields()
    #random_0D_trajectories()
