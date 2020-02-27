from pylab import *
from integrators.model_g import polynomial_order_4_centered as reaction_integrator

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


if __name__ == '__main__':
    random_0D_trajectories()
