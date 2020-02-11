import numpy as np
import datetime
from model_g import ModelG, steady_state
from util import bl_noise

x = np.linspace(-8, 8, 256)
dx = x[1] - x[0]
x, y, z= np.meshgrid(x, x, x)

G0, X0, Y0 = steady_state()

r2 = x*x+y*y+z*z
model_g = ModelG(
    G0 - np.exp(-0.1*r2)*1.5,
    X0 + np.exp(-0.2*r2)*bl_noise(x.shape, range(4, 12))*0.2,
    Y0 + np.exp(-0.3*r2)*bl_noise(x.shape)*0.1, dx,
    iterations=100
)

t = 0
then = datetime.datetime.now()
while t < 1.0:
    model_g.step()
    t += model_g.dt * model_g.iterations
    G, X, Y = model_g.numpy()
    print(t, G.min(), G.max(), X.min(), X.max(), Y.min(), Y.max())

print("Time elapsed", datetime.datetime.now() - then)
