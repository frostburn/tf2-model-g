import tensorflow as tf
import numpy as np
import util
from integrators.brusselerator import polynomial_order_4_centered as reaction_integrator


DEFAULT_PARAMS = {
    "A": 1.0,
    "B": 3.0,
    "k4": 1.0,
    "Dx": 0.75,
    "Dy": 3.0,
    "density_X": 1.0,
    "density_Y": 1.5,
    "base-density": 5.0,
    "viscosity": 0.1,
    "speed-of-sound": 0.2,
}


class FluidBrusselerator(object):
    """
    Brusselerator on a fluid medium
    """
    def __init__(self, concentration_X, concentration_Y, u, dx, params=None):
        self.dx = dx
        self.dt = 0.01 * dx

        self.params = params or DEFAULT_PARAMS

        self.X = tf.constant(concentration_X, 'float64')
        self.Y = tf.constant(concentration_Y, 'float64')

        l = self.X.shape[-1]
        if any(s != l for s in self.X.shape):
            raise ValueError('Only square grids supported')

        dims = len(self.X.shape)
        if len(u) != dims:
            raise ValueError("{0}-dimensional flow must have {0} components".format(dims))
        self.dims = dims

        ratio = 2*np.pi / (l*self.dx)
        delta = self.dt * ratio**2
        omega = np.arange(l)
        omega -= l * (2*omega > l)
        c2 = self.params["speed-of-sound"]
        viscosity = self.params["viscosity"]
        if dims == 1:
            raise ValueError("1D not supported")
        elif dims == 2:
            self.u = tf.constant(u[0], 'float64')
            self.v = tf.constant(u[1], 'float64')

            self.fft = tf.signal.fft2d
            self.ifft = tf.signal.ifft2d

            omega_x, omega_y = np.meshgrid(omega, omega)
            omega2 = omega_x**2 + omega_y**2
            omega2_x = tf.constant(omega2 + 1/3 * omega_x * (omega_x + omega_y), "complex128")
            omega2_y = tf.constant(omega2 + 1/3 * omega_y * (omega_x + omega_y), "complex128")
            decay_x = tf.exp(-viscosity * delta * omega2_x)
            decay_y = tf.exp(-viscosity * delta * omega2_y)

            omega_dx = tf.constant(1j * omega_x, 'complex128')
            omega_dy = tf.constant(1j * omega_y, 'complex128')

            decay_X = tf.exp(-self.params['Dx'] * delta * tf.constant(omega2, 'complex128'))
            decay_Y = tf.exp(-self.params['Dy'] * delta * tf.constant(omega2, 'complex128'))

            def flow_integrator(rho, u, v):
                """
                Flow is integrated with respect to the total log density (rho)
                """
                # Enter Fourier Domain
                f_rho = self.fft(tf.cast(rho, 'complex128'))
                waves_x = self.fft(tf.cast(u, 'complex128'))
                waves_y = self.fft(tf.cast(v, 'complex128'))

                # Viscosity and internal shear
                waves_x *= decay_x
                waves_y *= decay_y

                # Exit Fourier Domain
                u = tf.cast(self.ifft(waves_x), 'float64')
                v = tf.cast(self.ifft(waves_y), 'float64')

                # Calculate gradients
                rho_dx = tf.cast(self.ifft(f_rho * omega_dx), 'float64')
                rho_dy = tf.cast(self.ifft(f_rho * omega_dy), 'float64')
                u_dx = tf.cast(self.ifft(waves_x * omega_dx), 'float64')
                u_dy = tf.cast(self.ifft(waves_x * omega_dy), 'float64')
                v_dx = tf.cast(self.ifft(waves_y * omega_dx), 'float64')
                v_dy = tf.cast(self.ifft(waves_y * omega_dy), 'float64')
                divergence = u_dx + v_dy

                # This would handle log density continuity but it's actually handled individually for X and Y
                # rho -= (u*rho_dx + v*rho_dy + divergence) * self.dt

                # Self-advect flow
                du = -u*u_dx - v*u_dy
                dv = -u*v_dx - v*v_dy

                # Propagate pressure waves
                du -= c2 * rho_dx
                dv -= c2 * rho_dy

                # Apply strain
                du += viscosity * (rho_dx * (u_dx + u_dx) + rho_dy * (u_dy + v_dx) - 2/3*rho_dx * divergence)
                dv += viscosity * (rho_dx * (v_dx + u_dy) + rho_dy * (v_dy + v_dy) - 2/3*rho_dy * divergence)

                u += du * self.dt
                v += dv * self.dt

                return u, v, divergence

            def diffusion_advection_integrator(X, Y, u, v, divergence):
                f_X = self.fft(tf.cast(X, 'complex128'))
                f_Y = self.fft(tf.cast(Y, 'complex128'))

                f_X *= decay_X
                f_Y *= decay_Y

                X = tf.cast(self.ifft(f_X), 'float64')
                Y = tf.cast(self.ifft(f_Y), 'float64')

                X_dx = tf.cast(self.ifft(f_X * omega_dx), 'float64')
                X_dy = tf.cast(self.ifft(f_X * omega_dy), 'float64')
                Y_dx = tf.cast(self.ifft(f_Y * omega_dx), 'float64')
                Y_dy = tf.cast(self.ifft(f_Y * omega_dy), 'float64')

                X -= (u*X_dx + v*X_dy + X*divergence) * self.dt
                Y -= (u*Y_dx + v*Y_dy + Y*divergence) * self.dt
                return X, Y
        elif dims == 3:
            self.u = tf.constant(u[0], 'float64')
            self.v = tf.constant(u[1], 'float64')
            self.w = tf.constant(u[2], 'float64')

            self.fft = tf.signal.fft3d
            self.ifft = tf.signal.ifft3d

            omega_x, omega_y, omega_z = np.meshgrid(omega, omega, omega)
            omega2 = omega_x**2 + omega_y**2 + omega_z**2
            omega2_x = tf.constant(omega2 + 1/3 * omega_x * (omega_x + omega_y + omega_z), "complex128")
            omega2_y = tf.constant(omega2 + 1/3 * omega_y * (omega_x + omega_y + omega_z), "complex128")
            omega2_z = tf.constant(omega2 + 1/3 * omega_z * (omega_x + omega_y + omega_z), "complex128")
            decay_x = tf.exp(-viscosity * delta * omega2_x)
            decay_y = tf.exp(-viscosity * delta * omega2_y)
            decay_z = tf.exp(-viscosity * delta * omega2_z)

            omega_dx = tf.constant(1j * omega_x, 'complex128')
            omega_dy = tf.constant(1j * omega_y, 'complex128')
            omega_dz = tf.constant(1j * omega_z, 'complex128')

            decay_X = tf.exp(-self.params['Dx'] * delta * tf.constant(omega2, 'complex128'))
            decay_Y = tf.exp(-self.params['Dy'] * delta * tf.constant(omega2, 'complex128'))

            def flow_integrator(rho, u, v, w):
                # Enter Fourier Domain
                f_rho = self.fft(tf.cast(rho, 'complex128'))
                waves_x = self.fft(tf.cast(u, 'complex128'))
                waves_y = self.fft(tf.cast(v, 'complex128'))
                waves_z = self.fft(tf.cast(w, 'complex128'))

                # Viscosity and internal shear
                waves_x *= decay_x
                waves_y *= decay_y
                waves_z *= decay_z

                # Exit Fourier Domain
                u = tf.cast(self.ifft(waves_x), 'float64')
                v = tf.cast(self.ifft(waves_y), 'float64')
                w = tf.cast(self.ifft(waves_z), 'float64')

                # Calculate gradients
                rho_dx = tf.cast(self.ifft(f_rho * omega_dx), 'float64')
                rho_dy = tf.cast(self.ifft(f_rho * omega_dy), 'float64')
                rho_dz = tf.cast(self.ifft(f_rho * omega_dz), 'float64')

                u_dx = tf.cast(self.ifft(waves_x * omega_dx), 'float64')
                u_dy = tf.cast(self.ifft(waves_x * omega_dy), 'float64')
                u_dz = tf.cast(self.ifft(waves_x * omega_dz), 'float64')

                v_dx = tf.cast(self.ifft(waves_y * omega_dx), 'float64')
                v_dy = tf.cast(self.ifft(waves_y * omega_dy), 'float64')
                v_dz = tf.cast(self.ifft(waves_y * omega_dz), 'float64')

                w_dx = tf.cast(self.ifft(waves_z * omega_dx), 'float64')
                w_dy = tf.cast(self.ifft(waves_z * omega_dy), 'float64')
                w_dz = tf.cast(self.ifft(waves_z * omega_dz), 'float64')

                divergence = u_dx + v_dy + w_dz

                # This would handle log density continuity, but we do X and Y individually
                # rho -= (u*rho_dx + v*rho_dy + w*rho_dz + divergence) * self.dt

                # Self-advect flow
                du = -u*u_dx - v*u_dy - w*u_dz
                dv = -u*v_dx - v*v_dy - w*v_dz
                dw = -u*w_dx - v*w_dy - w*w_dz

                # Propagate pressure waves
                du -= c2 * rho_dx
                dv -= c2 * rho_dy
                dw -= c2 * rho_dz

                # Apply strain
                du += viscosity * (rho_dx * (u_dx + u_dx) + rho_dy * (u_dy + v_dx) + rho_dz * (u_dz + w_dx) - 2/3*rho_dx * divergence)
                dv += viscosity * (rho_dx * (v_dx + u_dy) + rho_dy * (v_dy + v_dy) + rho_dz * (v_dz + w_dy) - 2/3*rho_dy * divergence)
                dw += viscosity * (rho_dx * (w_dx + u_dz) + rho_dy * (w_dy + v_dz) + rho_dz * (w_dz + w_dz) - 2/3*rho_dz * divergence)

                u += du * self.dt
                v += dv * self.dt
                w += dw * self.dt

                return u, v, w, divergence
            def diffusion_advection_integrator(X, Y, u, v, w, divergence):
                f_X = self.fft(tf.cast(X, 'complex128'))
                f_Y = self.fft(tf.cast(Y, 'complex128'))

                f_X *= decay_X
                f_Y *= decay_Y

                X = tf.cast(self.ifft(f_X), 'float64')
                Y = tf.cast(self.ifft(f_Y), 'float64')

                X_dx = tf.cast(self.ifft(f_X * omega_dx), 'float64')
                X_dy = tf.cast(self.ifft(f_X * omega_dy), 'float64')
                X_dz = tf.cast(self.ifft(f_X * omega_dz), 'float64')
                Y_dx = tf.cast(self.ifft(f_Y * omega_dx), 'float64')
                Y_dy = tf.cast(self.ifft(f_Y * omega_dy), 'float64')
                Y_dz = tf.cast(self.ifft(f_Y * omega_dz), 'float64')

                X -= (u*X_dx + v*X_dy + w*X_dz + X*divergence) * self.dt
                Y -= (u*Y_dx + v*Y_dy + w*Y_dz + Y*divergence) * self.dt
                return X, Y
        else:
            raise ValueError('Only up to 3D supported')

        reaction_integrator_curried = lambda con_X, con_Y: reaction_integrator(con_X, con_Y, self.dt, self.params['A'], self.params['B'], self.params['k4'])

        self.reaction_integrator = tf.function(reaction_integrator_curried)
        self.flow_integrator = tf.function(flow_integrator)
        self.diffusion_advection_integrator = tf.function(diffusion_advection_integrator)

    def step(self):
        self.X, self.Y = self.reaction_integrator(self.X, self.Y)
        density_of_reactants = self.params['density_X'] * self.X + self.params['density_Y'] * self.Y
        rho = tf.math.log(self.params['base-density'] + density_of_reactants)
        if self.dims == 2:
            u, v = self.u, self.v  # Store unintegrated flow so that we're on the same timestep
            self.u, self.v, divergence = self.flow_integrator(rho, self.u, self.v)
            self.X, self.Y = self.diffusion_advection_integrator(self.X, self.Y, u, v, divergence)
        elif self.dims == 3:
            u, v, w = self.u, self.v, self.w  # Store unintegrated flow so that we're on the same timestep
            self.u, self.v, self.w, divergence = self.flow_integrator(rho, self.u, self.v, self.w)
            self.X, self.Y = self.diffusion_advection_integrator(self.X, self.Y, u, v, w, divergence)

    def numpy(self):
        if self.dims == 2:
            u = (self.u.numpy(), self.v.numpy())
        elif self.dims == 3:
            u = (self.u.numpy(), self.v.numpy(), self.w.numpy())
        return self.X.numpy(), self.Y.numpy(), u


if __name__ == '__main__':
    from util import bl_noise
    import pylab
    from matplotlib.animation import FuncAnimation

    N = 64
    x = np.linspace(-4, 4, N)
    dx = x[1] - x[0]

    if '3D':
        x, y, z = np.meshgrid(x, x, x)
        X = bl_noise(x.shape)*0.06
        Y = bl_noise(x.shape)*0.06

        flow = [
            np.exp(-(x)**2-(y-1)**2-z**2)*0.005,
            np.exp(-(x-2)**2-y**2-z**2)*0.005,
            np.exp(-x**2-y**2-z**2)*x*y*0.001
        ]
        fluid_brusselerator = FluidBrusselerator(X, Y, flow, dx)

        X, Y, (u, v, w) = fluid_brusselerator.numpy()
        plots = [pylab.imshow(Y[::-1,:, 0], extent=(-4,4,-4,4) , vmin=-1.2, vmax=1.2, cmap="cividis")]
        plots.append(pylab.quiver(x[::4,::4, 0], y[::4,::4, 0], u[::4,::4, 0] + 0.002, v[::4,::4, 0]+0.002))

        def update(frame):
            fluid_brusselerator.step()
            X, Y, (u, v, w) = fluid_brusselerator.numpy()
            print(frame, abs(X).max(), abs(Y).max(), abs(u).max())
            plots[0].set_data(Y[::-1,:, frame])
            plots[1].set_UVC(u[::4,::4, frame], v[::4,::4, frame])
            return plots
    else:
        x, y = np.meshgrid(x, x)
        X = np.exp(-x*x - 0.3*y*y) * (1 + 0.05*bl_noise(x.shape))
        Y = 0*x

        flow = [
            np.exp(-(x)**2-(y-1)**2)*0.5 - 0.1,
            0.1 - np.exp(-(x-2)**2-y**2)*0.5,
        ]
        fluid_brusselerator = FluidBrusselerator(X, Y, flow, dx)

        X, Y, (u, v) = fluid_brusselerator.numpy()
        plots = [pylab.imshow(Y[::-1,:], extent=(-4,4,-4,4) , vmin=-1.0, vmax=1.0, cmap="cividis")]
        plots.append(pylab.quiver(x[::4,::4], y[::4,::4], u[::4,::4], v[::4,::4]))

        def update(frame):
            for _ in range(3):
                fluid_brusselerator.step()
            X, Y, (u, v) = fluid_brusselerator.numpy()
            plots[0].set_data(Y[::-1,:])
            plots[1].set_UVC(u[::4,::4], v[::4,::4])
            return plots

    FuncAnimation(pylab.gcf(), update, frames=range(N), init_func=lambda: plots, blit=True, repeat=True, interval=50)
    pylab.show()