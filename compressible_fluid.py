import tensorflow as tf
import numpy as np
import util


class CompressibleFluid(object):
    """
    Simple integrator to test compressible fluid dynamics
    """
    def __init__(self, rho, u, dx, viscosity=0.1, speed_of_sound=0.1):
        self.dx = dx
        self.dt = 0.5 * dx

        # Work with log density for numerical stability
        self.rho = tf.constant(np.log(rho), 'float64')

        l = rho.shape[-1]
        if any(s != l for s in rho.shape):
            raise ValueError('Only square grids supported')

        dims = len(rho.shape)
        if len(u) != dims:
            raise ValueError("{0}-dimensional flow must have {0} components".format(dims))
        self.dims = dims

        ratio = 2*np.pi / (l*self.dx)
        delta = self.dt * ratio**2
        omega = np.arange(l)
        omega -= l * (2*omega > l)
        c2 = speed_of_sound**2
        if dims == 1:
            self.fft = tf.signal.fft
            self.ifft = tf.signal.ifft
            raise NotImplementedError("1D fluid equation is basically the heat equation, right?")
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

            def integrator(rho, u, v):
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

                # Handle log density continuity
                rho -= (u*rho_dx + v*rho_dy + divergence) * self.dt

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

                return rho, u, v
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

            def integrator(rho, u, v, w):
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

                # Handle log density continuity
                rho -= (u*rho_dx + v*rho_dy + w*rho_dz + divergence) * self.dt

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

                return rho, u, v, w
        else:
            raise ValueError('Only up to 3D supported')


        self.integrator = tf.function(integrator)

    def step(self):
        if self.dims == 2:
            self.rho, self.u, self.v = self.integrator(self.rho, self.u, self.v)
        elif self.dims == 3:
            self.rho, self.u, self.v, self.w = self.integrator(self.rho, self.u, self.v, self.w)

    def numpy(self):
        if self.dims == 2:
            u = (self.u.numpy(), self.v.numpy())
        elif self.dims == 3:
            u = (self.u.numpy(), self.v.numpy(), self.w.numpy())
        return np.exp(self.rho.numpy()), u


if __name__ == '__main__':
    from util import bl_noise
    import pylab
    from matplotlib.animation import FuncAnimation

    N = 128
    x = np.linspace(-4, 4, N)
    dx = x[1] - x[0]
    x, y, z = np.meshgrid(x, x, x)
    rho = np.exp(-x*x - 0.3*y*y - 0.5*z*z) * (1 + 0.05*bl_noise(x.shape)) + 2

    flow = [
        np.exp(-(x)**2-(y-1)**2)*0.1 - 0.02,
        0.02 - np.exp(-(x-2)**2-y**2)*0.1,
        np.exp(-x*x-y*y-z*z)*0.1-0.02,
    ]
    fluid = CompressibleFluid(rho, flow, dx, viscosity=0.02, speed_of_sound=0.2)

    rho, (u, v, w) = fluid.numpy()
    plots = [pylab.imshow(rho[::-1,:, N//2], extent=(-4,4,-4,4) , vmin=2, vmax=2.5, cmap="cividis")]
    plots.append(pylab.quiver(x[::4,::4, N//2], y[::4,::4, N//2], u[::4,::4, N//2], v[::4,::4, N//2]))

    def update(frame):
        fluid.step()
        rho, (u, v, w) = fluid.numpy()
        plots[0].set_data(rho[::-1,:, N//2])
        plots[1].set_UVC(u[::4,::4, N//2], v[::4,::4, N//2])
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()
