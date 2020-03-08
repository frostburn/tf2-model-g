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

            # TODO: Smooth out gradients for numerical stability
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

                # Handle log density continuity
                rho -= (u*rho_dx + v*rho_dy + u_dx + v_dy) * self.dt

                # Self-advect flow
                du = -u*u_dx - v*u_dy
                dv = -u*v_dx - v*v_dy

                # Propagate pressure waves
                du -= c2 * rho_dx
                dv -= c2 * rho_dy

                # Apply strain
                du += viscosity * (4/3*rho_dx * u_dx + rho_dy * (v_dx + u_dy))
                dv += viscosity * (4/3*rho_dy * v_dy + rho_dx * (v_dx + u_dy))

                u += du * self.dt
                v += dv * self.dt

                return rho, u, v
        elif dims == 3:
            # TODO
            self.fft = tf.signal.fft3d
            self.ifft = tf.signal.ifft3d
        else:
            raise ValueError('Only up to 3D supported')


        self.integrator = tf.function(integrator)

    def step(self):
        self.rho, self.u, self.v = self.integrator(self.rho, self.u, self.v)

    def numpy(self):
        return np.exp(self.rho.numpy()), (self.u.numpy(), self.v.numpy())


if __name__ == '__main__':
    from util import bl_noise
    import pylab
    from matplotlib.animation import FuncAnimation

    x = np.linspace(-4, 4, 128)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x)
    rho = np.exp(-x*x - 0.3*y*y) * (1 + 0.05*bl_noise(x.shape)) + 2

    fluid = CompressibleFluid(rho, [np.exp(-(x)**2-(y-1)**2)*0.1 - 0.02, 0.02 - np.exp(-(x-2)**2-y**2)*0.1], dx, viscosity=0.02, speed_of_sound=0.2)

    rho, (u, v) = fluid.numpy()
    plots = [pylab.imshow(rho[::-1,:], extent=(-4,4,-4,4) , vmin=2, vmax=2.5, cmap="cividis")]
    plots.append(pylab.quiver(x[::4,::4], y[::4,::4], u[::4,::4], v[::4,::4]))

    def update(frame):
        for _ in range(3):
            fluid.step()
        rho, (u, v) = fluid.numpy()
        plots[0].set_data(rho[::-1,:])
        plots[1].set_UVC(u[::4,::4], v[::4,::4])
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()
