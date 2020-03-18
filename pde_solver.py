import tensorflow as tf
import numpy as np


class PDESolver(object):
    """
    Base class for partial differential equation solvers
    """

    def __init__(self, dx, dt, shape):
        self.dx = dx
        self.dt = dt
        self.t = 0

        omega = []
        for s in shape:
            wave_numbers = np.arange(s)
            wave_numbers -= s * (2*wave_numbers > s)  # Deal with TensorFlow's uncentered FFT
            expected_span = 2*np.pi
            actual_span = s*dx
            omega.append(wave_numbers * expected_span / actual_span)
        self.omega = np.meshgrid(*omega, indexing='ij')
        self.dims = len(shape)
        # The naming is a bit off. These are not actual 'kernels'.
        # They are discrete fourier transforms of the periodic versions of the kernels
        if self.dims == 1:
            self.fft = tf.signal.fft
            self.ifft = tf.signal.ifft
            self.omega_x = self.omega[0]
            self.kernel_dx = tf.constant(1j * self.omega_x, 'complex128')
            self.kernel_laplacian = tf.constant(-self.omega_x**2, 'complex128')
        elif self.dims == 2:
            self.fft = tf.signal.fft2d
            self.ifft = tf.signal.ifft2d

            self.omega_x = self.omega[0]
            self.omega_y = self.omega[1]
            self.kernel_dx = tf.constant(1j * self.omega_x, 'complex128')
            self.kernel_dy = tf.constant(1j * self.omega_y, 'complex128')
            self.kernel_laplacian = tf.constant(-(self.omega_x**2 + self.omega_y**2), 'complex128')
        elif self.dims == 3:
            self.fft = tf.signal.fft3d
            self.ifft = tf.signal.ifft3d

            self.omega_x = self.omega[0]
            self.omega_y = self.omega[1]
            self.omega_z = self.omega[2]
            self.kernel_dx = tf.constant(1j * self.omega_x, 'complex128')
            self.kernel_dy = tf.constant(1j * self.omega_y, 'complex128')
            self.kernel_dz = tf.constant(1j * self.omega_z, 'complex128')
            self.kernel_laplacian = tf.constant(-(self.omega_x**2 + self.omega_y**2 + self.omega_z**2), 'complex128')
        else:
            raise ValueError('{} dimensions not supported'.format(self.dims))


if __name__ == '__main__':
    import pylab

    if '1D':
        x = np.linspace(-4, 4, 256)
        dx = x[1] - x[0]
        solver = PDESolver(dx, dx*0.1, x.shape)

        test_function = np.exp(-x*x) * (2 + np.sin(3*x))

        test_function_dx = -2*x*np.exp(-x*x) * (2 + np.sin(3*x)) + np.exp(-x*x) * (3 * np.cos(3*x))
        test_function_dx2 = 4*x**2*(np.sin(3*x) + 2)*np.exp(-x**2) - 12*x*np.cos(3*x)*np.exp(-x**2) - 2*(np.sin(3*x) +
    2)*np.exp(-x**2) - 9*np.exp(-x**2)*np.sin(3*x)

        test_function_dx_discrete = np.diff(test_function) / dx
        test_function_dx2_discrete = np.diff(test_function_dx_discrete) / dx

        f_test_function = solver.fft(tf.constant(test_function, 'complex128'))
        test_function_dx_fft = np.real(solver.ifft(f_test_function * solver.kernel_dx).numpy())
        test_function_dx2_fft = np.real(solver.ifft(f_test_function * solver.kernel_laplacian).numpy())

        pylab.plot(x, test_function)
        pylab.plot(x, test_function_dx)
        pylab.plot(x[:-1] + 0.5*dx, test_function_dx_discrete+0.1)
        pylab.plot(x, test_function_dx_fft+0.2)

        pylab.plot(x, test_function_dx2)
        pylab.plot(x[1:-1], test_function_dx2_discrete+0.1)
        pylab.plot(x, test_function_dx2_fft+0.2)
        pylab.show()
    if '2D':
        x = np.linspace(-4, 4, 256)
        dx = x[1] - x[0]
        y = (np.arange(200) - 100) * dx
        x, y = np.meshgrid(x, y, indexing='ij')

        solver = PDESolver(dx, dx*0.1, x.shape)

        test_function = np.exp(-x*x - 2*y*y)*x
        test_function_dx = np.exp(-x*x - 2*y*y) * (1 - 2*x**2)
        test_function_dy = -4*x*y * np.exp(-x*x - 2*y*y)
        test_function_nabla2 = 4*x**3*np.exp(-x**2 - 2*y**2) + 16*x*y**2*np.exp(-x**2 - 2*y**2) - 10*x*np.exp(-x**2 -
    2*y**2)

        f_test_function = solver.fft(tf.constant(test_function, 'complex128'))
        test_function_dx_fft = np.real(solver.ifft(f_test_function * solver.kernel_dx).numpy())
        test_function_dy_fft = np.real(solver.ifft(f_test_function * solver.kernel_dy).numpy())
        test_function_nabla2_fft = np.real(solver.ifft(f_test_function * solver.kernel_laplacian).numpy())

        pylab.plot(x[:,90], test_function_dx[:,90])
        pylab.plot(x[:,90], test_function_dx_fft[:,90]+0.02)
        pylab.plot(x[:,110], test_function_dy[:,110])
        pylab.plot(x[:,110], test_function_dy_fft[:,110]+0.02)
        pylab.plot(y[80], test_function_nabla2[80])
        pylab.plot(y[80], test_function_nabla2_fft[80]+0.02)
        pylab.show()

        pylab.imshow(test_function_dx_fft - test_function_dx)
        pylab.show()
        print(abs(test_function_dx_fft - test_function_dx).max())
        pylab.imshow(test_function_dy_fft - test_function_dy)
        pylab.show()
        print(abs(test_function_dy_fft - test_function_dy).max())
        pylab.imshow(test_function_nabla2_fft - test_function_nabla2)
        pylab.show()
        print(abs(test_function_nabla2_fft - test_function_nabla2).max())
    if '3D':
        x = np.linspace(-4, 4, 256)
        dx = x[1] - x[0]
        y = (np.arange(100) - 50) * dx
        z = (np.arange(128) - 64) * dx

        x, y, z = np.meshgrid(x, y, z, indexing='ij')

        solver = PDESolver(dx, dx*0.1, x.shape)

        test_function = np.exp(-x*x - 3*y*y - 2*z*z)
        test_function_dx = -2*x*np.exp(-x*x - 3*y*y - 2*z*z)
        test_function_dy = -6*y*np.exp(-x*x - 3*y*y - 2*z*z)
        test_function_dz = -4*z*np.exp(-x*x - 3*y*y - 2*z*z)
        test_function_nabla2 = (4*x**2 + 36*y**2 + 16*z**2 - 12)*np.exp(-x**2 - 3*y**2 - 2*z**2)

        f_test_function = solver.fft(tf.constant(test_function, 'complex128'))
        test_function_dx_fft = np.real(solver.ifft(f_test_function * solver.kernel_dx).numpy())
        test_function_dy_fft = np.real(solver.ifft(f_test_function * solver.kernel_dy).numpy())
        test_function_dz_fft = np.real(solver.ifft(f_test_function * solver.kernel_dz).numpy())
        test_function_nabla2_fft = np.real(solver.ifft(f_test_function * solver.kernel_laplacian).numpy())


        pylab.plot(z[50,60], test_function_dz_fft[50, 60])
        pylab.plot(z[50,60], test_function_dz[50, 60]+0.02)
        pylab.plot(y[50,:,30], test_function_dz_fft[50,:,30])
        pylab.plot(y[50,:,30], test_function_dz[50,:,30]+0.02)
        pylab.plot(x[:,70,50], test_function_dz_fft[:,70,50])
        pylab.plot(x[:,70,50], test_function_dz[:,70,50]+0.02)
        pylab.show()

        pylab.imshow(test_function_dx[50]-test_function_dx_fft[50])
        pylab.show()
        pylab.plot(test_function_dx[50,50]-test_function_dx_fft[50,50])
        pylab.show()
        print(abs(test_function_dx-test_function_dx_fft).max())

        pylab.imshow(test_function_dy[:,50]-test_function_dy_fft[:,50])
        pylab.show()
        print(abs(test_function_dy-test_function_dy_fft).max())

        pylab.imshow(test_function_dz[:,:,60]-test_function_dz_fft[:,:,60])
        pylab.show()
        pylab.plot(test_function_dz[:,60,60]-test_function_dz_fft[:,60,60])
        pylab.show()
        print(abs(test_function_dz-test_function_dz_fft).max())

        pylab.imshow(test_function_nabla2[40]-test_function_nabla2_fft[40])
        pylab.show()
        pylab.plot(test_function_nabla2[40,40]-test_function_nabla2_fft[40,40])
        pylab.show()
        print(abs(test_function_nabla2-test_function_nabla2_fft).max())
