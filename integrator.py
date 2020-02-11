import tensorflow as tf
import numpy as np


class Integrator(object):
    """
    Integrator for partial differential equations
    """

    def __init__(self, dx, dt, dims):
        if dims == 1:
            kernel = [1, -2, 1]
            self.conv = tf.nn.conv1d
        elif dims == 2:
            kernel = [
                0,  1, 0,
                1, -4, 1,
                0,  1, 0,
            ]
            self.conv = tf.nn.conv2d
        elif dims == 3:
            kernel = [
                0, 0, 0,
                0, 1, 0,
                0, 0, 0,

                0,  1, 0,
                1, -6, 1,
                0,  1, 0,

                0, 0, 0,
                0, 1, 0,
                0, 0, 0,
            ]
            self.conv = tf.nn.conv3d
        else:
            raise ValueError("Only up to 3D supported")

        self.dx = dx
        self.dt = dt
        self.kernel = tf.reshape(tf.constant(np.array(kernel) * dx*dx, dtype="float64"), [3]*dims + [1, 1])
        self.strides = [1] * (dims+2)
        self.pads = tf.constant([[0, 0]] + [[1, 1]]*dims + [[0, 0]])
