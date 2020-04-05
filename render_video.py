from __future__ import division

import argparse
import numpy as np
import tensorflow as tf
import progressbar
import imageio

from fluid_model_g import FluidModelG


RESOLUTIONS = {
    "2160p": (3840, 2160),
    "1440p": (2560, 1440),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
    "360p": (640, 360),
    "240p": (426, 240),
    "160p": (284, 160),
    "80p": (142, 80),
    "40p": (71, 40),
}


def make_video_frame(rgb, indexing='ij'):
    if indexing == 'ij':
        rgb = [tf.transpose(channel) for channel in rgb]
    frame = tf.stack(rgb, axis=-1)
    frame = tf.clip_by_value(frame, 0.0, 1.0)
    return tf.cast(frame * 255, 'uint8').numpy()


def nucleation_and_motion_in_G_gradient_2D(writer, args, R=16):
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

    dx = 2*R / args.height
    x = (np.arange(args.width) - args.width // 2) * dx
    y = (np.arange(args.height) - args.height // 2) * dx
    x, y = np.meshgrid(x, y, indexing='ij')

    def source_G(t):
        center = np.exp(-0.5*(t-5)**2) * 10
        gradient = (1+np.tanh(t-30)) * 0.0003
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
        dt=args.dt,
        params=params,
        source_functions=source_functions,
    )

    print("Rendering 'Nucleation and Motion in G gradient in 2D'")
    print("Lattice constant dx = {}, time step dt = {}".format(fluid_model_g.dx, fluid_model_g.dt))
    min_G = -4.672736908320116
    max_G = 0.028719261862332906
    min_X = -3.8935243721220334
    max_X = 1.2854028081816122
    min_Y = -0.7454193158963579
    max_Y = 4.20524950766914
    for n in progressbar.progressbar(range(args.num_frames)):
        fluid_model_g.step()
        if n % args.oversampling == 0:
            rgb = [
                6*(-fluid_model_g.G + max_G) / (max_G - min_G),
                5*(fluid_model_g.Y - min_Y) / (max_Y - min_Y),
                0.7*(fluid_model_g.X - min_X) / (max_X - min_X),
            ]
            zero_line = 1 - tf.exp(-600 * fluid_model_g.Y**2)
            frame = make_video_frame([c * zero_line for c in rgb])
            writer.append_data(frame)
    #     max_G = max(max_G, tf.reduce_max(fluid_model_g.G).numpy())
    #     min_G = min(min_G, tf.reduce_min(fluid_model_g.G).numpy())
    #     max_X = max(max_X, tf.reduce_max(fluid_model_g.X).numpy())
    #     min_X = min(min_X, tf.reduce_min(fluid_model_g.X).numpy())
    #     max_Y = max(max_Y, tf.reduce_max(fluid_model_g.Y).numpy())
    #     min_Y = min(min_Y, tf.reduce_min(fluid_model_g.Y).numpy())

    # print(min_G, max_G, min_X, max_X, min_Y, max_Y)


if __name__ == '__main__':
    episodes = {
        'nucleation_and_motion': nucleation_and_motion_in_G_gradient_2D,
    }

    parser = argparse.ArgumentParser(description='Render audio samples')
    parser.add_argument('episode', choices=episodes.keys())
    parser.add_argument('outfile', type=str, help='Output file name')
    parser.add_argument('--resolution', choices=RESOLUTIONS.keys(), help='Video and simulation grid resolution', default='240p')
    # parser.add_argument('--width', type=int, help='Video and simulation grid width', metavar='W')
    # parser.add_argument('--height', type=int, help='Video and simulation grid height', metavar='H')
    parser.add_argument('--framerate', type=int, help='Video frame rate', default=24)
    parser.add_argument('--oversampling', type=int, help='Add extra simulation time steps between video frames for stability', default=1)
    parser.add_argument('--video-quality', type=int, help='Video quality factor', default=10)
    parser.add_argument('--video-duration', type=float, help='Duration of video to render in seconds', default=1.0)
    parser.add_argument('--simulation-duration', type=float, help='Amount of simulation to run', default=1.0)
    args = parser.parse_args()

    writer = imageio.get_writer(args.outfile, fps=args.framerate, quality=args.video_quality, macro_block_size=1)

    # Compute derived parameters
    args.width, args.height = RESOLUTIONS[args.resolution]
    args.aspect = args.width / args.height
    args.num_frames = int(args.video_duration * args.oversampling * args.framerate)
    args.dt = args.simulation_duration / args.num_frames

    episodes[args.episode](writer, args)
    writer.close()
