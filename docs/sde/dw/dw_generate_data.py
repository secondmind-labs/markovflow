"""
Script to generate data for Double-Well and save as npz files that can be used later for inference and learning.
"""
import os
import argparse

import tensorflow as tf
import numpy as np
from gpflow.config import default_float
import matplotlib.pyplot as plt

from docs.sde.sde_exp_utils import generate_dw_data, plot_observations

DTYPE = default_float()


def set_seed(seed: int):
    """Set seed values"""
    tf.random.set_seed(seed)
    np.random.seed(seed)


def create_output_dir(output_dir: str, seed: int):
    """Create the output directory"""
    output_dir = os.path.join(output_dir, str(seed))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print("Data already exists. Overwriting!!!")

    return output_dir


def generate_data(q: float, x0: float, t0: float, t1: float, noise_var: float,
                  dt: float, n_observations: int) -> (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):
    """Generate data for the OU SDE using the parameters"""

    noise_stddev = np.sqrt(noise_var)

    observation_vals, observation_grid, \
    latent_process, time_grid, test_values, test_grid, _ = generate_dw_data(q=q, x0=x0, t0=t0, t1=t1,
                                                                            simulation_dt=dt,
                                                                            noise_stddev=noise_stddev,
                                                                            n_observations=n_observations,
                                                                            dtype=DTYPE)

    return observation_vals, observation_grid, latent_process, time_grid, test_values, test_grid


def plot_data(o_grid: tf.Tensor, o_vals: tf.Tensor, time_grid: tf.Tensor, latent_process: tf.Tensor,
              t0: float, t1: float, output_dir: str, test_values: tf.Tensor, test_grid: tf.Tensor):
    """Plot the generated data"""
    plt.rcParams["figure.figsize"] = [15, 5]
    plot_observations(o_grid.numpy(), o_vals.numpy())
    plt.plot(time_grid, tf.reshape(latent_process, (-1)), label="Latent Process", alpha=0.2, color="gray")
    plt.plot(test_grid.numpy().reshape(-1), test_values.numpy().reshape(-1), 'x', color="red", ms=8, mew=2,
             label="Test Observations (Y)")
    plt.xlabel("Time (t)")
    plt.ylabel("y(t)")
    plt.xlim([t0, t1])
    plt.title("Observations")
    plt.legend()

    plt.savefig(os.path.join(output_dir, "data.svg"))
    plt.show()


if __name__ == '__main__':
    # python dw_generate_data.py -s 128 -q 1.2 -n 10 -v 0.5 -t0 0. -t1 5. -dt 0.01 -x0 1.

    parser = argparse.ArgumentParser(description='Generate data for OU process')

    parser.add_argument('-s', '--seed', type=int, help='Seed value to be used when generating the data.', required=True)
    parser.add_argument('-q', type=float, help="Spectral density of the Brownian motion", required=True)
    parser.add_argument('-n', '--n-observations', type=int,
                        help="Number of observations", required=True)

    parser.add_argument('-v', '--noise-variance', type=float, default=0.1, help="Noise variance")
    parser.add_argument('-t0', type=float, default=0., help="t0 value")
    parser.add_argument('-t1', type=float, default=1., help="t1 value")
    parser.add_argument('-dt', type=float, default=.01, help="time-step value for simulation")
    parser.add_argument('-x0', type=float, default=0., help="x0 value")
    parser.add_argument('-dir', type=str, default="data/", help="Directory path to save the output")

    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = create_output_dir(args.dir, args.seed)

    observation_vals, observation_grid, \
    latent_process, time_grid, test_values, test_grid = generate_data(args.q, args.x0, args.t0,
                                                                      args.t1, args.noise_variance,
                                                                      args.dt, args.n_observations)

    plot_data(observation_grid, observation_vals, time_grid, latent_process, args.t0, args.t1, output_dir, test_values,
              test_grid)

    np.savez(os.path.join(output_dir, "data.npz"), observation_grid=observation_grid,
             observation_vals=observation_vals, latent_process=latent_process, test_grid=test_grid,
             test_vals=test_values, time_grid=time_grid, q=args.q, noise_stddev=np.sqrt(args.noise_variance),
             x0=args.x0)
