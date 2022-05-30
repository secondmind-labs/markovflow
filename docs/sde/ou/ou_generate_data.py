"""
Script to generate data for OU and save as npz files that can be used later for inference and learning.
"""
import os
import tensorflow as tf
import numpy as np
from gpflow.config import default_float
import matplotlib.pyplot as plt

from docs.sde.sde_exp_utils import generate_ou_data, plot_observations

seed = 119
tf.random.set_seed(seed)
np.random.seed(seed)
DTYPE = default_float()

"""
Parameters
"""
decay = .5  # specify without the negative sign
q = 1.2
noise_var = 0.05
x0 = 5.

t0, t1 = 0., 40.
simulation_dt = 0.01  # Used for Euler-Maruyama
n_observations = 20

output_dir = "data/"

output_dir = os.path.join(output_dir, str(seed))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    print("Data already exists. Overwriting!!!")

"""
Generate observations
"""
noise_stddev = np.sqrt(noise_var)

observation_vals, observation_grid, latent_process, time_grid = generate_ou_data(decay=decay, q=q, x0=x0, t0=t0, t1=t1,
                                                                                 simulation_dt=simulation_dt,
                                                                                 noise_stddev=noise_stddev,
                                                                                 n_observations=n_observations,
                                                                                 dtype=DTYPE)

plt.rcParams["figure.figsize"] = [15, 5]
plot_observations(observation_grid.numpy(), observation_vals.numpy())
plt.plot(time_grid, tf.reshape(latent_process, (-1)), label="Latent Process", alpha=0.2, color="gray")
plt.xlabel("Time (t)")
plt.ylabel("y(t)")
# plt.ylim([-2, 5])
plt.xlim([t0, t1])
plt.title("Observations")
plt.legend()

plt.savefig(os.path.join(output_dir, "data.svg"))
np.savez(os.path.join(output_dir, "data.npz"), observation_grid=observation_grid,
         observation_vals=observation_vals, latent_process=latent_process,
         time_grid=time_grid, q=q, noise_stddev=noise_stddev, decay=decay, x0=x0)

plt.show()
