"""
Script to compare the output of SDE-SSM and GPR Taylor.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import wandb
import tensorflow as tf
from gpflow import default_float
from gpflow.likelihoods import Gaussian

from markovflow.kernels import OrnsteinUhlenbeck
from markovflow.models.variational_cvi import CVIGaussianProcessSitesGrid


os.environ['WANDB_MODE'] = 'offline'
"""Logging init"""
wandb.init(project="VI-SDE", entity="vermaprakhar")

DTYPE = default_float()
plt.rcParams["figure.figsize"] = [15, 5]

"""
Parameters
"""
data_dir = "../data/38"

"""
Get Data
"""
data_path = os.path.join(data_dir, "data.npz")
data = np.load(data_path)
decay = data["decay"]
q = data["q"]
noise_stddev = data["noise_stddev"]
x0 = data["x0"]
observation_vals = data["observation_vals"]
observation_grid = data["observation_grid"]
latent_process = data["latent_process"]
time_grid = data["time_grid"]
t0 = time_grid[0]
t1 = time_grid[-1]

plt.scatter(observation_grid, observation_vals)
plt.plot(time_grid, tf.reshape(latent_process, (-1)), label="Latent Process", alpha=0.2, color="gray")
plt.xlabel("Time (t)")
plt.ylabel("y(t)")
plt.xlim([t0, t1])
plt.title("Observations")
plt.legend()
plt.show()

print(f"True decay value of the OU SDE is {decay}")
print(f"Noise std-dev is {noise_stddev}")

input_data = (observation_grid, tf.constant(tf.squeeze(observation_vals, axis=0)))

dt = 0.0001
time_grid = tf.cast(np.arange(t0, t1 + dt, dt), dtype=DTYPE).numpy()

"""
GPR - Taylor
"""
likelihood = Gaussian(noise_stddev**2)

kernel = OrnsteinUhlenbeck(decay=decay, diffusion=q)

cvi_model = CVIGaussianProcessSitesGrid(input_data=input_data, kernel=kernel, likelihood=likelihood,
                                        time_grid=time_grid, learning_rate=0.9)
for _ in range(5):
    cvi_model.update_sites()

print(f"CVI-GPR ELBO: {cvi_model.classic_elbo()}")

posterior_ssm = cvi_model.dist_q

process_covariance = tf.square(posterior_ssm.cholesky_process_covariances).numpy().reshape((-1, 1))

expected_q = q * (time_grid[1] - time_grid[0])
expected_q = np.repeat(expected_q, process_covariance.shape[0], axis=0).reshape((-1, 1))

np.testing.assert_array_almost_equal(expected_q, process_covariance)
