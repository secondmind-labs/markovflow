import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow
from gpflow import default_float
from gpflow.likelihoods import Gaussian
from gpflow.utilities.bijectors import positive

from markovflow.kernels import OrnsteinUhlenbeck
from markovflow.models.variational_cvi import CVIGaussianProcessTaylorKernel

import sys
sys.path.append("..")
from sde_exp_utils import plot_observations

DTYPE = default_float()
plt.rcParams["figure.figsize"] = [15, 5]

"""
Parameters
"""
data_dir = "data/36"

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

plot_observations(observation_grid, observation_vals)
plt.plot(time_grid, tf.reshape(latent_process, (-1)), label="Latent Process", alpha=0.2, color="gray")
plt.xlabel("Time (t)")
plt.ylabel("y(t)")
plt.xlim([t0, t1])
plt.title("Observations")
plt.legend()
plt.show()

input_data = (observation_grid, tf.constant(tf.squeeze(observation_vals, axis=0)))

likelihood = Gaussian(noise_stddev**2)
"""
CVI-GPR Taylor
"""
kernel = OrnsteinUhlenbeck(decay=decay, diffusion=q)

gpr_taylor_model = CVIGaussianProcessTaylorKernel(input_data, kernel=kernel, likelihood=likelihood,
                                                  time_grid=time_grid, learning_rate=.9)

for _ in range(2):
    gpr_taylor_model.update_sites()

gpr_log_likelihood = gpr_taylor_model.log_likelihood().numpy()
print(f"CVI-GPR (Taylor) Likelihood : {gpr_log_likelihood}")
print(f"CVI-GPR (Taylor) ELBO: {gpr_taylor_model.classic_elbo()}")

m_cvi_taylor, S_cvi_taylor = gpr_taylor_model.dist_q.marginals
S_std_cvi_taylor = tf.sqrt(S_cvi_taylor).numpy() + noise_stddev
m_cvi_taylor = m_cvi_taylor.numpy()

"""Plot"""
plot_observations(observation_grid, observation_vals)
plt.plot(time_grid.reshape(-1), m_cvi_taylor.reshape(-1))

plt.fill_between(
            time_grid,
            y1=(m_cvi_taylor.reshape(-1) - 2 * S_std_cvi_taylor.reshape(-1)).reshape(-1, ),
            y2=(m_cvi_taylor.reshape(-1) + 2 * S_std_cvi_taylor.reshape(-1)).reshape(-1, ),
            edgecolor="red",
            label="CVI-Taylor",
            facecolor=(0, 0, 0, 0.),
            linestyle='dashed'
        )

plt.legend()
plt.show()
