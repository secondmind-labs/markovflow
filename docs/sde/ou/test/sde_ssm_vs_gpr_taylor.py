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

from markovflow.sde.sde import OrnsteinUhlenbeckSDE
from markovflow.kernels import OrnsteinUhlenbeck
from markovflow.models.cvi_sde import SDESSM

import sys
sys.path.append("../..")
from sde_exp_utils import predict_ssm, plot_observations, plot_posterior, get_cvi_gpr_taylor, predict_cvi_gpr_taylor

os.environ['WANDB_MODE'] = 'offline'
"""Logging init"""
wandb.init(project="VI-SDE", entity="vermaprakhar")

DTYPE = default_float()
plt.rcParams["figure.figsize"] = [15, 5]

"""
Parameters
"""
data_dir = "../data/45"

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

print(f"True decay value of the OU SDE is {decay}")
print(f"Noise std-dev is {noise_stddev}")

input_data = (observation_grid, tf.constant(tf.squeeze(observation_vals, axis=0)))

"""
GPR - Taylor
"""
likelihood_gpr = Gaussian(noise_stddev**2)

kernel = OrnsteinUhlenbeck(decay=decay, diffusion=q)

cvi_gpr_taylor_model, cvi_taylor_params, cvi_taylor_elbo_vals = get_cvi_gpr_taylor(input_data, kernel, time_grid,
                                                                                   likelihood_gpr, train=False,
                                                                                   sites_lr=.9)

print(f"CVI-GPR (Taylor) ELBO: {cvi_gpr_taylor_model.classic_elbo()}")

"""
SDE-SSM
"""
# Prior SDE
true_decay = decay * tf.ones((1, 1), dtype=DTYPE)
true_q = q * tf.ones((1, 1), dtype=DTYPE)
prior_sde_vgp = OrnsteinUhlenbeckSDE(decay=true_decay, q=true_q)

# likelihood
likelihood_vgp = Gaussian(noise_stddev**2)

sde_ssm_model = SDESSM(input_data=input_data, prior_sde=prior_sde_vgp, grid=time_grid, likelihood=likelihood_vgp,
                       learning_rate=0.9)

sde_ssm_model.initial_mean = tf.reshape(kernel.initial_mean(tf.TensorShape(1)), sde_ssm_model.initial_mean.shape)
sde_ssm_model.initial_chol_cov = tf.reshape(tf.linalg.cholesky(kernel.initial_covariance(time_grid[..., 0:1])), sde_ssm_model.initial_chol_cov.shape)
sde_ssm_model.fx_covs = sde_ssm_model.initial_chol_cov.numpy().item()**2 + 0 * sde_ssm_model.fx_covs
sde_ssm_model._linearize_prior()

sde_ssm_elbo = [sde_ssm_model.classic_elbo().numpy().item()]
for _ in range(4):
    sde_ssm_model.update_sites()
    sde_ssm_elbo.append(sde_ssm_model.classic_elbo().numpy().item())

"""
Predict Posterior
"""
plot_observations(observation_grid, observation_vals)
m_gpr, s_std_gpr = predict_cvi_gpr_taylor(cvi_gpr_taylor_model, noise_stddev)
sde_ssm_m, sde_ssm_std = predict_ssm(sde_ssm_model, noise_stddev)

"""
Compare Posterior
"""
plot_posterior(m_gpr, s_std_gpr, time_grid, "GPR")
plot_posterior(sde_ssm_m, sde_ssm_std, time_grid, "SDE-SSM")
plt.legend()

plt.show()

"""SDE-SSM and GPR should give same posterior"""
np.testing.assert_array_almost_equal(sde_ssm_m, m_gpr)
np.testing.assert_array_almost_equal(sde_ssm_std.reshape(-1), s_std_gpr.reshape(-1))
