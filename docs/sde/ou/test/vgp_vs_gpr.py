"""
Script to compare the output of VGP and GPR Taylor.
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
from markovflow.models.vi_sde import VariationalMarkovGP

import sys
sys.path.append("../..")
from sde_exp_utils import predict_vgp, plot_observations, plot_posterior, get_cvi_gpr_taylor, predict_cvi_gpr_taylor

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

# changing dt
dt = 0.001
time_grid = tf.cast(np.arange(t0, t1 + dt, dt), dtype=DTYPE).numpy()

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
VGP
"""
# Prior SDE
true_decay = decay * tf.ones((1, 1), dtype=DTYPE)
true_q = q * tf.ones((1, 1), dtype=DTYPE)
prior_sde_vgp = OrnsteinUhlenbeckSDE(decay=true_decay, q=true_q)

# likelihood
likelihood_vgp = Gaussian(noise_stddev**2)

vgp_model = VariationalMarkovGP(input_data=input_data,
                                prior_sde=prior_sde_vgp, grid=time_grid, likelihood=likelihood_vgp,
                                lr=0.05, prior_params_lr=0.01)

vgp_model.p_initial_cov = tf.reshape(kernel.initial_covariance(time_grid[..., 0:1]), vgp_model.p_initial_cov.shape)
vgp_model.q_initial_cov = tf.identity(vgp_model.p_initial_cov)
vgp_model.p_initial_mean = tf.reshape(kernel.initial_mean(tf.TensorShape(1)), vgp_model.p_initial_mean.shape)
vgp_model.q_initial_mean = tf.identity(vgp_model.p_initial_mean)

vgp_model.A = decay + 0. * vgp_model.A

v_gp_elbo, v_gp_prior_vals = vgp_model.run(update_prior=False)

"""
Predict Posterior
"""
plot_observations(observation_grid, observation_vals)
m_gpr, s_std_gpr = predict_cvi_gpr_taylor(cvi_gpr_taylor_model, noise_stddev)
m_vgp, s_std_vgp = predict_vgp(vgp_model, noise_stddev)

"""
Compare Posterior
"""
plot_posterior(m_gpr, s_std_gpr, time_grid, "GPR")
plot_posterior(m_vgp, s_std_vgp, time_grid, "VGP")
plt.legend()

plt.show()


"""ELBO comparison"""
plt.plot(cvi_taylor_elbo_vals, color="black", label="CVI-GPR (Taylor)")
plt.plot(v_gp_elbo, label="VGP")
plt.title("ELBO")
plt.legend()
plt.show()

"""GPR and VGP should give same posterior"""
np.testing.assert_array_almost_equal(m_vgp, m_gpr, decimal=2)
np.testing.assert_array_almost_equal(s_std_vgp, s_std_gpr.reshape(-1), decimal=2)
