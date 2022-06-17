"""Learning bound for SDE parameters"""

import matplotlib.pyplot as plt
import numpy as np
import os
import wandb

import tensorflow as tf
from gpflow import default_float
from gpflow.likelihoods import Gaussian
from gpflow.base import Parameter

from markovflow.sde.sde import DoubleWellSDE, PriorDoubleWellSDE
from markovflow.models.cvi_sde import SDESSM

import sys
sys.path.append("..")
from sde_exp_utils import get_gpr, predict_vgp, predict_ssm, predict_gpr, plot_observations, plot_posterior, \
    get_cvi_gpr, predict_cvi_gpr

DTYPE = default_float()
plt.rcParams["figure.figsize"] = [15, 5]

os.environ['WANDB_MODE'] = 'offline'
"""Logging init"""
wandb.init(project="VI-SDE", entity="vermaprakhar")

"""
Parameters
"""
data_dir = "data/91"
model_used = "learning"  # path to inference or learning model

"""
Generate observations for a linear SDE
"""
data_path = os.path.join(data_dir, "data.npz")
data = np.load(data_path)
q = data["q"]
noise_stddev = data["noise_stddev"]
x0 = data["x0"]
observation_vals = data["observation_vals"]
observation_grid = data["observation_grid"]
latent_process = data["latent_process"]
time_grid = data["time_grid"]
t0 = time_grid[0]
t1 = time_grid[-1]

true_dw_sde = DoubleWellSDE(q=q * tf.ones((1, 1), dtype=DTYPE))

plt.clf()
plot_observations(observation_grid, observation_vals)
plt.plot(time_grid, tf.reshape(latent_process, (-1)), label="Latent Process", alpha=0.2, color="gray")
plt.xlabel("Time (t)")
plt.ylabel("y(t)")
plt.ylim([-2, 2])
plt.xlim([t0, t1])
plt.title("Observations")
plt.legend()
plt.show()

print(f"Noise std-dev is {noise_stddev}")

input_data = (observation_grid, tf.constant(tf.squeeze(observation_vals, axis=0)))

"""
SDE-SSM Model
"""
# Prior SDE
true_q = q * tf.ones((1, 1), dtype=DTYPE)
prior_sde_ssm = PriorDoubleWellSDE(q=true_q)

# likelihood
likelihood_ssm = Gaussian(noise_stddev**2)

# model
ssm_model = SDESSM(input_data=input_data, prior_sde=prior_sde_ssm, grid=time_grid, likelihood=likelihood_ssm,
                   learning_rate=1.)

# Load trained model variables
data_sites = np.load(os.path.join(data_dir, model_used, "ssm_data_sites.npz"))
ssm_model.data_sites.nat1 = Parameter(data_sites["nat1"])
ssm_model.data_sites.nat2 = Parameter(data_sites["nat2"])
ssm_model.data_sites.log_norm = Parameter(data_sites["log_norm"])

print(f"ELBO : {ssm_model.classic_elbo().numpy().item()}")

"""ELBO BOUND"""
n = 100

a_value_range = np.linspace(0.2, 6, n).reshape((-1, 1))
c_value_range = np.linspace(0.2, 2., n).reshape((1, -1))

a_value_range = np.repeat(a_value_range, n, axis=1)
c_value_range = np.repeat(c_value_range, n, axis=0)

ssm_elbo_vals = []

true_q = q * tf.ones((1, 1), dtype=DTYPE)

for a, c in zip(a_value_range.reshape(-1), c_value_range.reshape(-1)):
    print(f"Calculating ELBO bound for a={a}, c={c}")
    ssm_model.prior_sde = PriorDoubleWellSDE(q=true_q, initial_a_val=a, initial_c_val=c)
    ssm_model._linearize_prior()  # To linearize the new prior
    ssm_elbo_vals.append(ssm_model.classic_elbo().numpy().item())

ssm_elbo_vals = np.array(ssm_elbo_vals).reshape((n, n)).T

plt.clf()
plt.subplots(1, 1, figsize=(5, 5))

c = plt.pcolormesh(a_value_range[:, 0], c_value_range[0], ssm_elbo_vals,
                   vmin=np.min(ssm_elbo_vals), vmax=np.max(ssm_elbo_vals), shading='auto')
plt.colorbar(c)
plt.savefig(os.path.join(data_dir, "learning", "ssm_learning_bound.svg"))
plt.show()

np.savez(os.path.join(data_dir, "learning", "ssm_learning_bound.npz"), elbo=ssm_elbo_vals, a=a_value_range,
         c=c_value_range)