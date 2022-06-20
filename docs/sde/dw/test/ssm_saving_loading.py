"""Double-well SDE CVI vs SDE VI"""
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb
import tensorflow as tf
from gpflow import default_float
from gpflow.likelihoods import Gaussian

from markovflow.sde.sde import PriorDoubleWellSDE
from markovflow.models.cvi_sde import SDESSM

os.environ['WANDB_MODE'] = 'offline'
"""Logging init"""
wandb.init(project="VI-SDE", entity="vermaprakhar")

DTYPE = default_float()
plt.rcParams["figure.figsize"] = [15, 5]

"""
Parameters
"""
data_dir = "../data/82"

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

print(f"Noise std-dev is {noise_stddev}")

input_data = (observation_grid, tf.constant(tf.squeeze(observation_vals, axis=0)))

"""
SDE-SSM
"""

true_q = q * tf.ones((1, 1), dtype=DTYPE)
prior_sde_ssm = PriorDoubleWellSDE(q=true_q)

# likelihood
likelihood_ssm = Gaussian(noise_stddev**2)

"""Model 1"""
ssm_model = SDESSM(input_data=input_data, prior_sde=prior_sde_ssm, grid=time_grid, likelihood=likelihood_ssm,
                   learning_rate=1., prior_params_lr=0.01)

ssm_model.update_sites()
ssm_model.update_prior_sde()
ssm_model._linearize_prior()

"""
Model 2
"""
ssm_model2 = SDESSM(input_data=input_data, prior_sde=prior_sde_ssm, grid=time_grid, likelihood=likelihood_ssm,
                    learning_rate=1.)

# Load trained model variables
ssm_model2.data_sites.nat1 = ssm_model.data_sites.nat1
ssm_model2.data_sites.nat2 = ssm_model.data_sites.nat2
ssm_model2.data_sites.log_norm = ssm_model.data_sites.log_norm

ssm_model2.sites_nat1 = ssm_model.sites_nat1
ssm_model2.sites_nat2 = ssm_model.sites_nat2

ssm_model2.fx_mus = tf.identity(ssm_model.fx_mus)
ssm_model2.fx_covs = tf.identity(ssm_model.fx_covs)

ssm_model2.prior_sde.a = tf.identity(ssm_model.prior_sde.a)
ssm_model2.prior_sde.c = tf.identity(ssm_model.prior_sde.c)

ssm_model2._linearize_prior()

"""Compare ELBO"""
print(f"Trained model ELBO : {ssm_model.classic_elbo()}")
print(f"Model2 ELBO : {ssm_model2.classic_elbo()}")
