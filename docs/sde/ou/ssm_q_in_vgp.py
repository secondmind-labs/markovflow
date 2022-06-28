"""Train SDE-SSM model and put posterior drift vlues in VGP to calculate the ELBO"""

import os

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import default_float
from gpflow.likelihoods import Gaussian
import wandb
from markovflow.sde.sde import PriorOUSDE
from markovflow.kernels import OrnsteinUhlenbeck
from markovflow.models.cvi_sde import SDESSM
from markovflow.models.vi_sde import VariationalMarkovGP
from markovflow.state_space_model import StateSpaceModel
import matplotlib.pyplot as plt


DTYPE = default_float()
plt.rcParams["figure.figsize"] = [15, 5]

os.environ['WANDB_MODE'] = 'offline'
wandb.init(project="VI-SDE", entity="vermaprakhar")

"""
Parameters
"""
data_dir = "data/36"

prior_initial_decay_val = .8 + 0 * tf.abs(tf.random.normal((1, 1), dtype=DTYPE))  # Used when learning prior sde

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

print(f"True decay value of the OU SDE is {decay}")
print(f"Noise std-dev is {noise_stddev}")

input_data = (observation_grid, tf.constant(tf.squeeze(observation_vals, axis=0)))

dt = time_grid[1] - time_grid[0]

kernel = OrnsteinUhlenbeck(decay=prior_initial_decay_val.numpy().item(), diffusion=q)
gpflow.set_trainable(kernel.diffusion, False)

"""
SDE-SSM
"""

prior_decay = prior_initial_decay_val
true_q = q * tf.ones((1, 1), dtype=DTYPE)
prior_sde = PriorOUSDE(initial_val=-1*prior_decay, q=true_q)  # As prior OU SDE doesn't have a negative sign inside it.

# likelihood
likelihood = Gaussian(noise_stddev**2)

# model
ssm_model = SDESSM(input_data=input_data, prior_sde=prior_sde, grid=time_grid, likelihood=likelihood,
                   learning_rate=1.)
ssm_model.initial_mean = tf.reshape(kernel.initial_mean(tf.TensorShape(1)), ssm_model.initial_mean.shape)
ssm_model.initial_chol_cov = tf.reshape(tf.linalg.cholesky(kernel.initial_covariance(time_grid[..., 0:1])),
                                        ssm_model.initial_chol_cov.shape)
ssm_model.fx_covs = ssm_model.initial_chol_cov.numpy().item()**2 + 0 * ssm_model.fx_covs
ssm_model._linearize_prior()

for _ in range(2):
    ssm_model.update_sites()

"""Convert SDE-SSM posterior to drift params"""
# posterior_ssm = ssm_model.dist_q
# posterior_A = -1 * (posterior_ssm.state_transitions - np.eye(1, dtype=input_data[0].dtype)) / dt
# posterior_b = posterior_ssm.state_offsets / dt

posterior_A, posterior_b, ssm_params = ssm_model.get_posterior_drift_params()

ssm_m, ssm_S = ssm_model.dist_q.marginals
ssm_m = ssm_m.numpy()
ssm_S = ssm_S.numpy()

# Testing posterior SSM to see if the conversion was done rights
sde_posterior_ssm = StateSpaceModel(
                        state_transitions=ssm_params[0],
                        state_offsets=ssm_params[1],
                        chol_initial_covariance=ssm_params[2],
                        chol_process_covariances=ssm_params[3],
                        initial_mean=ssm_params[4],
                    )

np.testing.assert_array_almost_equal(ssm_m, sde_posterior_ssm.marginal_means)
np.testing.assert_array_almost_equal(ssm_S, sde_posterior_ssm.marginal_covariances)

"""VGP"""
vgp_model = VariationalMarkovGP(input_data=input_data, prior_sde=prior_sde, grid=time_grid, likelihood=likelihood)
                                # ,forward_ssm_q=tf.math.square(ssm_params[3]))

vgp_model.p_initial_cov = tf.reshape(kernel.initial_covariance(time_grid[..., 0:1]), vgp_model.p_initial_cov.shape)
vgp_model.p_initial_mean = tf.reshape(kernel.initial_mean(tf.TensorShape(1)), vgp_model.p_initial_mean.shape)

vgp_model.q_initial_mean = tf.reshape(ssm_params[4], vgp_model.q_initial_mean.shape)
vgp_model.q_initial_cov = tf.reshape(tf.math.square(ssm_params[2]), shape=vgp_model.q_initial_cov.shape)

# -1 because of how VGP is parameterized
vgp_model.A = -1 * tf.concat([posterior_A, -1 * tf.ones((1, 1, 1), dtype=posterior_A.dtype)], axis=0)
vgp_model.b = tf.concat([posterior_b, tf.zeros((1, 1), dtype=posterior_b.dtype)], axis=0)

vgp_m, vgp_S = vgp_model.forward_pass
vgp_m = vgp_m.numpy()
vgp_S = vgp_S.numpy()

print(ssm_model.classic_elbo())
print(vgp_model.elbo())

"""Train VGP"""
# dt = 0.001
# time_grid_finer = tf.cast(np.arange(t0, t1 + dt, dt), dtype=DTYPE).numpy()
time_grid_finer = time_grid

vgp_model_1 = VariationalMarkovGP(input_data=input_data, prior_sde=prior_sde, grid=time_grid_finer,
                                  likelihood=likelihood, lr=0.01)

vgp_model_1.p_initial_cov = tf.reshape(kernel.initial_covariance(time_grid[..., 0:1]), vgp_model_1.p_initial_cov.shape)
vgp_model_1.p_initial_mean = tf.reshape(kernel.initial_mean(tf.TensorShape(1)), vgp_model_1.p_initial_mean.shape)

vgp_model_1.run_inference_till_convergence(update_prior=False, update_initial_statistics=True)

"""Plotting"""
plt.subplots(1, 1, figsize=(5, 5))
plt.scatter(4, vgp_model.elbo(), label="VGP (Not trained)")
plt.scatter(4, vgp_model_1.elbo(), label="VGP")
plt.scatter(4, ssm_model.classic_elbo(), label="SDE-SSM")
plt.xticks([])
plt.title("ELBO (Convergence value)")
plt.legend()
plt.show()

plt.clf()
# plt.vlines(time_grid.reshape(-1), -50, 50, color="black", alpha=0.2)
plt.plot(time_grid_finer.reshape(-1), vgp_model_1.A.numpy().reshape(-1), label="VGP (Trained)")
plt.plot(time_grid.reshape(-1), vgp_model.A.numpy().reshape(-1), label="VGP (SDE-SSM Params)", alpha=0.4)
plt.legend()
plt.title("A")
plt.show()

plt.clf()
# plt.vlines(time_grid.reshape(-1), -50, 50, color="black", alpha=0.2)
plt.plot(time_grid_finer.reshape(-1), vgp_model_1.b.numpy().reshape(-1), label="VGP (Trained)")
plt.plot(time_grid.reshape(-1), vgp_model.b.numpy().reshape(-1), label="VGP (SDE-SSM Params)", alpha=0.4)
plt.legend()
plt.title("b")
plt.show()
