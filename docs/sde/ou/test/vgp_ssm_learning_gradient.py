"""
After convergence of inference the gradient of the learning objective should be the same for both the models.
"""

import os

import matplotlib.pyplot as plt
import wandb
import numpy as np
import tensorflow as tf
from gpflow import default_float
from gpflow.likelihoods import Gaussian

from markovflow.sde.sde import PriorOUSDE
from markovflow.kernels import OrnsteinUhlenbeck
from markovflow.models.cvi_sde import SDESSM
from markovflow.models.vi_sde import VariationalMarkovGP

from markovflow.sde.sde_utils import KL_sde
from markovflow.sde.sde_utils import linearize_sde

os.environ['WANDB_MODE'] = 'offline'
"""Logging init"""
wandb.init(project="VI-SDE", entity="vermaprakhar")

DTYPE = default_float()
plt.rcParams["figure.figsize"] = [15, 5]

"""
Parameters
"""
data_dir = "../data/12"

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

kernel = OrnsteinUhlenbeck(decay=decay, diffusion=q)

"""
SDE-SSM
"""
# Prior SDE
true_q = q * tf.ones((1, 1), dtype=DTYPE)
prior_sde = PriorOUSDE(q=true_q)

# likelihood
likelihood = Gaussian(noise_stddev**2)

sde_ssm_model = SDESSM(input_data=input_data, prior_sde=prior_sde, grid=time_grid, likelihood=likelihood,
                       learning_rate=0.9)

sde_ssm_model.initial_mean = tf.reshape(kernel.initial_mean(tf.TensorShape(1)), sde_ssm_model.initial_mean.shape)
sde_ssm_model.initial_chol_cov = tf.reshape(tf.linalg.cholesky(kernel.initial_covariance(time_grid[..., 0:1])),
                                            sde_ssm_model.initial_chol_cov.shape)
sde_ssm_model.fx_covs = sde_ssm_model.initial_chol_cov.numpy().item()**2 + 0 * sde_ssm_model.fx_covs
sde_ssm_model._linearize_prior()

"""
VGP
"""
# creater a finer grid for VGP
# dt = 0.001
# time_grid = tf.cast(np.arange(t0, t1 + dt, dt), dtype=DTYPE).numpy()

vgp_model = VariationalMarkovGP(input_data=input_data,
                                prior_sde=prior_sde, grid=time_grid, likelihood=likelihood,
                                lr=0.08, prior_params_lr=0.01)

vgp_model.p_initial_cov = tf.reshape(kernel.initial_covariance(time_grid[..., 0:1]), vgp_model.p_initial_cov.shape)
vgp_model.q_initial_cov = tf.identity(vgp_model.p_initial_cov)
vgp_model.p_initial_mean = tf.reshape(kernel.initial_mean(tf.TensorShape(1)), vgp_model.p_initial_mean.shape)
vgp_model.q_initial_mean = tf.identity(vgp_model.p_initial_mean)
vgp_model.A = -1 * prior_sde.decay + 0. * vgp_model.A

print(f"SDE-SSM converges at ELBO : {sde_ssm_model.classic_elbo()}")
print(f"VGP converges at ELBO : {vgp_model.elbo()}")


def func(vgp_model: VariationalMarkovGP):
    m, S = vgp_model.forward_pass

    # remove the final state
    m = tf.stop_gradient(m[:-1])
    S = tf.stop_gradient(S[:-1])

    A = vgp_model.A[:-1]
    b = vgp_model.b[:-1]

    return KL_sde(vgp_model.prior_sde, A, b, m, S, vgp_model.dt) - vgp_model.KL_initial_state()


# Checking the gradient of the learning objective for SSM
def learning_gradient(ssm_model: SDESSM):
    def dist_p():
        fx_mus = ssm_model.fx_mus[:, :-1, :]
        fx_covs = ssm_model.fx_covs[:, :-1, :, :]

        return linearize_sde(sde=ssm_model.prior_sde, transition_times=ssm_model.time_points, q_mean=fx_mus,
                             q_covar=fx_covs, initial_mean=ssm_model.initial_mean,
                             initial_chol_covariance=ssm_model.initial_chol_cov,
                             )

    def loss():
        return ssm_model.kl(dist_p=dist_p()) + ssm_model.loss_lin(dist_p=dist_p())

    with tf.GradientTape() as tape:
        tape.watch([ssm_model.prior_sde.trainable_variables])
        ssm_val = loss()
        ssm_grad = tape.gradient(ssm_val, [ssm_model.prior_sde.trainable_variables])

    print(f"SDE-SSM objective value : {ssm_val}")
    print(f"SDE-SSM objective gradient : {ssm_grad}")

learning_gradient(sde_ssm_model)

# Checking the gradient of the learning objective for VGP
with tf.GradientTape() as tape:
    tape.watch([vgp_model.prior_sde.trainable_variables])
    vgp_val = func(vgp_model)
    vgp_grad = tape.gradient(vgp_val, [vgp_model.prior_sde.trainable_variables])

print(f"VGP objective value : {vgp_val}")
print(f"VGP objective gradient : {vgp_grad}")
