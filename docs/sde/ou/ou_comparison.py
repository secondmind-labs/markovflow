"""OU SDE CVI vs SDE VI"""
import os

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow import default_float
from gpflow.likelihoods import Gaussian

# FIXME: Remove this later. This is for training on Lab's GPU.
# Restrict TensorFlow to only use the first GPU
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     tf.config.set_visible_devices(gpus[1], 'GPU')

from markovflow.sde.sde import OrnsteinUhlenbeckSDE, PriorOUSDE
from markovflow.kernels import OrnsteinUhlenbeck
from markovflow.models.cvi_sde import SDESSM
from markovflow.models.vi_sde import VariationalMarkovGP

import sys
sys.path.append("..")
from sde_exp_utils import get_gpr, predict_vgp, predict_ssm, predict_gpr, plot_observations, plot_posterior, \
    get_cvi_gpr, predict_cvi_gpr, get_cvi_gpr_taylor, predict_cvi_gpr_taylor

DTYPE = default_float()
plt.rcParams["figure.figsize"] = [15, 5]

"""
Parameters
"""
data_dir = "data/98"

learn_prior_sde = True
prior_initial_decay_val = 2. + 0 * tf.abs(tf.random.normal((1, 1), dtype=DTYPE))  # Used when learning prior sde

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

plt.clf()
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

if learn_prior_sde:
    plot_save_dir = os.path.join(data_dir, "learning")
else:
    plot_save_dir = os.path.join(data_dir, "inference")

if not os.path.exists(plot_save_dir):
    os.makedirs(plot_save_dir)
"""
GPR - Taylor
"""
likelihood_gpr = Gaussian(noise_stddev**2)

if learn_prior_sde:
    kernel = OrnsteinUhlenbeck(decay=prior_initial_decay_val.numpy().item(), diffusion=q)
    gpflow.set_trainable(kernel.diffusion, False)
else:
    kernel = OrnsteinUhlenbeck(decay=decay, diffusion=q)

cvi_gpr_taylor_model, cvi_taylor_params, cvi_taylor_elbo_vals = get_cvi_gpr_taylor(input_data, kernel, time_grid, likelihood_gpr,
                                                             train=learn_prior_sde, sites_lr=1.)
cvi_prior_decay_values = -1 * np.array(cvi_taylor_params[0])

print(f"CVI-GPR (Taylor) ELBO: {cvi_gpr_taylor_model.classic_elbo()}")

"""
SDE-SSM
"""
# Prior SDE
if learn_prior_sde:
    prior_decay = prior_initial_decay_val
    true_q = q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_ssm = PriorOUSDE(initial_val=-1*prior_decay, q=true_q)  # As prior OU SDE doesn't have a negative sign inside it.
else:
    true_decay = decay * tf.ones((1, 1), dtype=DTYPE)
    true_q = q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_ssm = OrnsteinUhlenbeckSDE(decay=true_decay, q=true_q)

# likelihood
likelihood_ssm = Gaussian(noise_stddev**2)

# model
ssm_model = SDESSM(input_data=input_data, prior_sde=prior_sde_ssm, grid=time_grid, likelihood=likelihood_ssm,
                   learning_rate=0.9, prior_params_lr=0.01)
ssm_model.initial_mean = tf.reshape(kernel.initial_mean(tf.TensorShape(1)), ssm_model.initial_mean.shape)
ssm_model.initial_chol_cov = tf.reshape(tf.linalg.cholesky(kernel.initial_covariance(time_grid[..., 0:1])), ssm_model.initial_chol_cov.shape)
ssm_model.fx_covs = ssm_model.initial_chol_cov.numpy().item()**2 + 0 * ssm_model.fx_covs
ssm_model._linearize_prior()

ssm_elbo, ssm_prior_prior_vals = ssm_model.run(update_prior=learn_prior_sde)
if learn_prior_sde:
    ssm_prior_decay_values = ssm_prior_prior_vals[0]

"""
VGP
"""
# Prior SDE
if learn_prior_sde:
    prior_decay = prior_initial_decay_val
    true_q = q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_vgp = PriorOUSDE(initial_val=-1*prior_decay, q=true_q)  # As prior OU SDE doesn't have a negative sign inside it.
else:
    true_decay = decay * tf.ones((1, 1), dtype=DTYPE)
    true_q = q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_vgp = OrnsteinUhlenbeckSDE(decay=true_decay, q=true_q)

# likelihood
likelihood_vgp = Gaussian(noise_stddev**2)

vgp_model = VariationalMarkovGP(input_data=input_data,
                                prior_sde=prior_sde_vgp, grid=time_grid, likelihood=likelihood_vgp,
                                lr=0.01, prior_params_lr=0.01)

vgp_model.p_initial_cov = tf.reshape(kernel.initial_covariance(time_grid[..., 0:1]), vgp_model.p_initial_cov.shape)
vgp_model.q_initial_cov = tf.reshape(ssm_model.fx_covs[:, 0], shape=vgp_model.q_initial_cov.shape) #tf.identity(vgp_model.p_initial_cov)
vgp_model.p_initial_mean = tf.reshape(kernel.initial_mean(tf.TensorShape(1)), vgp_model.p_initial_mean.shape)
vgp_model.q_initial_mean = tf.reshape(ssm_model.fx_mus[:, 0], shape=vgp_model.q_initial_mean.shape) #tf.identity(vgp_model.p_initial_mean)

vgp_model.A = decay + 0. * vgp_model.A

v_gp_elbo, v_gp_prior_vals = vgp_model.run(update_prior=learn_prior_sde)
if learn_prior_sde:
    v_gp_prior_decay_values = v_gp_prior_vals[0]

"""
Predict Posterior
"""
plt.clf()
plot_observations(observation_grid, observation_vals)
m_gpr, s_std_gpr = predict_cvi_gpr_taylor(cvi_gpr_taylor_model, noise_stddev)
m_ssm, s_std_ssm = predict_ssm(ssm_model, noise_stddev)
m_vgp, s_std_vgp = predict_vgp(vgp_model, noise_stddev)
"""
Compare Posterior
"""
plot_posterior(m_gpr, s_std_gpr, time_grid, "GPR")
plot_posterior(m_ssm, s_std_ssm, time_grid, "SDE-SSM")
plot_posterior(m_vgp, s_std_vgp, time_grid, "VGP")
plt.legend()

plt.savefig(os.path.join(plot_save_dir, "posterior.svg"))

plt.show()

"""
Plot drift evolution
"""
if learn_prior_sde:
    plt.clf()
    plt.hlines(-1 * decay, 0, max(len(v_gp_prior_decay_values), len(ssm_prior_decay_values)),
               label="True Value", color="black", linestyles="dashed")
    plt.plot(v_gp_prior_decay_values, label="VGP", color="green")
    plt.plot(ssm_prior_decay_values, label="SDE-SSM", color="blue")
    plt.plot(cvi_prior_decay_values, label="CVI-GPR", color="red")
    plt.title("Prior Learning (decay)")
    plt.legend()
    plt.ylabel("decay")
    plt.savefig(os.path.join(plot_save_dir, "prior_learning_decay.svg"))
    plt.show()

    print("Q values: ")
    print(f"GPR : {kernel.diffusion.numpy().item()}")
    print(f"SDE-SSM : {prior_sde_ssm.q.numpy().item()}")
    print(f"VGP : {prior_sde_vgp.q.numpy().item()}")

"""ELBO comparison"""
plt.clf()
plt.plot(cvi_taylor_elbo_vals, color="black", label="CVI-GPR (Taylor)")
plt.plot(ssm_elbo, label="SDE-SSM")
plt.plot(v_gp_elbo, label="VGP")
plt.title("ELBO")
plt.legend()
plt.savefig(os.path.join(plot_save_dir, "elbo.svg"))
plt.show()


"""
ELBO Bound
"""
if not learn_prior_sde:
    decay_value_range = np.linspace(0.01, decay + 2.5, 10)
    gpr_taylor_elbo_vals = []
    ssm_elbo_vals = []
    vgp_elbo_vals = []
    true_q = q * tf.ones((1, 1), dtype=DTYPE)

    for decay_val in decay_value_range:
        kernel = OrnsteinUhlenbeck(decay=decay_val, diffusion=q)
        cvi_gpr_taylor_model.orig_kernel = kernel
        gpr_taylor_elbo_vals.append(cvi_gpr_taylor_model.classic_elbo().numpy().item())

        ssm_model.prior_sde = OrnsteinUhlenbeckSDE(decay=decay_val, q=true_q)
        ssm_model._linearize_prior()  # To linearize the new prior
        ssm_elbo_vals.append(ssm_model.classic_elbo())

        vgp_model.prior_sde = OrnsteinUhlenbeckSDE(decay=decay_val, q=true_q)
        vgp_elbo_vals.append(vgp_model.elbo())

    plt.clf()
    plt.subplots(1, 1, figsize=(5, 5))
    plt.plot(decay_value_range, ssm_elbo_vals, label="SDE-SSM")
    plt.plot(decay_value_range, vgp_elbo_vals, label="VGP")
    plt.plot(decay_value_range, gpr_taylor_elbo_vals, label="CVI-GPR (Taylor) ELBO", alpha=0.2, linestyle="dashed",
             color="black")
    plt.vlines(decay, np.min(gpr_taylor_elbo_vals), np.max(gpr_taylor_elbo_vals))
    plt.legend()
    plt.savefig(os.path.join(plot_save_dir, "elbo_bound.svg"))
    plt.show()


"Save SDE-SSM data"
np.savez(os.path.join(plot_save_dir, "ssm_data_sites.npz"), nat1=ssm_model.data_sites.nat1.numpy(),
         nat2=ssm_model.data_sites.nat2.numpy(), log_norm=ssm_model.data_sites.log_norm.numpy())

np.savez(os.path.join(plot_save_dir, "ssm_inference.npz"), m=m_ssm, S=tf.square(s_std_ssm))
np.savez(os.path.join(plot_save_dir, "ssm_elbo.npz"), elbo=ssm_elbo)

if learn_prior_sde:
    np.savez(os.path.join(plot_save_dir, "ssm_learnt_sde.npz"), decay=ssm_prior_decay_values)

"Save VGP data"
np.savez(os.path.join(plot_save_dir, "vgp_A_b.npz"), A=vgp_model.A.numpy(), b=vgp_model.b.numpy())
np.savez(os.path.join(plot_save_dir, "vgp_lagrange.npz"), psi_lagrange=vgp_model.psi_lagrange.numpy(),
         lambda_lagrange=vgp_model.lambda_lagrange.numpy())

np.savez(os.path.join(plot_save_dir, "vgp_inference.npz"), m=m_vgp, S=tf.square(s_std_vgp))
np.savez(os.path.join(plot_save_dir, "vgp_elbo.npz"), elbo=v_gp_elbo)
if learn_prior_sde:
    np.savez(os.path.join(plot_save_dir, "vgp_learnt_sde.npz"), decay=v_gp_prior_decay_values)


"""SDE-SSM and VGP should give same posterior"""
np.testing.assert_array_almost_equal(m_vgp, m_ssm, decimal=2)
np.testing.assert_array_almost_equal(s_std_vgp, s_std_ssm.reshape(-1), decimal=2)
