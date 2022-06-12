"""Double-well SDE CVI vs SDE VI"""
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from gpflow import default_float
from gpflow.likelihoods import Gaussian

from markovflow.sde.sde import DoubleWellSDE, PriorDoubleWellSDE
from markovflow.models.cvi_sde import SDESSM
from markovflow.models.vi_sde import VariationalMarkovGP

import sys
sys.path.append("..")
from sde_exp_utils import get_gpr, predict_vgp, predict_ssm, predict_gpr, plot_observations, plot_posterior, \
    get_cvi_gpr, predict_cvi_gpr

DTYPE = default_float()
plt.rcParams["figure.figsize"] = [15, 5]

"""
Parameters
"""
data_dir = "data/63"

learn_prior_sde = True

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

if learn_prior_sde:
    plot_save_dir = os.path.join(data_dir, "learning")
else:
    plot_save_dir = os.path.join(data_dir, "inference")

if not os.path.exists(plot_save_dir):
    os.makedirs(plot_save_dir)

"""
SDE-SSM
"""
# Prior SDE
if learn_prior_sde:
    true_q = q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_ssm = PriorDoubleWellSDE(q=true_q)
else:
    true_q = q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_ssm = DoubleWellSDE(q=true_q)

# likelihood
likelihood_ssm = Gaussian(noise_stddev**2)

# model
ssm_model = SDESSM(input_data=input_data, prior_sde=prior_sde_ssm, grid=time_grid, likelihood=likelihood_ssm,
                   learning_rate=0.5, prior_params_lr=0.01)
ssm_elbo, ssm_prior_prior_vals = ssm_model.run(update_prior=learn_prior_sde)
if learn_prior_sde:
    ssm_prior_a_values = ssm_prior_prior_vals[0]
    ssm_prior_c_values = ssm_prior_prior_vals[1]

"""
VGP
"""
# Prior SDE
if learn_prior_sde:
    true_q = q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_vgp = PriorDoubleWellSDE(q=true_q)
else:
    true_q = q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_vgp = DoubleWellSDE(q=true_q)

# likelihood
likelihood_vgp = Gaussian(noise_stddev**2)

vgp_model = VariationalMarkovGP(input_data=input_data,
                                prior_sde=prior_sde_vgp, grid=time_grid, likelihood=likelihood_vgp,
                                lr=0.05, prior_params_lr=0.01)

v_gp_elbo, v_gp_prior_vals = vgp_model.run(update_prior=learn_prior_sde)
if learn_prior_sde:
    v_gp_prior_a_values = v_gp_prior_vals[0]
    v_gp_prior_c_values = v_gp_prior_vals[1]

"""
Predict Posterior
"""
plt.clf()
plot_observations(observation_grid, observation_vals)
m_ssm, s_std_ssm = predict_ssm(ssm_model, noise_stddev)
m_vgp, s_std_vgp = predict_vgp(vgp_model, noise_stddev)
"""
Compare Posterior
"""
plot_posterior(m_ssm, s_std_ssm, time_grid, "SDE-SSM")
plot_posterior(m_vgp, s_std_vgp, time_grid, "VGP")
plt.legend()

plt.savefig(os.path.join(plot_save_dir, "posterior.svg"))

plt.show()

"""
Plot drift evolution
"""
if learn_prior_sde:

    x = np.linspace(-2, 2, 20).reshape((-1, 1))

    true_drift = true_dw_sde.drift(x, None)
    sde_ssm_learnt_drift = prior_sde_ssm.drift(x, None)
    vgp_learnt_drift = prior_sde_vgp.drift(x, None)

    plt.subplots(1, 1, figsize=(5, 5))

    plt.clf()
    plt.plot(x, vgp_learnt_drift, label="VGP", color="green")
    plt.plot(x, true_drift, label="True drift", color="black")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])

    plt.title("Drift")
    plt.legend()
    plt.savefig(os.path.join(plot_save_dir, "drift.svg"))
    plt.show()

    # print(f"SSM learnt drift : f(x) = {ssm_prior_a_values[-1]} * x * ({ssm_prior_c_values[-1]} - x^2)")
    print(f"VGP learnt drift : f(x) = {v_gp_prior_a_values[-1]} * x * ({v_gp_prior_c_values[-1]} - x^2)")


"""ELBO comparison"""
plt.clf()
plt.plot(ssm_elbo, label="SDE-SSM")
plt.plot(v_gp_elbo[2:], label="VGP")
plt.title("ELBO")
plt.legend()
plt.savefig(os.path.join(plot_save_dir, "elbo.svg"))
plt.show()


"Save SDE-SSM data"
np.savez(os.path.join(plot_save_dir, "ssm_data_sites.npz"), nat1=ssm_model.data_sites.nat1.numpy(),
         nat2=ssm_model.data_sites.nat2.numpy(), log_norm=ssm_model.data_sites.log_norm.numpy())

np.savez(os.path.join(plot_save_dir, "ssm_inference.npz"), m=m_ssm, S=tf.square(s_std_ssm))
np.savez(os.path.join(plot_save_dir, "ssm_elbo.npz"), elbo=ssm_elbo)

if learn_prior_sde:
    np.savez(os.path.join(plot_save_dir, "ssm_learnt_sde.npz"), a=ssm_prior_a_values, c=ssm_prior_c_values)

"Save VGP data"
np.savez(os.path.join(plot_save_dir, "vgp_A_b.npz"), A=vgp_model.A.numpy(), b=vgp_model.b.numpy())
np.savez(os.path.join(plot_save_dir, "vgp_lagrange.npz"), psi_lagrange=vgp_model.psi_lagrange.numpy(),
         lambda_lagrange=vgp_model.lambda_lagrange.numpy())

np.savez(os.path.join(plot_save_dir, "vgp_inference.npz"), m=m_vgp, S=tf.square(s_std_vgp))
np.savez(os.path.join(plot_save_dir, "vgp_elbo.npz"), elbo=v_gp_elbo)
if learn_prior_sde:
    np.savez(os.path.join(plot_save_dir, "vgp_learnt_sde.npz"), a=v_gp_prior_a_values,
             c=v_gp_prior_c_values)
