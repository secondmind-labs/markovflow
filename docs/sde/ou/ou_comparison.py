"""OU SDE CVI vs SDE VI"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow import default_float
from gpflow.likelihoods import Gaussian

from markovflow.sde.sde import OrnsteinUhlenbeckSDE, PriorOUSDE
from markovflow.kernels import OrnsteinUhlenbeck
from markovflow.models.cvi_sde import SDESSM
from markovflow.models.vi_sde import VariationalMarkovGP
from ou_utils import generate_data, get_gpr, predict_vgp, predict_ssm, predict_gpr, plot_observations, plot_posterior, \
    get_cvi_gpr, predict_cvi_gpr

tf.random.set_seed(83)
np.random.seed(83)
DTYPE = default_float()
plt.rcParams["figure.figsize"] = [15, 5]

"""
Parameters
"""
decay = 2.  # specify without the negative sign
q = 0.5
noise_var = 0.05
x0 = 1.

t0, t1 = 0.0, 5.
simulation_dt = 0.01  # Used for Euler-Maruyama
n_observations = 30

learn_prior_sde = True
prior_initial_decay_val = tf.random.normal((1, 1), dtype=DTYPE)  # Used when learning prior sde

"""
Generate observations for a linear SDE
"""
noise_stddev = np.sqrt(noise_var)

observation_vals, observation_grid, latent_process, time_grid = generate_data(decay=decay, q=q, x0=x0, t0=t0, t1=t1,
                                                                              simulation_dt=simulation_dt,
                                                                              noise_stddev=noise_stddev,
                                                                              n_observations=n_observations,
                                                                              dtype=DTYPE)

plot_observations(observation_grid.numpy(), observation_vals.numpy())
plt.plot(time_grid, tf.reshape(latent_process, (-1)), label="Latent Process", alpha=0.2, color="gray")
plt.xlabel("Time (t)")
plt.ylabel("y(t)")
plt.ylim([-2, 2])
plt.xlim([t0, t1])
plt.title("Observations")
plt.legend()
plt.show()

print(f"True decay value of the OU SDE is {decay}")
print(f"Noise std-dev is {noise_stddev}")

input_data = (observation_grid, tf.constant(tf.squeeze(observation_vals, axis=0)))
"""
GPR
"""
likelihood_gpr = Gaussian(noise_stddev**2)

if learn_prior_sde:
    kernel = OrnsteinUhlenbeck(decay=prior_initial_decay_val.numpy().item(), diffusion=q)
else:
    kernel = OrnsteinUhlenbeck(decay=decay, diffusion=q)

cvi_gpr_model = get_cvi_gpr(input_data, kernel, likelihood_gpr, train=learn_prior_sde)

gpr_model = get_gpr(input_data, kernel, train=learn_prior_sde, noise_stddev=noise_stddev)
gpr_log_likelihood = gpr_model.log_likelihood().numpy()
print(f"GPR Likelihood : {gpr_log_likelihood}")

print(f"CVI-GPR ELBO: {cvi_gpr_model.classic_elbo()}")
"""
SDE-SSM
"""
# Prior SDE
if learn_prior_sde:
    prior_decay = prior_initial_decay_val
    true_q = q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_ssm = PriorOUSDE(initial_val=prior_decay, q=true_q)
else:
    true_decay = decay * tf.ones((1, 1), dtype=DTYPE)
    true_q = q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_ssm = OrnsteinUhlenbeckSDE(decay=true_decay, q=true_q)

# likelihood
likelihood_ssm = Gaussian(noise_stddev**2)

# model
ssm_model = SDESSM(input_data=input_data, prior_sde=prior_sde_ssm, grid=time_grid, likelihood=likelihood_ssm,
                   learning_rate=0.9)

# For OU we know this relation for variance
ssm_model.initial_chol_cov = tf.linalg.cholesky((q/(2 * decay)) * tf.ones_like(ssm_model.initial_chol_cov))
ssm_model.fx_covs = ssm_model.initial_chol_cov.numpy().item()**2 + 0 * ssm_model.fx_covs

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
    prior_sde_vgp = PriorOUSDE(initial_val=prior_decay, q=true_q)
else:
    true_decay = decay * tf.ones((1, 1), dtype=DTYPE)
    true_q = q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_vgp = OrnsteinUhlenbeckSDE(decay=true_decay, q=true_q)

# likelihood
likelihood_vgp = Gaussian(noise_stddev**2)

vgp_model = VariationalMarkovGP(input_data=input_data,
                                prior_sde=prior_sde_vgp, grid=time_grid, likelihood=likelihood_vgp,
                                lr=0.1)
vgp_model.p_initial_cov = (q/(2 * decay)) * tf.ones((1, 1), dtype=DTYPE)  # For OU we know this relation for variance

v_gp_elbo, v_gp_prior_vals = vgp_model.run(update_prior=learn_prior_sde)
if learn_prior_sde:
    v_gp_prior_decay_values = v_gp_prior_vals[0]

"""
Predict Posterior
"""
plot_observations(observation_grid.numpy(), observation_vals.numpy())
# m_gpr, s_std_gpr = predict_gpr(gpr_model, time_grid.numpy())  # FOR GPR MODEL
m_gpr, s_std_gpr = predict_cvi_gpr(cvi_gpr_model, time_grid.numpy(), noise_stddev)
m_ssm, s_std_ssm = predict_ssm(ssm_model, noise_stddev)
m_vgp, s_std_vgp = predict_vgp(vgp_model, noise_stddev)
"""
Compare Posterior
"""
plot_posterior(m_gpr, s_std_gpr, time_grid.numpy(), "CVI-GPR")
plot_posterior(m_ssm, s_std_ssm, time_grid.numpy(), "SDE-SSM")
plot_posterior(m_vgp, s_std_vgp, time_grid.numpy(), "VGP")
plt.legend()
plt.show()

"""
Plot drift evolution
"""
if learn_prior_sde:
    plt.hlines(-1 * decay, 0, max(len(v_gp_prior_decay_values), len(ssm_prior_decay_values)),
               label="True Value", color="black", linestyles="dashed")
    plt.plot(v_gp_prior_decay_values, label="VGP", color="green")
    plt.plot(ssm_prior_decay_values, label="SDE-SSM", color="blue")
    plt.hlines(-1 * cvi_gpr_model.kernel.decay, 0, max(len(v_gp_prior_decay_values), len(ssm_prior_decay_values)),
               label="CVI-GPR", color="red")
    plt.title("Prior Learning")
    plt.legend()
    plt.show()


"""ELBO comparison"""
plt.hlines(gpr_log_likelihood, 0, len(v_gp_elbo)-2, color="black", label="Log Likelihood", alpha=0.2,
           linestyles="dashed")
plt.plot(ssm_elbo, label="SDE-SSM")
plt.plot(v_gp_elbo[2:], label="VGP")
plt.title("ELBO")
plt.legend()
plt.show()

"""
ELBO Bound
"""
if not learn_prior_sde:
    decay_value_range = np.linspace(0.01, decay + 0.5, 10)
    gpr_log_likelihood_vals = []
    ssm_elbo_vals = []
    vgp_elbo_vals = []
    true_q = q * tf.ones((1, 1), dtype=DTYPE)

    for decay_val in decay_value_range:
        kernel = OrnsteinUhlenbeck(decay=decay_val, diffusion=q)
        gpr_model = get_gpr(input_data, kernel, train=False, noise_stddev=noise_stddev)
        gpr_log_likelihood = gpr_model.log_likelihood().numpy()
        gpr_log_likelihood_vals.append(gpr_log_likelihood)

        ssm_model.prior_sde = OrnsteinUhlenbeckSDE(decay=decay_val, q=true_q)
        ssm_model.initial_chol_cov = tf.linalg.cholesky((q/(2 * decay_val)) * tf.ones_like(ssm_model.initial_chol_cov))
        ssm_model.fx_covs = ssm_model.initial_chol_cov.numpy().item() ** 2 + 0 * ssm_model.fx_covs
        ssm_elbo_vals.append(ssm_model.classic_elbo())

        vgp_model.prior_sde = OrnsteinUhlenbeckSDE(decay=decay_val, q=true_q)
        vgp_model.p_initial_cov = (q / (2 * decay_val)) * tf.ones((1, 1), dtype=DTYPE)
        vgp_elbo_vals.append(vgp_model.elbo())

    plt.subplots(1, 1, figsize=(5, 5))
    plt.plot(decay_value_range, ssm_elbo_vals, label="SDE-SSM")
    plt.plot(decay_value_range, vgp_elbo_vals, label="VGP")
    plt.plot(decay_value_range, gpr_log_likelihood_vals, label="Log-likelihood", alpha=0.2, linestyle="dashed",
             color="black")
    plt.vlines(decay, np.min(gpr_log_likelihood_vals), np.max(gpr_log_likelihood_vals))
    plt.legend()
    plt.show()
