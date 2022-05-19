"""OU SDE VI"""

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from gpflow import default_float
from gpflow.likelihoods import Gaussian
from gpflow import set_trainable

from markovflow.sde.sde import OrnsteinUhlenbeckSDE, PriorOUSDE
from markovflow.sde.sde_utils import euler_maruyama
from markovflow.models.gaussian_process_regression import GaussianProcessRegression
from markovflow.kernels import OrnsteinUhlenbeck
from markovflow.models.vi_sde import VariationalMarkovGP
import copy

tf.random.set_seed(83)
np.random.seed(83)
DTYPE = default_float()
plt.rcParams["figure.figsize"] = [15, 5]

"""
Parameters
"""
decay = .5  # specify without the negative sign
q = 0.2
noise_var = 0.1
x0 = 1.

t0, t1 = 0.0, 5.
num_transitions = 1000
observations_skip = 50

convergence_tol = 1e-2

learn_prior_sde = False
"""
HELPER FUNCTION
"""
def plot_model(model, predict_f=False, compare=False):

    if predict_f:
        f_mu, f_var = model.posterior.predict_y(time_grid)
        label = "GPR"
    else:
        f_mu, f_var = variational_gp.forward_pass
        f_var = (tf.reshape(f_var, (-1)) + noise_stddev**2).numpy()

        label = "Variational-GP"

    f_mu = f_mu.numpy().reshape(-1)
    f_std = np.sqrt(f_var)

    plt.plot(observation_grid.numpy().reshape(-1), simulated_values.numpy().reshape(-1), 'kx', ms=8, mew=2)
    plt.plot(time_grid.numpy().reshape(-1), f_mu.reshape(-1), ms=8, mew=2)

    plt.fill_between(
        time_grid,
        y1=(f_mu.reshape(-1) - 2 * f_std.reshape(-1)).reshape(-1,),
        y2=(f_mu.reshape(-1) + 2 * f_std.reshape(-1)).reshape(-1,),
        alpha=.2,
        label=label
    )

    plt.xlabel("Time")
    plt.ylabel("Label")
    plt.xlim([t0, t1])

    if not compare:
        plt.legend()
        plt.show()


def get_gpr_model(kernel):
    """
    This is just a function for convenience but it uses variables from outside like input data, q, noise-variance.
    """
    gpr_model = GaussianProcessRegression(input_data=(tf.constant(observation_grid),
                                                      tf.constant(tf.squeeze(simulated_values, axis=0))),
                                          kernel=kernel,
                                          chol_obs_covariance=noise_stddev * tf.eye(state_dim, dtype=DTYPE)
                                          )

    # Need to learn GPR too in this case
    if learn_prior_sde:
        opt = tf.optimizers.Adam()

        @tf.function
        def opt_step():
            opt.minimize(gpr_model.loss, gpr_model.trainable_variables)

        for _ in range(50):
            opt_step()

        print(f"After training kernel parameters are : {kernel.trainable_variables}")

    return gpr_model

"""
Generate observations for a linear SDE
"""

decay = decay * tf.ones((1, 1), dtype=DTYPE)
q = q * tf.ones((1, 1), dtype=DTYPE)
noise_stddev = np.sqrt(noise_var)

state_dim = 1
num_batch = 1
x0_shape = (num_batch, state_dim)
x0 = x0 + tf.zeros(x0_shape, dtype=DTYPE)


time_grid = tf.cast(tf.linspace(t0, t1, num_transitions), dtype=DTYPE)

# Observation at every even place
observation_grid = tf.gather(time_grid, list(np.arange(10, time_grid.shape[0], observations_skip)))

ou_sde = OrnsteinUhlenbeckSDE(decay=decay, q=q)
latent_states = euler_maruyama(ou_sde, x0, observation_grid)

# Adding observation noise
simulated_values = latent_states + tf.random.normal(latent_states.shape, stddev=noise_stddev, dtype=DTYPE)

plt.scatter(observation_grid, simulated_values, label="Observations (Y)")
plt.scatter(observation_grid, latent_states, label="Latent States (X)", alpha=0.5)
# plt.vlines(list(time_grid), -2, 2, color="red", alpha=0.1, label="Grid")
plt.xlabel("Time (t)")
plt.ylabel("y(t)")
# plt.yticks([-2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2])
plt.ylim([-2, 2])
# plt.xticks(list(time_grid))
plt.xlim([t0, t1])
plt.title("Observations")
plt.legend()
plt.show()

print(f"True decay value of the OU SDE is {decay}")
print(f"Noise std-dev is {noise_stddev}")

"""
GPR
"""
if learn_prior_sde:
    kernel = OrnsteinUhlenbeck(decay=1., diffusion=q.numpy().item())
    set_trainable(kernel.diffusion, False)
else:
    kernel = OrnsteinUhlenbeck(decay=decay.numpy().item(), diffusion=q.numpy().item())
gpr_model = get_gpr_model(kernel)
gpr_log_likelihood = gpr_model.log_likelihood().numpy()
print(f"GPR Likelihood : {gpr_log_likelihood}")

"""
Prior SDE
"""
if learn_prior_sde:
    prior_decay = -1.  # tf.random.normal((1, 1), dtype=DTYPE)
    prior_sde = PriorOUSDE(initial_val=prior_decay, q=q)
else:
    prior_sde = copy.deepcopy(ou_sde)

"""
Likelihood
"""
likelihood = Gaussian(noise_stddev**2)

"""
VGP
"""
variational_gp = VariationalMarkovGP(input_data=(observation_grid, tf.constant(tf.squeeze(simulated_values, axis=0))),
                                     prior_sde=prior_sde, grid=time_grid, likelihood=likelihood,
                                     lr=0.5)
variational_gp.p_initial_cov = (q.numpy()/(2 * decay.numpy())) * tf.ones((1, 1), dtype=DTYPE)  # For OU we know this relation for variance

v_gp_elbo = []
prior_decay_values = [prior_sde.decay.numpy().item()]
itr = 1

while itr < 10 or (0 < (v_gp_elbo[-1] - v_gp_elbo[-2]) > convergence_tol):
    variational_gp.run_inference()
    variational_gp.update_initial_statistics(lr=0.1)
    v_gp_elbo.append(variational_gp.elbo())

    itr = itr + 1
    if itr % 2 == 0:
        print(f"ELBO at iteration {itr} = {v_gp_elbo[-1]}")

    if learn_prior_sde:
        for _ in range(5):
            variational_gp.update_prior()
            prior_decay_values.append(variational_gp.prior_sde.decay.numpy().item())

plt.plot(variational_gp.lambda_lagrange.numpy().reshape(-1))
plt.title("Lambda Lagrange")
plt.show()

plt.plot(variational_gp.psi_lagrange.numpy().reshape(-1))
plt.title("Psi Lagrange")
plt.show()

plt.plot(variational_gp.A.numpy().reshape(-1))
plt.title("A")
plt.show()

plt.plot(variational_gp.b.numpy().reshape(-1))
plt.title("b")
plt.show()

"""
Compare Posterior
"""
plot_model(gpr_model, predict_f=True, compare=True)
plot_model(variational_gp)

"""
Plot drift evolution
"""
if learn_prior_sde:
    plt.hlines(-1 * decay.numpy().item(), 0, len(prior_decay_values))
    plt.plot(prior_decay_values, label="Learnt decay")
    plt.show()


"""ELBO comparison"""
plt.hlines(gpr_log_likelihood, 0, len(v_gp_elbo), color="red", label="Log Likelihood")
plt.plot(v_gp_elbo[2:], label="VGP")
plt.ylabel("ELBO")
plt.legend()
plt.show()

"""
ELBO Bound
"""
if not learn_prior_sde:
    decay_value_range = np.linspace(0.05, 1, 20)
    gpr_log_likelihood_vals = []
    vgp_elbo_vals = []

    for decay_val in decay_value_range:
        kernel = OrnsteinUhlenbeck(decay=decay_val, diffusion=q.numpy().item())
        gpr_log_likelihood_vals.append(get_gpr_model(kernel).log_likelihood().numpy())

        variational_gp.prior_sde = OrnsteinUhlenbeckSDE(decay=decay_val, q=q)
        variational_gp.p_initial_cov = (q/(2 * decay_val)) * tf.ones((1, 1), dtype=DTYPE)
        vgp_elbo_vals.append(variational_gp.elbo())

    plt.subplots(1, 1, figsize=(5, 5))
    plt.plot(decay_value_range, vgp_elbo_vals, label="VGP")
    plt.plot(decay_value_range, gpr_log_likelihood_vals, label="Log-likelihood")
    plt.vlines(decay.numpy().item(), np.min(gpr_log_likelihood_vals), np.max(gpr_log_likelihood_vals))
    # plt.xlim([-2, 2])
    plt.legend()
    plt.show()