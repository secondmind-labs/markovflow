"""SDE CVI"""

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from gpflow import default_float
from gpflow.likelihoods import Gaussian

from markovflow.sde.sde import OrnsteinUhlenbeckSDE
from markovflow.sde.sde_utils import euler_maruyama
from markovflow.models.gaussian_process_regression import GaussianProcessRegression
from markovflow.kernels.matern import Matern12
from markovflow.models.vi_sde import VariationalMarkovGP

tf.random.set_seed(83)
np.random.seed(83)
DTYPE = default_float()
plt.rcParams["figure.figsize"] = [15, 5]


# HELPER FUNCTION
def plot_model(model, predict_f=False, compare=False):

    if predict_f:
        f_mu, f_var = model.posterior.predict_y(time_grid)
        label = "GPR"
    else:
        f_mu, f_var = variational_gp.forward_pass
        f_var = (f_var + noise_stddev**2).numpy()

        label = "Variational-GP"

    f_mu = f_mu.numpy()
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


# %% [markdown]
"""
## Step 1: Generate observations for a linear SDE
"""

decay = .5 * tf.ones((1, 1), dtype=DTYPE)
q = .5 * tf.ones((1, 1), dtype=DTYPE)
noise_stddev = 1e-1

state_dim = 1
num_batch = 1
x0_shape = (num_batch, state_dim)
x0 = 1 + tf.zeros(x0_shape, dtype=DTYPE)

t0, t1 = 0.0, 1.0
num_transitions = 100
time_grid = tf.cast(tf.linspace(t0, t1, num_transitions), dtype=DTYPE)

# Observation at every even place
observation_grid = tf.gather(time_grid, list(np.arange(0, time_grid.shape[0], 5)))

ou_sde = OrnsteinUhlenbeckSDE(decay=decay, q=q)
latent_states = euler_maruyama(ou_sde, x0, observation_grid)

# Adding observation noise
simulated_values = latent_states + tf.random.normal(latent_states.shape, stddev=noise_stddev, dtype=DTYPE)

plt.scatter(observation_grid, simulated_values, label="Observations (Y)")
plt.scatter(observation_grid, latent_states, label="Latent States (X)", alpha=0.5)
plt.vlines(list(time_grid), -2, 2, color="red", alpha=0.2, label="Grid")
plt.xlabel("Time (t)")
plt.ylabel("y(t)")
plt.yticks([-2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2])
plt.ylim([-2, 2])
# plt.xticks(list(time_grid))
plt.xlim([t0, t1])
plt.title("Observations")
plt.legend()
plt.show()

print(f"True decay value of the OU SDE is {decay}")
print(f"Noise std-dev is {noise_stddev}")

# %% [markdown]
"""
## Step 2: Prior SDE
"""
# %%
# prior_decay = tf.random.normal((1, 1), dtype=DTYPE)
# prior_sde = OrnsteinUhlenbeckSDE(q=q)

# %% [markdown]
"""
## Step 3: Likelihood
"""
# %%
likelihood = Gaussian(noise_stddev**2)

# %% [markdown]
"""
## Step 4: model
"""
# %%
variational_gp = VariationalMarkovGP(input_data=(observation_grid, tf.constant(tf.squeeze(simulated_values, axis=0))),
                                     prior_sde=ou_sde, grid=time_grid, likelihood=likelihood)

for i in range(10):
    variational_gp.run_inference()
    print(f"Iteration {i+1}")
    print(f"A.sum = {tf.reduce_sum(variational_gp.A)}")

plt.plot(variational_gp.lambda_lagrange.numpy().reshape(-1))
plt.title("Lambda Lagrange")
plt.show()

plt.plot(variational_gp.psi_lagrange.numpy().reshape(-1))
plt.title("Psi Lagrange")
plt.show()

"""
## Step 6: GPR
"""
kernel = Matern12(lengthscale=1., variance=1.)

gpr_model = GaussianProcessRegression(input_data=(tf.constant(observation_grid),
                                           tf.constant(tf.squeeze(simulated_values, axis=0))),
                                      kernel=kernel,
                                      chol_obs_covariance=noise_stddev * tf.eye(state_dim, dtype=DTYPE)
                                    )

iters = 10
opt = tf.optimizers.Adam()

@tf.function
def opt_step():
    opt.minimize(gpr_model.loss, gpr_model.kernel.trainable_variables)

for _ in range(iters):
    opt_step()

print(gpr_model.kernel.trainable_variables)

# Compare
plot_model(gpr_model, predict_f=True, compare=True)
plot_model(variational_gp)
