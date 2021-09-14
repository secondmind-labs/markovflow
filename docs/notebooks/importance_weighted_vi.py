# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Classification using importance-weighted SGPR

This notebook explains how to use Markovflow to build and optimise a GP classifier (in 1D of
course!) using importance-weighted variational inference.
"""
# %%
import numpy as np
import tensorflow as tf
from gpflow.ci_utils import ci_niter
from gpflow.likelihoods import Bernoulli

from markovflow.models.iwvi import ImportanceWeightedVI
from markovflow.kernels import Matern32

import matplotlib.pyplot as plt
# %%
# Setup
learning_rate = 1e-3
importance_K = 10

# toy data
num_data = 100
time_points = np.linspace(0, 10, num_data).reshape(-1,)
observations = np.cos(2*np.pi * time_points / 3.).reshape(-1, 1) + np.random.randn(num_data, 1) * .8
observations = (observations > 0).astype(float)
data = (tf.convert_to_tensor(time_points), tf.convert_to_tensor(observations))

# %%
# model setup
num_inducing = 20
inducing_points = np.linspace(-1, 11, num_inducing).reshape(-1,)
kernel = Matern32(lengthscale=2.0, variance=4.0)
likelihood = Bernoulli()
m = ImportanceWeightedVI(kernel=kernel,
                         inducing_points=tf.constant(inducing_points, dtype=tf.float64),
                         likelihood=likelihood,
                         num_importance_samples=importance_K)


# %%
# optimizer setup
variational_variables = m.dist_q.trainable_variables
hyperparam_variables = m.kernel.trainable_variables
adam_variational = tf.optimizers.Adam(learning_rate)
adam_hyper = tf.optimizers.Adam(learning_rate)

_dregs = lambda: -m.dregs_objective(data)
_iwvi_elbo = lambda: -m.elbo(data)

@tf.function
def step():
    adam_variational.minimize(_dregs, var_list=variational_variables)
    adam_hyper.minimize(_iwvi_elbo, var_list=hyperparam_variables)

@tf.function
def elbo_eval():
    return m.elbo(data)


# %%
# a function to plot the data and model fit

def plot(model):

    time_grid = np.linspace(0, 10, 200).reshape(-1,)

    num_samples = 50
    samples_q_s = model.posterior.proposal_process.sample_state(time_grid, num_samples)
    samples_iwvi = model.posterior.sample_f(time_grid, num_samples, input_data=data)

    _, axarr = plt.subplots(2, 1, sharex=True, sharey=True)
    # plot data
    axarr[0].plot(time_points, observations, 'kx')
    axarr[0].set_title('proposal')
    axarr[0].plot(time_grid, samples_q_s[..., 0].numpy().T, alpha=.1, color='red')

    axarr[1].plot(time_points, observations, 'kx')
    axarr[1].set_title('importance-weighted')
    axarr[1].plot(time_grid, samples_iwvi[..., 0].numpy().T, alpha=.1, color='blue')
    axarr[1].set_ylim(-1.5, 2.5)

    # plot mean by numerically integrating the iwvi posterior
    eps = 1e-3
    inv_link = lambda x : eps + (1-eps) * likelihood.invlink(x)
    probs = m.posterior.expected_value(time_grid, data, inv_link)
    axarr[1].plot(time_grid, probs, color='black', lw=1.6)

# %%
plot(m)

# %%
# the optimisation loop
elbos, elbo_stds = [], []
max_iter = ci_niter(2000)
for i in range(max_iter):
    step()
    if i % 10 == 0:
        elbos_i = [elbo_eval().numpy() for _ in range(10)]
        elbos.append(np.mean(elbos_i))
        elbo_stds.append(np.std(elbos_i))
        print(i, elbos[-1], elbo_stds[-1])

# %%
plot(m)
