# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Regression using a piecewise kernel
"""

# %% [markdown]
"""
This notebook explains how to use Piecewise kernels in Markovflow models.

We focus on the Sparse Variational Gaussian Process model.

Our probabilistic model for this data is:
$$
\begin{align}
f \sim \mathcal{GP}(0, k(., .)) \\
y_i \sim f(x_i) + \mathcal{N}(0, \epsilon^2)
\end{align}
$$


**NOTE:** If you have difficulty running this notebook, consider clearing the output and then restarting the kernel.
"""
# %%
# %load_ext autoreload
# %autoreload 2

# %%
# Setup

import warnings

# Turn off warnings
warnings.simplefilter('ignore')

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow

from gpflow import default_float, set_trainable
from gpflow.ci_utils import ci_niter
from gpflow.likelihoods import Gaussian

from markovflow.models.sparse_variational import SparseVariationalGaussianProcess
from markovflow.kernels import Matern52
from markovflow.kernels import PiecewiseKernel
from markovflow.ssm_natgrad import SSMNaturalGradient
FLOAT_TYPE = default_float()

# uncomment in notebook
# try:
#     from IPython import get_ipython
#     get_ipython().run_line_magic('matplotlib', 'inline')
# except AttributeError:
#     print('Magic function can only be used in IPython environment')
#     matplotlib.use('Agg')

plt.rcParams["figure.figsize"] = [15, 8]

# %% [markdown]
"""
## Step 1: Generate training data

First, let's generate a frequency modulated wave form.
"""
# %%
def create_observations(time_points: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    A helper function to create training data.
    :param time_points: Time points to generate observations for.
    :return: Tuple[x,y] Data that represents the observations' shapes:
        X = [num_points, 1],
        Y = [num_points, state_dim , 1] where state_dim is currently 1
    """
    omega_ = np.exp((time_points - 50.) / 6.) / (1. + np.exp((time_points - 50.) / 6.)) + 0.1
    y = (np.cos(time_points * omega_ / 3) * np.sin(time_points * omega_ / 3)).reshape(-1, 1)
    return time_points, y + np.random.randn(*y.shape) * .1

# Generate some observations
N = 300
time_points, observations = create_observations(np.linspace(0, 100, N))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time_points, observations, 'C0x', ms=8, mew=2)
plt.xlabel("Time")
plt.ylabel("Label")
plt.show()
plt.close()
# %% [markdown]
"""
## Step 2: Build a Piecewise kernel
"""

# %%
num_inducing = 30
step_z = N // num_inducing
num_change_points = 5
step_c = num_inducing // num_change_points

# What happens if you don't choose your inducing points from your data
inducing_points = time_points[::step_z]
inducing_points = np.linspace(np.min(time_points), np.max(time_points), num_inducing)

# What happens if you don't choose your change points from your inducing points
change_points = inducing_points[::step_c]
change_points = np.linspace(np.min(time_points), np.max(time_points), num_change_points)

assert num_change_points == len(change_points)

base = Matern52
state_dim = 3
variances = np.array([1.] * (num_change_points + 1))
lengthscales = np.array([4.] * (num_change_points + 1))

ks = [base(variance=variances[l],
            lengthscale=lengthscales[l])
      for l in range(num_change_points + 1)]
[set_trainable(k._state_mean, True) for k in ks]
kernel = PiecewiseKernel(
    ks, tf.convert_to_tensor(change_points, dtype=default_float()))

# %% [markdown]
"""
## Step 3: Build and optimise a model
"""

# %%
# Create a likelihood object
bernoulli_likelihood = Gaussian()

s2vgp = SparseVariationalGaussianProcess(
    kernel=kernel,
    inducing_points=tf.convert_to_tensor(inducing_points, dtype=default_float()),
    likelihood=bernoulli_likelihood
)

# equivalent to loss = -vgpc.elbo()
input_data = (time_points, observations)
loss = s2vgp.loss(input_data)

# Before optimisation, calculate the loss of the observations given the current kernel parameters
print("Loss before optimisation: ", loss.numpy())

# %%
# Start at a small learning rate 
adam_learning_rate = 0.005
natgrad_learning_rate = .9
max_iter = ci_niter(500)

adam_opt = tf.optimizers.Adam(learning_rate=adam_learning_rate)
natgrad_opt = SSMNaturalGradient(gamma=natgrad_learning_rate, momentum=False)
set_trainable(s2vgp.dist_q, False)
adam_var_list = s2vgp.trainable_variables
set_trainable(s2vgp.dist_q, True)


@tf.function
def loss(input_data):
    return -s2vgp.elbo(input_data)


@tf.function
def opt_step(input_data):
    natgrad_opt.minimize(lambda : loss(input_data), s2vgp.dist_q)
    adam_opt.minimize(lambda : loss(input_data), adam_var_list)

def plot_model(s2vgp):
    pred = s2vgp.posterior
    latent_mean, latent_var = pred.predict_f(tf.constant(time_points))
    predicted_mean, predicted_cov = latent_mean.numpy(), latent_var.numpy()
    # Plot the means and covariances for these future time points
    fig, ax = plt.subplots(1, 1)
    ax.plot(time_points, observations, 'C0x', ms=8, mew=2)

    ax.plot(time_points, predicted_mean, 'C0', lw=2)
    ax.fill_between(time_points,
                     predicted_mean[:, 0] - 2 * np.sqrt(predicted_cov[:, 0]),
                     predicted_mean[:, 0] + 2 * np.sqrt(predicted_cov[:, 0]),
                     color='C0', alpha=0.2)

    cp = s2vgp.kernel.change_points.numpy()
    ax.vlines(cp, ymin=-1, ymax=1, colors='blue', label='change points')
    z_ = s2vgp.inducing_inputs.numpy()
    ax.vlines(z_, ymin=-1, ymax=-.9, colors='red', label='inducing points')

    ax.set_xlim((0., 100.))
    ax.set_ylim((-1.1, 1.1))
    return fig
    # plt.show()


for i in range(max_iter):
    opt_step(input_data)
    if i % 10 == 0:
        print("Iteration:", i, ", Loss:", s2vgp.loss(input_data).numpy())
        fig = plot_model(s2vgp)
        plt.show()

print("Loss after optimisation: ", s2vgp.loss(input_data).numpy())

# Save our trained hyperparamters (these will be used in Step 8)
saved_hyperparams = kernel.trainable_variables

# %%
fig = plot_model(s2vgp)

# %% [markdown]
"""
We can see how our kernel parameters have changed from our initial values. 
"""

# %%
gpflow.utilities.print_summary(s2vgp._kernel)

# %%
