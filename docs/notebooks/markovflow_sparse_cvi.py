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
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Basic classification using the SparseCVIGaussianProcess model
"""

# %% [markdown]
"""
This notebook explains how to perform GP inference using the Markovflow CVIGaussianProcess model.
Here, we perform binary classification with time as the input.

As with GPR, the observations do not have to be regularly spaced. However, they do need to be sequential. We denote the input/output tuples as $(x_i, y_i)_{1 \leq i \leq n}$, where $x_i$ is a scalar value and $y_i \in \{0, 1\}$.

Our probabilistic model for this data is:
$$
\begin{align}
f \sim \mathcal{GP}(0, k(., .)) \\
y_i \sim \mathcal{B}(\Phi(f(x_i)))
\end{align}
$$

where $\Phi$ is a function that maps $f(x_i)$ to $[0, 1]$, the probability that $y_i=1$. In practice, we choose $\Phi$ to be the standard normal cumulative distribution function (also known as the probit function) which maps to $[0, 1]$.

**NOTE:** If you have difficulty running this notebook, consider clearing the output and then restarting the kernel.
"""
# %%
# Setup

import warnings

# Turn off warnings
warnings.simplefilter('ignore')


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from gpflow.likelihoods import Bernoulli

from markovflow.models.sparse_variational_cvi import SparseCVIGaussianProcess
from markovflow.kernels import Matern52
from markovflow.base import default_float, ci_niter


np.random.seed(0)
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

First, let's generate some binary data $X = (x_1, \dots, x_n)$ and $Y = (y_1, \dots, y_n)^T$.
"""
# %%
# Generate some observations
num_data  = 300
num_inducing = 30

time_points = np.linspace(0 , 1, num_data)
inducing_points = np.linspace(0 , 1, num_inducing)
F = np.cos(time_points * 20).reshape(-1, 1)
observations = (F + np.random.randn(*F.shape) > 0).astype(float)
data = (time_points, observations)


# %% [markdown]
"""
## Step 2: Choose a kernel
"""
# %%

kernel = Matern52(lengthscale=.2, variance=5.0)

# We see Matern12 has only two dimensions (therefore there is less risk of overparameterising)
print(kernel.state_dim)

# %% [markdown]
"""
## Step 3: Build and optimise a model

This is a classification problem with outputs between `[0,1]`, so we create a variational GP model using a Bernoulli likelihood.
"""
# %%

# Create a likelihood object
likelihood = Bernoulli()

scvi = SparseCVIGaussianProcess(kernel=kernel,
                          inducing_points=tf.constant(inducing_points),
                          likelihood=likelihood,
                          learning_rate=.1)


def plot_model(model):

    f_mu, f_var = model.posterior.predict_f(time_points)
    f_mu = f_mu.numpy()
    f_std = np.sqrt(f_var)

    plt.figure(figsize=(10, 6))
    plt.vlines(inducing_points,
               ymin=observations.min(), ymax=observations.max(),
               color='r', label='inducing points')
    plt.plot(time_points, observations, 'kx', ms=8, mew=2, label='data')
    plt.plot(time_points, F, 'b', ms=8, mew=2, label='underlying $f$')
    plt.plot(time_points, f_mu, 'C0', ms=8, mew=2, label='posterior prediction')
    plt.fill_between(
        time_points,
        y1 = (f_mu - 2 * f_std).reshape(-1,),
        y2 = (f_mu + 2 * f_std).reshape(-1,),
        alpha=.2, facecolor='C0'
    )
    plt.xlabel("Time")
    plt.ylabel("Label")
    plt.legend()
    plt.show()


plot_model(scvi)

max_iter = ci_niter(400)
for i in range(max_iter):
    if i % 20 == 0:
        print(i, scvi.classic_elbo(data))
    scvi.update_sites(data)
plot_model(scvi)


