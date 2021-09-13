# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# Factor Analysis
"""

# %%
# Debug magic:
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import os

from gpflow import default_float

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %% [markdown]
"""
## Generate the data
"""

# %% [markdown]
"""
### Generate the latents
"""

# %%
jitter = 1e-6

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from markovflow.base import ci_niter
from markovflow.kernels import Matern52, Matern32
from markovflow.ssm_natgrad import SSMNaturalGradient

kernels = [Matern52(lengthscale=1, variance=1, jitter=jitter), Matern32(lengthscale=10, variance=1, jitter=jitter)]

Bn = 1  # batch
m = 1  # output dimension per latent

N = 300  # number of datapoints
o = 5  # observation dimension output dimension
k = len(kernels)  # number of latents GPs


# %%
#  Ground truth latent function
def Gfn(X):
    G = np.empty((1, X.shape[0], k))
    G[0, :, 0] = np.sin(X)
    G[0, :, 1] = np.sin(X/5)
    return G


X_grid = 20*np.linspace(0,1, 500)

X = 20*np.sort(np.random.rand(N))
G = Gfn(X)
G_grid = Gfn(X_grid)


plt.figure()
plt.plot(X_grid, G_grid[0, ...], 'k-',alpha=.2)
plt.plot(X, G[0, ...], 'x')
    
# inducing point locations and make X and Z right shape
n_inducing = 40
Z = np.linspace(np.min(X), np.max(X), n_inducing)

# %% [markdown]
"""
### Combine the latents
"""

# %%
A = np.random.rand(Bn, o, k)
def Afn(times):
    x = np.einsum('t,...ik->...tik', times, A)
    return x


B_gen = np.random.rand(k, k)
print(B_gen.shape)

# N data, k mixture outputs, 4 latents, m outputs per latent
W = np.einsum('...ij,jk->...ik', Afn(X), B_gen)
print(W.shape)
print(G.shape)
F = np.einsum('...tik,...tk->...ti', W, G)

likelihood_variance = 0.1 # oberservation noise variance.
eta = np.random.normal(np.zeros_like(F), likelihood_variance)
Y = F + eta

print(F.shape)
print(Y.shape)

# %%
_ = plt.plot(X, F[0, ...], '-', alpha=1/np.sqrt(k))

# %%
# Plot something, maybe use B = 2, plot B graphs.
_ = plt.plot(X, Y[0, ...], 'x', alpha=1/np.sqrt(k))

# %% [markdown]
"""
We now have X as our time data, F as our FA noiseless output, and G as our latent functions we will model with a GP prior

X is shape - batch (i.e. 1) x num data
Z is shape - batch (i.e. 1) x num inducing
Y is shape - batch (i.e. k == 10) x num data

We need to learn Bn == 3 GPs and combine them
"""

# %% [markdown]
"""
## Fit the data and recover the latents
"""

# %% [markdown]
"""
### Create a SVGP model using a GPFA kernel
"""

# %%
import tensorflow as tf

from gpflow.likelihoods import Gaussian

from markovflow.models.sparse_variational import SparseVariationalGaussianProcess as SVGP
from markovflow.kernels import FactorAnalysisKernel
from markovflow.base import ci_niter

tf_X = tf.constant(np.repeat(X[None, :], Bn, axis=0), default_float())  # [Bn, N]
tf_Z = tf.constant(np.repeat(Z[None, :], Bn, axis=0), default_float())  # [Bn, N]
tf_Y = tf.constant(Y, default_float())  # [k, N, m]

tf_A = tf.constant(A)


def tf_Afn(times):
    x = tf.einsum('...t,...ik->...tik', times, tf_A)
    return x 


kernel = FactorAnalysisKernel(tf_Afn, kernels, o, True)

# Create the SVGP model
lik = Gaussian(likelihood_variance)

svgp = SVGP(kernel=kernel, inducing_points=tf_Z, likelihood=lik)

data = (tf_X, tf_Y)

print(tf_X.shape)
print(tf_Z.shape)
print(tf_Y.shape)

# %% [markdown]
"""
### Train the SVGP Factor analyis model

Everytime the ELBO starts to reduce we reduce the learning rate and continue
"""

# %%
opt_ng = SSMNaturalGradient(.99)
opt_adam = tf.optimizers.Adam(0.01)

last_elbo = tf.Variable(0., dtype=default_float())


@tf.function
def loss():
    elbo = svgp.elbo(data)
    last_elbo.assign(elbo)
    return -elbo


@tf.function
def step():
    opt_adam.minimize(loss, svgp.kernel.trainable_variables)
    opt_ng.minimize(loss, ssm=svgp.dist_q)


elbos = []
max_iter = ci_niter(100)
for it in range(max_iter):
    step()
    elbos.append(last_elbo.value())
    if it % 10 == 0:
        print(it, last_elbo.value())


plt.figure(figsize=(12, 6))
plt.plot(elbos)

# %%
last_elbo = tf.Variable(0., dtype=default_float())

# %%
last_elbo

# %% [markdown]
"""
### Generate the marginal means and variances

We do this at the training points for the latent and obervable processes
"""

# %%
x_grid = np.linspace(X.min(), X.max(), 500)
X_grid = np.repeat(x_grid[None, :], Bn, axis = 0) # [Bn, N]
tf_X_grid = tf.constant(X_grid)


posterior = svgp.posterior
f_mus, f_covs = posterior.predict_f(X_grid)

# %%
print(f_mus.shape, f_covs.shape)
f_means = f_mus[0, ...]
f_vars = f_covs[0, ...]

# %%
print(f_means.shape, f_vars.shape)


# %% [markdown]
"""
### Plot the obervable processes
"""

# %%
plt.figure(figsize=(12, 6))

f_means = f_mus[0,...]
f_vars = f_covs[0,...]

cmap = matplotlib.cm.get_cmap('viridis')
cols = cmap(np.linspace(0, 1, o))

plt.figure(figsize=(12, 6))
for ind in range(o):
    plt.plot(X,
             ind + Y[0, :, ind],
             color=cols[ind], marker='x', linestyle='none')
    
    m = f_means[..., ind]
    s = f_vars[..., ind]
    plt.plot(x_grid, ind + m, color=cols[ind], lw=2)
    lb = m - 2*np.sqrt(s)
    ub = m + 2*np.sqrt(s)
    plt.fill_between(x_grid, ind + lb, ind + ub, color=cols[ind], alpha=0.2)
# %% [markdown]
# ### Plot the latents

# %%
s_mus, s_covs = posterior.predict_state(X_grid)
latent_emission_model = svgp.kernel._latent_components.generate_emission_model(X_grid)
g_mus, g_covs = latent_emission_model.project_state_marginals_to_f(s_mus, s_covs)
g_means = g_mus[0, ...]
g_vars = g_covs[0, ...]

cmap = matplotlib.cm.get_cmap('viridis')
cols = cmap(np.linspace(0, 1, o))

plt.figure(figsize=(12, 6))
for ind in range(k):
    plt.plot(X,
             G[0, :, ind],
             color=cols[ind], marker='x', linestyle='none')
    
    m = g_means[..., ind]
    s = g_vars[..., ind]
    plt.plot(x_grid, ind + m, color=cols[ind], lw=2)
    lb = m - 2*np.sqrt(s)
    ub = m + 2*np.sqrt(s)
    plt.fill_between(x_grid, ind + lb, ind + ub, color=cols[ind], alpha=0.2)
# %%
print(g_mus.shape, g_vars.shape)
