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
# # Demo of MultiStageLikelihood with plain SVGP model
#
# We demonstrate a MultiStageLikelihood driven by a multi-output SVGP model. We fit the model to samples from the
# MultiStageLikelihood given toy functions from a Gaussian process draw.
#

# %%
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import set_trainable
from gpflow.ci_utils import ci_niter
from matplotlib import pyplot as plt

# %%
import markovflow.kernels as mfk
from markovflow.models.sparse_variational import SparseVariationalGaussianProcess as SVGP
from markovflow.ssm_natgrad import SSMNaturalGradient
from markovflow.likelihoods.mutlistage_likelihood import MultiStageLikelihood

# %% [markdown]
# ## Generate artificial data
#
# We draw toy functions for the three likelihood parameters from a Gaussian process.

# %%
N = 100  # number of training points
X_train = np.arange(N).astype(float)
L = 3  # number of latent functions

# %%
# Define the kernel
k1a = gpflow.kernels.Periodic(
    gpflow.kernels.Matern52(variance=1.0, lengthscales=3.0), period=12.0
)
k1b = gpflow.kernels.Matern52(variance=1.0, lengthscales=30.0)
k2 = gpflow.kernels.Matern32(variance=0.1, lengthscales=5.0)
k = k1a * k1b + k2

# Draw three independent functions from the same Gaussian process
X = X_train
num_latent = L
K = k(X[:, None])
np.random.seed(123)
v = np.random.randn(len(K), num_latent)
# We draw samples from a GP with kernel k(.) evaluated at X by reparameterizing:
# f ~ N(0, K) → f = chol(K) v, v ~ N(0, I), where chol(K) chol(K)ᵀ = K
f = np.linalg.cholesky(K + 1e-6 * np.eye(len(K))) @ v

# We shift the third function to increase the mean of the Poisson component to 20 to make it easier to identify
f += np.array([0.0, 0.0, np.log(20)]).reshape(1, L)

# Plot all three functions
plt.figure()
for i in range(num_latent):
    plt.plot(X, f[:, i])
_ = plt.xticks(np.arange(0, 100, 12))

# %% [markdown]
# The above latent GPs represent how the three likelihood parameters will change over time.

# %%
# Define the likelihood
lik = MultiStageLikelihood()
# Draw observations from the likelihood given the functions `f` from the previous step
Y = lik.sample_y(tf.convert_to_tensor(f, dtype=gpflow.default_float()))

# %%
# Plot the observations
_ = plt.plot(X, Y, ".")

# %%
Y_train = Y
data = (X_train, Y_train)


# %% [markdown]
# ## Create the model
# (Note: as we are replicating the modelling task, we pretend we don't know the underlying processes that created the artificial data.)
#
# We decide to create 3 GPs each with an independent Matern kernel, as we need three functions to drive the three parameters of the likelihood.
# This will correspond to learning 6 hyperparameters (3 GPs with 2 hyper-parameters each).

# %%
# Create kernels
kern_list = [mfk.Matern32(variance=1.0, lengthscale=10.0, jitter=1e-6) for _ in range(L)]


# %%
# Create multi-output kernel from kernel list
ker = mfk.IndependentMultiOutput(kern_list)

# %%
# Create evenly spaced inducing points
num_inducing = N // 10
Z = np.linspace(X_train.min(), X_train.max(), num_inducing)

# %%
# create multi-output inducing variables from Z
inducing_variable = tf.constant(Z)


# %%
likelihood = MultiStageLikelihood()

# %%
model = SVGP(
    kernel=ker, likelihood=likelihood, inducing_points=inducing_variable, mean_function=None,
)

# %%
X_grid = X

# %% [markdown]
# ## Optimise the model
#
# NatGrads and Adam for SVGP

# %%
adam_learning_rate = 0.001
natgrad_learning_rate = 0.05

# %%
adam_opt = tf.optimizers.Adam(learning_rate=adam_learning_rate)
natgrad_opt = SSMNaturalGradient(gamma=natgrad_learning_rate, momentum=False)

# Stop Adam from optimizing the variational parameters

set_trainable(model.dist_q, False)
adam_var_list = model.trainable_variables
set_trainable(model.dist_q, True)

# %% [markdown]
# We separate the training process into hyper-parameters and variational parameters, which in practice, can result in better training.
#
# For the variational parameters we can use much larger step sizes using natural gradients (whereas the hyper-parameters
# take smaller steps using Adam).
#
# This results in optimisation being more efficient (i.e. faster and better results).

# %%
# Variables optimized by the Adam optimizer:
print(adam_var_list)


# %%
@tf.function
def model_loss():
    return -model.elbo(data)


# %%
print(model_loss().numpy())


# %%
@tf.function
def step():

    # first take step with hyper-parameters with Adam
    adam_opt.minimize(model_loss, var_list=adam_var_list)
    # then variational parameters with NatGrad
    natgrad_opt.minimize(model_loss, model.dist_q)


# %%
@tf.function
def sample_y(X, num_samples, correlated: bool = False):
    if correlated:
        # this path may give Cholesky errors
        f_samples = model.posterior.sample_f(X, num_samples)
    else:
        f_mean, f_var = model.posterior.predict_f(X)
        f_samples = tfp.distributions.Normal(f_mean, tf.sqrt(f_var)).sample(num_samples)
    return likelihood.sample_y(f_samples)


# %%
# the arguments to a tf.function-wrapped function need to be Tensors, not numpy arrays:
X_grid_tensor = tf.convert_to_tensor(X_grid)

# %%
iterations = ci_niter(100)

# run the optimization
for it in range(iterations):
    if it % 10 == 0:

        plt.figure()
        y_samples = sample_y(X_grid_tensor, 1000)
        lower, upper = np.quantile(y_samples, q=[0.05, 0.95], axis=0).squeeze(-1)
        plt.plot(X_grid, lower, "b-")
        plt.plot(X_grid, upper, "b-")
        # f_samples = model.posterior.sample_f(X_grid, 10)
        # plt.plot(X_grid, f_samples[..., 0].numpy().T, 'b-')
        plt.plot(X_train, Y_train, "k.")
        # plt.savefig("test_%03d.png" % it)
        # plt.close()

    step()
    print(it, model_loss())

# %% [markdown]
# ## Compare inferred functions with ground truth

# %%
Fmean, Fvar = model.predict_f(X_grid)

# %%
for i in range(num_latent):
    plt.plot(X_grid, Fmean[:, i], f"C{i}-")
    plt.fill_between(
        X_grid,
        Fmean[:, i] - 2 * tf.sqrt(Fvar[:, i]),
        Fmean[:, i] + 2 * tf.sqrt(Fvar[:, i]),
        color=f"C{i}",
        alpha=0.3,
    )

    plt.plot(X, f[:, i], f"C{i}--")

# Custom legend:
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.legend(
    loc="upper right",
    handles=[
        Line2D([0], [0], color="k", ls="--", label="Ground truth"),
        Line2D([0], [0], color="k", label="Inferred mean"),
        Patch(facecolor="k", alpha=0.3, label="Inferred uncertainty"),
    ],
)

# %% [markdown]
# The above plot shows that our linear combination of GPs with Matern kernels has done a reasonable job of representing the ground truth.
