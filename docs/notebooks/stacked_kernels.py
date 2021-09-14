# ---
# jupyter:
#   jupytext:
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

# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.ci_utils import ci_niter

from gpflow.likelihoods import Gaussian

from markovflow.models import SparseVariationalGaussianProcess
from markovflow.kernels import Matern12, Matern32
from markovflow.kernels.sde_kernel import IndependentMultiOutputStack
from markovflow.ssm_natgrad import SSMNaturalGradient

# %% [markdown]
# # Stacked kernels and multiple outputs
# This notebook is about _stacked kernels_, which is one way to get multiple outputs in MarkovFlow. 
#
# Stacked kernels use a leading 'batch' dimension to compute multiple kernels together. Conceptually, if a kernel matrix is of dimensions `[N x N]`, then a stacked kernel produces an object of shape `[S x N x N]`. All of the markovflow computations will have this extra leading dimension. For example the state-transition matrices will be of shape `[S, T, D, D]`, where T is the number of time points and D is the state dimension. 
#
# The _data_, however, are expected to be of shape `[N x S]`, so the `S` dimension should follow, not lead. This convention makes the stacked kernel compatible with likelihoods that can handle multiple outputs and processes. 
#
# The advantage of this approach to multiple-outputs is that it is computationally efficient, because all computations can loop over this leading `S` dimension instead of augmenting the state dimension of the process. However, using a similar parameterization as an approximate posterior in an inference problem is a bit restrictive since it forces the posterior processes to be independent which may not always be an appropriate assumption.
#  
# You may also be interesed in another style of multiple output kernels in MarkovFlow, where the state-dimensions of the processes are concatenated. In that case, the computational complexity grows cubically with the number of outputs, since the state dimension is growing. See the factor_analysis notebook.

# %% [markdown]
# ## A simple example
#
# We'll build a model with two outputs using a stacked kernel. We use the sparse GP object from markovflow to do inference.

# %%
# constants
num_data = 300
num_inducing = 50
num_outputs = 2
lengthscales = [0.05, 0.05]

# %%
# construct a simple data set with correlated noise
X = np.linspace(0, 1, num_data)
X_tf = tf.broadcast_to(X, (num_outputs, num_data)) # duplicate
F = np.hstack([np.sin(10 * X)[:, None], np.cos(15 * X)[:, None]])
Sigma = np.array([[0.1, 0.08], [0.08, 0.1]])
noise = np.random.multivariate_normal(np.zeros(2), Sigma, num_data)
Y = F + noise
data = (X_tf, tf.convert_to_tensor(Y))

# %%
# constuct a stacked kernel with two outputs
k1 = Matern12(lengthscale=lengthscales[0], variance=1.)
k2 = Matern32(lengthscale=lengthscales[1], variance=1.)
kern = IndependentMultiOutputStack([k1, k2], jitter=1e-6)

# construct a model
lik = Gaussian(1.)
Z = tf.broadcast_to(np.linspace(0, 1, num_inducing), (num_outputs, num_inducing))
m = SparseVariationalGaussianProcess(kern, lik, Z)


# %%
def plot():
    # plot the data with predictions:
    p = m.posterior
    mu, var = p.predict_y(X_tf)
    for i in [0, 1]:
        plt.plot(X, Y[:, i], f'C{i}x', alpha=0.5)
        plt.plot(X, mu[:, i], f'C{i}')
        std = tf.math.sqrt(var[:, i])
        plt.plot(X, mu[:, i] + 2 * std, f'C{i}--')
        plt.plot(X, mu[:, i] - 2 * std, f'C{i}--')

plot()
_ = plt.title('The model fit before optimization')


# %%
def optimize(model):
    # we'll use the natural gradient optimizer for the variational parameters and the Adam optimizer for hyper-parameters
    opt_ng = SSMNaturalGradient(.5)
    opt_adam = tf.optimizers.Adam(0.05)

    @tf.function
    def step():
        opt_adam.minimize(lambda : -model.elbo(data), model._likelihood.trainable_variables + model._kernel.trainable_variables)
        opt_ng.minimize(lambda : -model.elbo(data), ssm=m.dist_q)
    
    @tf.function
    def elbo():
        return m.elbo(data)
    
    max_iter = ci_niter(400)
    for i in range(max_iter):
        step()
        if i % 50 == 0:
            print(f"Iteration {i}, elbo:{elbo().numpy():.4}")


# %%
optimize(m)

# %%
plot()
_ = plt.title('The model fit after optimization')

# %% [markdown]
# ## What about broadcasting?
# Since we're using a leading dimension in the stacked kernel, we might be worried about whether this impedes markovflow's ability to fit a model to multiple independent datasets. Fear not! extra leading dimensions are still handled (and looped over appropriately), and parameter sharing of the kernels (between datasets, not outputs) still happens smoothly. 
#
# In this example, we'll fit a model with a heteroskedastic likelihood to multiple datasets simultaneously. The likelihood requires two GP outputs to model a *single* data column. One of the GPs models the mean of the data, and the other models the variance. We'll generate multiple datasets, construct the outline of a very simple likelihood and fit the whole shebang in a single markovflow model. Each dataset gets its own GPs, but the kernels paraemters are shared amongst datasets. 

# %%
num_data = 300
num_datasets = 2
num_inducing = 30
num_gp_outputs = 2
num_data_outputs = 1
lengthscales = [0.05, 0.5]

# %%
# generate datasets from sinusoidal means and time varying noise variances (exponentiated sinusoids)
Xs, Ys = [], []
for d in range(num_datasets):
    X = np.linspace(0, 1, num_data)
    amplitudes = np.random.rand(2) * np.array([1, 0.5]) + np.array([3, 2])
    phases = np.random.randn(2) * 2 * np.pi
    frequencies =  np.array([10, 2])
    f1, f2 = [np.sin(2*np.pi * X * omega + phi) * a for omega, phi, a in zip(frequencies, phases, amplitudes)]
    Y = f1 + np.random.randn(*f2.shape) * np.exp(0.5 * f2)
    Ys.append(Y.reshape(num_data, num_data_outputs))
    Xs.append(tf.broadcast_to(X, (num_gp_outputs, num_data)))

Xs = tf.convert_to_tensor(Xs)  # [num_datasets, num_gps, num_data]
Ys = tf.convert_to_tensor(Ys)  # [num_datasets, num_data, num_data_outputs]
data = (Xs, Ys)

# %%
from markovflow.likelihoods import Likelihood
import tensorflow_probability as tfp

class HetGaussian(Likelihood):
    def log_probability_density(self, f, y):
        mu, logvar = f[..., 0], f[..., 1]
        return tfp.distributions.Normal(loc=mu, scale=tf.exp(0.5 * logvar)).log_p(y[..., 0])
    
    def variational_expectations(self, f_means, f_covariances, observations):
        f1, f2 = f_means[..., 0], f_means[..., 1]
        variances = f_covariances # assume independent GPs
        v1, v2 = variances[..., 0], variances[..., 1]
        return -0.5 * (np.log(2*np.pi) + f2 + 
                       tf.exp(-f2 + 0.5 * v2) * (tf.square(f1 - observations[..., 0]) + v1))
    def predict_density(self, f_means, f_covariances, observations):
        raise NotImplementedError

    def predict_mean_and_var(self, f_means, f_covariances):
        raise NotImplementedError


# %%
# constuct a stacked kernel with two outputs
k1 = Matern32(lengthscale=.05, variance=1.)
k2 = Matern12(lengthscale=.5, variance=1.)
kern = IndependentMultiOutputStack([k1, k2])

# construct a model
lik = HetGaussian()
Z = tf.broadcast_to(np.linspace(0, 1, num_inducing), (num_datasets, num_gp_outputs, num_inducing))
m = SparseVariationalGaussianProcess(kern, lik, Z)

print(m.elbo(data))
# %%
from gpflow.optimizers import Scipy
opt = Scipy()
opt.minimize(lambda: -m.elbo(data), m.trainable_variables, options=dict(maxiter=ci_niter(1000)))

# %%
mus, _ = m.posterior.predict_f(Xs)

f, axes = plt.subplots(num_datasets, 1, sharex=True, sharey=True, figsize=(8, 6))
for i, (Y, ax, mu) in enumerate(zip(Ys, axes, mus)):
    ax.plot(Y, 'C0.', alpha=0.3)
    ax.plot(mu[:, 0], 'C1')
    ax.plot(mu[:, 0] + 2 * tf.exp(0.5 * mu[:, 1]), 'C1--')
    ax.plot(mu[:, 0] - 2 * tf.exp(0.5 * mu[:, 1]), 'C1--')
    ax.set_title(f'dataset {i}')
