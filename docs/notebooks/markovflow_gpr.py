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
# Basic regression using the GPR model

GMRF (Gaussian Markov Random Fields) correspond to Gaussian Process models that are parametrised by the inverse of the covariance: the precision.

This notebook explains how to use Markovflow to build and optimise a GP regression model for a time series. **NOTE:** Markovflow does not require that the observations in a time series are regularly spaced.
"""
# %%
# Setup
import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf

# Turn off warnings
from markovflow.kernels import Matern32
from markovflow.models import GaussianProcessRegression
from markovflow.base import default_float, ci_niter

FLOAT_TYPE = default_float()

warnings.simplefilter('ignore')

try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except AttributeError:
    print('Magic function can only be used in IPython environment')
    matplotlib.use('Agg')

plt.rcParams["figure.figsize"] = [15, 8]

# %% [markdown]
"""
## Step 1: Generate training data
Usually it is a good idea to normalise the data, because by default most kernels revert to a mean of zero.
"""
# %%
def create_observations(time_points: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    A helper function to create training data.
    :param time_points: Points in time.
    :return: Tuple[x,y] Data that represents the observation shapes:
        X = [num_points, 1],
        Y = [num_points, state_dim , 1] where state_dim is currently 1
    """
    observations = np.sin(12 * time_points[..., None])
    observations += np.random.randn(len(time_points), 1) * 0.1
    observations += 3
    return time_points, observations

# Generate some observations
time_points, observations = create_observations(np.arange(5.0, 20.0))

# Usually it is a good idea to normalise the data, because by default most kernels revert to a mean of zero
norm_observations = (observations - np.mean(observations)) / np.std(observations)

plt.plot(time_points.squeeze(), norm_observations.squeeze(), 'C0x')
plt.xlim((0., 30.))
plt.ylim((-3., 3.))
plt.show()


# %% [markdown]
"""
## Step 2: Choose a kernel and create the model
Markovflow provides several SDE (Stochastic Differential Equation) kernels. Your domain knowledge is encoded in your choice of kernel, or combination of kernels. For more information, see
[Choosing and combining kernels](./choosing_and_combining_kernels.ipynb).

In this example we use a Matern 1/2 kernel. However, given our knowledge of the data, a periodic kernel might be more appropriate. Why not try it?

We also use an observation covariance of 0.001. The observation covariance is the amount of noise that we believe exists in our observations (the measurement noise).

**NOTE:** In this example `observation_covariance` is not set as trainable.  This is the equivalent of specifying a specific value for the measurement noise. Try varying the magnitude of this parameter, or making it trainable.
"""
# %%
# Add some noise to the observations
observation_covariance = tf.constant([[0.0001]], dtype=FLOAT_TYPE)

# Create a GPR model
kernel = Matern32(lengthscale=8.0, variance=1.0)
input_data = (tf.constant(time_points), tf.constant(norm_observations))
gpr = GaussianProcessRegression(input_data=input_data, kernel=kernel,
                                chol_obs_covariance=tf.linalg.cholesky(observation_covariance))

# %% [markdown]
"""
We can calculate the marginal likelihood (that is, the probability of the observed data given the model) before we optimise the model.
**NOTE:** We are using log likelihood, so a probability of 1.0 (the data definitely came from the model) is equal to a log likelihood of zero (with lower probabilities increasingly negative).
"""
# %%
# Before optimisation, calculate the log likelihood of the observations given the current kernel parameters 
print(gpr.log_likelihood())

# %% [markdown]
"""
After optimisation, the probability of the data given the model should have increased (that is, the log likelihood should have increased).
"""
# %%
opt = tf.optimizers.Adam()

@tf.function
def opt_step():
    opt.minimize(gpr.loss, gpr.trainable_variables)

max_iter = ci_niter(4000)
for _ in range(max_iter):
    opt_step()

print(gpr.log_likelihood())


# %% [markdown]
"""
## Step 3: Generate a mean for the training data
We can use the model's posterior `predict_f` function to get the mean of the true function values (observations without noise).
"""
# %%
mean, _ = gpr.posterior.predict_f(gpr.time_points)
# Plot the results
plt.plot(time_points, norm_observations, 'C0x')
plt.plot(time_points, mean, mew=2)
plt.xlim((0., 30.))
plt.ylim((-3., 3.))
plt.show()

# %% [markdown]
"""
## Step 4: Make a prediction for the future
The GPR model's `posterior` supports interpolation and extrapolation of the underlying state-space model.

For example, the `gpr.posterior.predict_f` function predicts means and covariances for the specified time points.
"""
# %%
# Generate some time points in the future
future_time_points = np.arange(time_points[-1] + 0.01, time_points[-1] + 10.0, 0.1)

predicted_mean, predicted_cov = \
    gpr.posterior.predict_f(tf.constant(future_time_points, dtype=FLOAT_TYPE))
predicted_mean, predicted_cov = predicted_mean.numpy(), predicted_cov.numpy()

# Plot the means and covariances for these future time points
plt.plot(time_points, mean, mew=2)
plt.plot(future_time_points, predicted_mean, 'C0', lw=2)
plt.fill_between(future_time_points,
                 predicted_mean[:, 0] - 2 * np.sqrt(predicted_cov[:, 0]),
                 predicted_mean[:, 0] + 2 * np.sqrt(predicted_cov[:, 0]),
                 color='C0', alpha=0.2)
plt.xlim((0., 30.))
plt.ylim((-3., 3.))
plt.show()

# %% [markdown]
"""
The `gpr.posterior.sample_f` function samples from the posterior probability distribution. Note the variance of the initial points of the generated sampled trajectories. This is a result of the observation covariance we specified earlier.
"""
# %%
samples = gpr.posterior.sample_f(tf.constant(future_time_points, dtype=FLOAT_TYPE), 50)

# Plot the same as previous
plt.plot(time_points, mean, mew=2)
plt.plot(future_time_points, predicted_mean, 'C0', lw=2)
plt.fill_between(future_time_points,
                 predicted_mean.squeeze() - 2 * np.sqrt(predicted_cov.squeeze()),
                 predicted_mean.squeeze() + 2 * np.sqrt(predicted_cov.squeeze()),
                 color='C0', alpha=0.2)
# Add the samples
plt.plot(future_time_points[..., None], np.swapaxes(samples, 0, 1).squeeze())
plt.xlim((0., 30.))
plt.ylim((-3., 3.))
plt.show()

# %% [markdown]
"""
## Step 5: Show a history of confidence levels
The `gpr.posterior.predict_f` gets the posterior of the latent function at arbitrary time points.
To demonstrate this, we can generate a set of time points that begin before the training data and
extend them into the future. Note how the model is very certain about the fit in the region where there
is data (the confidence intervals are small), whereas the uncertainty grows when we predict in the future.
"""
# %%
# Generate some other time points to evaluate
intermediate_time_points = np.arange(0.0, time_points[-1] + 10.0, 0.1)
predicted_mean, predicted_cov = \
    gpr.posterior.predict_f(tf.constant(intermediate_time_points, dtype=FLOAT_TYPE))
predicted_mean, predicted_cov = predicted_mean.numpy(), predicted_cov.numpy()

# Plot the results
plt.plot(intermediate_time_points, predicted_mean, 'C0', lw=2)
plt.fill_between(intermediate_time_points,
                 predicted_mean.squeeze() - 2 * np.sqrt(predicted_cov.squeeze()),
                 predicted_mean.squeeze() + 2 * np.sqrt(predicted_cov.squeeze()),
                 color='C0', alpha=0.2)
plt.xlim((0., 30.))
plt.ylim((-3., 3.))
plt.show()

# %% [markdown]
"""
## Step 6: Observe more data in the future
When new data becomes available, we can see how the variance collapses (the confidence increases) at the new point (`t=23`).
"""
# %%
new_time, new_ob = create_observations(np.array([23.]))
# Normalise the new point based on the original observation's mean and std
new_ob = (new_ob - np.mean(observations)) / np.std(observations)
time_points = np.concatenate([time_points, new_time], axis=0)
new_observations = np.concatenate([norm_observations, new_ob], axis=0)

gpr._time_points = tf.constant(time_points)
gpr._observations = tf.constant(new_observations)

predicted_mean, predicted_cov = \
    gpr.posterior.predict_f(tf.constant(intermediate_time_points, dtype=FLOAT_TYPE))
# Plot the results
plt.plot(intermediate_time_points, predicted_mean, 'C0', lw=2)
plt.fill_between(intermediate_time_points,
                 predicted_mean[:, 0] - 2 * np.sqrt(predicted_cov[:, 0]),
                 predicted_mean[:, 0] + 2 * np.sqrt(predicted_cov[:, 0]),
                 color='C0', alpha=0.2)
plt.xlim((0., 30.))
plt.ylim((-3., 3.))
plt.show()
