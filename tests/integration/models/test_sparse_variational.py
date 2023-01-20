#
# Copyright (c) 2021 The Markovflow Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Module containing the integration tests for the `SparseVariationalGaussianProcess` class."""
from typing import Callable, List, Tuple

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from gpflow.likelihoods import Bernoulli, Gaussian

from markovflow.kernels import Matern12
from markovflow.mean_function import LinearMeanFunction
from markovflow.models import GaussianProcessRegression, SparseVariationalGaussianProcess
from markovflow.models.variational import VariationalGaussianProcess
from tests.tools.generate_random_objects import generate_random_time_observations

OUT_DIM = 1
LENGTH_SCALE = 2.0
VARIANCE = 2.25
NUM_DATA = 8


@pytest.fixture(name="svgp_gpr_optim_setup")
def _svgp_gpr_optim_setup(batch_shape, output_dim):
    time_points, observations, kernel, variance = _setup(batch_shape, output_dim)

    # nasty hack because tensorflow complains about uninitialised variables when passed as params.
    kernel._lengthscale = tf.constant(LENGTH_SCALE, dtype=tf.float64)
    kernel._variance = tf.constant(VARIANCE, dtype=tf.float64)

    chol_obs_covariance = tf.eye(output_dim, dtype=tf.float64) * variance
    input_data = (tf.constant(time_points), tf.constant(observations))
    gpr = GaussianProcessRegression(
        input_data=input_data, kernel=kernel, chol_obs_covariance=chol_obs_covariance
    )

    likelihood = Gaussian(variance=variance)
    svgp = SparseVariationalGaussianProcess(
        kernel=kernel,
        likelihood=likelihood,
        inducing_points=time_points,
        initial_distribution=gpr.posterior.gauss_markov_model,
    )

    # do not train any hyper-parameters for these tests
    for t in likelihood.trainable_variables + kernel.trainable_variables:
        t._trainable = False

    return svgp, gpr


def _svgp_gpr_setup(batch_shape, output_dim, mean_function):
    time_points, observations, kernel, variance = _setup(batch_shape, output_dim)

    chol_obs_covariance = tf.eye(output_dim, dtype=gpflow.default_float()) * tf.sqrt(variance)
    input_data = (tf.constant(time_points), tf.constant(observations))
    gpr = GaussianProcessRegression(
        input_data=input_data,
        kernel=kernel,
        mean_function=mean_function,
        chol_obs_covariance=chol_obs_covariance,
    )

    likelihood = Gaussian(variance)
    svgp = SparseVariationalGaussianProcess(
        kernel=kernel,
        likelihood=likelihood,
        inducing_points=time_points,
        mean_function=mean_function,
    )

    # do not train any hyper-parameters for these tests
    for t in likelihood.trainable_variables + kernel.trainable_variables:
        t._trainable = False

    return svgp, gpr


def _svgp_vgp_setup(batch_shape, output_dim):
    time_points, observations, kernel, _ = _setup(batch_shape, output_dim)

    likelihood = Bernoulli()
    vgp = VariationalGaussianProcess(
        input_data=(time_points, observations), kernel=kernel, likelihood=likelihood
    )

    svgp = SparseVariationalGaussianProcess(
        kernel=kernel, likelihood=likelihood, inducing_points=time_points
    )

    # do not train any hyper-parameters for these tests
    for t in likelihood.trainable_variables + kernel.trainable_variables:
        t._trainable = False

    return svgp, vgp


def _setup(batch_shape: Tuple, output_dim: int):
    time_points, observations = generate_random_time_observations(
        obs_dim=output_dim, num_data=NUM_DATA, batch_shape=batch_shape
    )
    time_points = tf.constant(time_points, dtype=gpflow.default_float())
    observations = tf.constant(observations, dtype=gpflow.default_float())

    kernel = Matern12(lengthscale=LENGTH_SCALE, variance=VARIANCE, output_dim=output_dim)

    observation_noise = 1.0
    variance = tf.constant(observation_noise, dtype=gpflow.default_float())

    return time_points, observations, kernel, variance


def test_svgp_elbo_optimal(with_tf_random_seed, svgp_gpr_optim_setup):
    """Test that the value of the ELBO at the optimum is the same as the GPR Log Likelihood."""
    svgp, gpr = svgp_gpr_optim_setup

    input_data = (gpr._time_points, gpr._observations)
    np.testing.assert_allclose(svgp.elbo(input_data), gpr.log_likelihood())


def test_svgp_grad_optimal(with_tf_random_seed, svgp_gpr_optim_setup):
    """Test that the gradient of the ELBO at the optimum is zero."""
    svgp, gpr = svgp_gpr_optim_setup

    input_data = (gpr._time_points, gpr._observations)
    print(svgp.trainable_variables)

    with tf.GradientTape() as tape:
        elbo = svgp.elbo(input_data)
    eval_grads = tape.gradient(elbo, svgp.trainable_variables)
    for grad in eval_grads:
        np.testing.assert_allclose(grad, 0.0, atol=1e-9)


def _test_svgp_vs_gpr(svgp, gpr, tol):
    """
    Test that the SVGP with Gaussian Likelihood and number of inducing points equal to
    number of data-points gives the same results as GPR.
    """
    input_data = (gpr._time_points, gpr._observations)
    opt = tf.optimizers.Adam(learning_rate=1e-2)

    @tf.function
    def opt_step():
        opt.minimize(lambda: svgp.loss(input_data), svgp.trainable_variables)

    true_likelihood = gpr.log_likelihood()
    for _ in range(100):  # number of tries
        for __ in range(100):  # iterations per try
            opt_step()
        trained_likelihood = svgp.elbo(input_data)
        if np.allclose(trained_likelihood, true_likelihood, atol=tol, rtol=tol):
            break
    np.testing.assert_allclose(trained_likelihood, true_likelihood, atol=tol, rtol=tol)


def test_svgp_vs_gpr_batch(with_tf_random_seed, batch_shape, output_dim):
    """
    Test that the SVGP with Gaussian Likelihood and number of inducing points equal to
    number of data-points gives the same results as GPR.

    Tested with different batch shapes and output dimensions.
    """
    svgp, gpr = _svgp_gpr_setup(batch_shape, output_dim, None)
    _test_svgp_vs_gpr(svgp, gpr, tol=1e-6)


def test_svgp_vs_gpr_means(with_tf_random_seed, output_dim):
    """
    Test that the SVGP with Gaussian Likelihood and number of inducing points equal to
    number of data-points gives the same results as GPR.

    Tested with a mean function.
    """
    svgp, gpr = _svgp_gpr_setup(tuple(), output_dim, LinearMeanFunction(1.5))
    # TODO(sam): output dim of 2 with mean function seems to reduce the tightness of fit for no
    #  reason.
    _test_svgp_vs_gpr(svgp, gpr, tol=1e-4)


def test_svgp_vs_vgp_non_gaussian(with_tf_random_seed, batch_shape):
    """
    Test that the SVGP with non-Gaussian Likelihood and number of inducing points equal to
    number of data-points is the same as VGP.

    This does not test convergence, only that after a given number of steps the marginal
    likelihoods are the same.
    """
    svgp, vgp = _svgp_vgp_setup(batch_shape, 1)
    input_data = (vgp._time_points, vgp._observations)

    vgp_opt = tf.optimizers.SGD(learning_rate=1e-2)
    svgp_opt = tf.optimizers.SGD(learning_rate=1e-2)

    def vgp_opt_step():
        vgp_opt.minimize(vgp.loss, vgp.trainable_variables)

    def svgp_opt_step(input_data):
        svgp_opt.minimize(lambda: svgp.loss(input_data), svgp.trainable_variables)

    for _ in range(50):
        vgp_opt_step()
        svgp_opt_step(input_data)

    np.testing.assert_allclose(vgp.elbo(), svgp.elbo(input_data), atol=1e-6)


def get_dataset_iterator(
    batch_shape: tf.TensorShape,
    input_data: Tuple[tf.Tensor, tf.Tensor],
    minibatch_size: int,
    transform: Callable[[tf.data.Dataset], tf.data.Dataset],
):
    time_points, observations = input_data
    batch_size = batch_shape.num_elements()

    def _get_dataset_iterator(data: tf.Tensor):
        data_shape = list(data.shape)
        data_shape[len(batch_shape)] = minibatch_size
        unpacked_data = tf.unstack(tf.reshape(data, [batch_size, tf.size(data) / batch_size]))

        dataset = tf.data.Dataset.zip(
            tuple(tf.data.Dataset.from_tensor_slices(data_slice) for data_slice in unpacked_data)
        )
        for minibatch in transform(dataset):
            yield tf.reshape(tf.stack(minibatch), data_shape)

    for mb_tp, mb_obs in zip(
        _get_dataset_iterator(time_points), _get_dataset_iterator(observations)
    ):
        yield mb_tp, mb_obs


# def test_svgp_minibatching(with_tf_random_seed, batch_shape):
def test_svgp_minibatching(with_tf_random_seed, batch_shape):
    """
    Test that SVGP with minibatching when using all the available data and when not minibatching
    gives the same ELBO
    """
    svgp, vgp = _svgp_vgp_setup(batch_shape, 1)

    input_data = (vgp.time_points, vgp.observations)
    svgp._num_data = NUM_DATA
    elbo_full = svgp.elbo(input_data)

    minibatch_size = 3
    train_iter = get_dataset_iterator(
        batch_shape=batch_shape,
        input_data=input_data,
        minibatch_size=minibatch_size,
        transform=lambda ds: ds.repeat().shuffle(NUM_DATA).batch(minibatch_size),
    )
    elbo_3 = svgp.elbo(next(train_iter))

    assert elbo_full != elbo_3

    minibatch_size = NUM_DATA
    train_iter = get_dataset_iterator(
        batch_shape=batch_shape,
        input_data=input_data,
        minibatch_size=minibatch_size,
        transform=lambda ds: ds.batch(minibatch_size),
    )
    elbo_N = svgp.elbo(next(train_iter))
    np.testing.assert_allclose(elbo_full, elbo_N)


def test_svgp_log_prior_density(with_tf_random_seed, output_dim):
    """Test that trainable parameters are correctly identified and log_prior_density computed."""
    # gpflow's priors seem to require dtype=float32
    old_default_float = gpflow.default_float()
    gpflow.config.set_default_float(np.float32)
    mf_coef = gpflow.Parameter(1.5)
    svgp, _ = _svgp_gpr_setup(tuple(), output_dim, LinearMeanFunction(mf_coef))
    gpflow.config.set_default_float(old_default_float)
    # get all parameters
    loglike_var = svgp.likelihood.variance
    kernel_var = svgp.kernel.variance
    kernel_ls = svgp.kernel.lengthscale
    # set parameters to not trainable
    for t in svgp.trainable_variables:
        t._trainable = False
    assert svgp.trainable_parameters == ()
    assert svgp.log_prior_density() == 0
    # set these 4 parameters to trainable
    for t in [mf_coef, loglike_var, kernel_var, kernel_ls]:
        gpflow.set_trainable(t, True)
    assert set(svgp.trainable_parameters) == set([mf_coef, loglike_var, kernel_var, kernel_ls])
    # since they have no priors, the density should be 0
    assert svgp.log_prior_density() == 0
    # add priors and check the prior density is the correct sum
    mf_coef.prior = tfp.distributions.Normal(loc=1.0, scale=10.0)
    kernel_var.prior = tfp.distributions.Gamma(concentration=2.0, rate=3.0)
    kernel_ls.prior = tfp.distributions.Gamma(concentration=0.5, rate=2.0)
    loglike_var.prior = tfp.distributions.Normal(loc=0.0, scale=10.0)
    assert svgp.log_prior_density() == tf.add_n(
        [x.log_prior_density() for x in [kernel_var, kernel_ls, loglike_var, mf_coef]]
    )
