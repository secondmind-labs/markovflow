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
"""Module containing the integration tests for the `Kernel` class."""
import gpflow
import gpflow.kernels as kernels_gpf
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float

from markovflow.kernels import Matern12, Matern32, Matern52, OrnsteinUhlenbeck, PiecewiseKernel
from markovflow.models.spatio_temporal_variational import SparseSpatioTemporalKernel
from markovflow.utils import kronecker_product
from tests.tools.generate_random_objects import generate_random_time_points
from tests.tools.state_space_model import f_covariances, stich_state_space_models

kernels_tfp = tfp.math.psd_kernels


@pytest.fixture(name="time_points")
def _setup_time_points(batch_shape):
    """Create random time points with batch_shape."""
    num_data = 9
    return generate_random_time_points(expected_range=4.0, shape=batch_shape + (num_data,))


@pytest.fixture(
    name="kernels",
    params=[
        (Matern12, kernels_tfp.MaternOneHalf, kernels_gpf.Matern12),
        (Matern32, kernels_tfp.MaternThreeHalves, kernels_gpf.Matern32),
        (Matern52, kernels_tfp.MaternFiveHalves, kernels_gpf.Matern52),
    ],
)
def _setup_kernels(request):
    """
    Triplet of the MarkovFlow, TensorFlow Probability and GPFlow Kernels
    """
    return request.param


def test_matern_covariance(with_tf_random_seed, time_points, kernels):
    """
    Test that the covariance produced by Matern kernels is correct.
    The kinds of Matern kernels, such as Matern1/2, Matern3/2 is given by the kernels argument.
    """
    length_scale = 0.9
    amplitude = 1.1
    variance = amplitude * amplitude
    *batch_shape, _ = time_points.shape

    markovflow_kernel, tf_kernel, _ = kernels

    kernel = markovflow_kernel(length_scale, variance, output_dim=1)
    ssm = kernel.state_space_model(tf.constant(time_points))

    f_covs = f_covariances(ssm, kernel.generate_emission_model(time_points))

    # create the tfp version to test against
    tfp_kern = tf_kernel(
        amplitude=amplitude * tf.ones(batch_shape, dtype=default_float()),
        length_scale=length_scale * tf.ones(batch_shape, dtype=default_float()),
    )

    tfp_covariances = tfp_kern.matrix(time_points[..., None], time_points[..., None])

    np.testing.assert_allclose(f_covs, tfp_covariances, rtol=1e-6)


def test_matern_covariance_gpflow(with_tf_random_seed, kernels):
    """
    Test the gradient of the full covariance matrix with respect to the lengthscale is the same
    as GPFlow.
    """
    num_data = 9
    time_points = generate_random_time_points(expected_range=4.0, shape=(num_data,))

    length_scale = 0.9
    variance = 1.21

    markovflow_kernel, _, gpflow_kernel = kernels
    kernel = markovflow_kernel(lengthscale=length_scale, variance=variance, output_dim=1)
    ssm = kernel.state_space_model(tf.constant(time_points))

    gpf_kernel = gpflow_kernel(variance=variance, lengthscales=length_scale)
    gpf_covs = gpf_kernel.K(time_points[:, None], time_points[:, None])

    f_covs = f_covariances(ssm, kernel.generate_emission_model(time_points))

    np.testing.assert_allclose(gpf_covs, f_covs, rtol=1e-6)


def test_matern_covariance_grad_gpflow(with_tf_random_seed, kernels):
    """
    Test the gradient of the full covariance matrix with respect to the lengthscale is the same
    as GPFlow.
    """
    num_data = 9
    time_points = generate_random_time_points(expected_range=4.0, shape=(num_data,))

    length_scale = 0.9
    variance = 1.21

    markovflow_kernel, _, gpflow_kernel = kernels

    with tf.GradientTape() as tape:
        kernel = markovflow_kernel(lengthscale=length_scale, variance=variance, output_dim=1)
        ssm = kernel.state_space_model(tf.constant(time_points))
        f_covs = f_covariances(ssm, kernel.generate_emission_model(time_points))
    mkf_grads = tape.gradient(f_covs, kernel._lengthscale.unconstrained_variable)

    with tf.GradientTape() as tape:
        gpf_kernel = gpflow_kernel(variance=variance, lengthscales=length_scale)
        gpf_covs = gpf_kernel.K(time_points[:, None], time_points[:, None])
    gpf_grads = tape.gradient(gpf_covs, gpf_kernel.lengthscales.unconstrained_variable)

    np.testing.assert_allclose(gpf_grads, mkf_grads, rtol=1e-6)


def _kernel_marginal_covs(time_points, kernels):
    length_scale = 0.9
    variance = 1.21

    markovflow_kernel, _, _ = kernels

    kernel = markovflow_kernel(lengthscale=length_scale, variance=variance, output_dim=1)
    ssm = kernel.state_space_model(tf.constant(time_points))

    broadcasted_p_inf = tf.broadcast_to(
        kernel.steady_state_covariance,
        tuple(ssm.batch_shape) + (ssm.num_transitions + 1, ssm.state_dim, ssm.state_dim),
    )

    return kernel, ssm.marginal_covariances, broadcasted_p_inf


def test_kernel_marginal(with_tf_random_seed, time_points, kernels):
    """
    Test that the marginal covariance produced by Matern kernels is correct.

    For the prior this should just be the steady state covariance.
    """
    _, covariances, broadcasted_p_inf = _kernel_marginal_covs(time_points, kernels)
    np.testing.assert_allclose(broadcasted_p_inf, covariances, atol=1e-4)


def test_matern_marginal_grad_var(time_points, kernels):
    """
    Test gradients of the marginal covariance with respect to the variance hyperparameter.

    This should be the same as the gradient of the steady state covariance with respect to
    the variance.
    """
    with tf.GradientTape() as tape:
        kernel, covariances, _ = _kernel_marginal_covs(time_points, kernels)
    grad_marginal_var = tape.gradient(covariances, kernel._variance.unconstrained_variable)

    with tf.GradientTape() as tape:
        kernel, _, broadcasted_p_inf = _kernel_marginal_covs(time_points, kernels)
    grad_p_inf_var = tape.gradient(broadcasted_p_inf, kernel._variance.unconstrained_variable)

    np.testing.assert_allclose(grad_marginal_var, grad_p_inf_var, rtol=1e-4)


def test_matern_marginal_grad_l(with_tf_random_seed, time_points, kernels):
    """
    Test gradients of the marginal covariance with respect to the length scale hyperparameter.

    This should be the same as the gradient of the steady state covariance with respect to
    the length scale.
    """
    with tf.GradientTape() as tape:
        kernel, covariances, _ = _kernel_marginal_covs(time_points, kernels)

    grad_marginal_l = tape.gradient(covariances, kernel._lengthscale.unconstrained_variable)

    with tf.GradientTape() as tape:
        kernel, _, broadcasted_p_inf = _kernel_marginal_covs(time_points, kernels)

    if isinstance(kernel, Matern12):
        # the steady state covariance of the Matern12 doesn't depend on the lengthscale
        # so Tensorflow would throw an error if we asked for the gradient
        grad_p_inf_l = tf.constant(0.0, dtype=broadcasted_p_inf.dtype)
    else:
        grad_p_inf_l = tape.gradient(broadcasted_p_inf, kernel._lengthscale.unconstrained_variable)

    np.testing.assert_allclose(grad_marginal_l, grad_p_inf_l, atol=1e-6, rtol=1e-6)


def test_piecewise_kernel_with_shared_base():
    """
    Test that a piecewise kernel built of many instances of the same base kernel
    is similar to the base kernel
    """
    # construct piecewise stationary kernel
    num_change_points = 5
    change_points = np.arange(num_change_points).astype(float)
    tf_change_points = tf.convert_to_tensor(change_points, dtype=default_float())
    base_kernel = Matern32
    ks = [base_kernel(1.0, 1.0) for _ in range(num_change_points + 1)]
    pk = PiecewiseKernel(ks, tf_change_points)

    x = np.linspace(change_points.min() + 1, change_points.max() + 1, 100).reshape(-1,)
    X = tf.convert_to_tensor(x, dtype=default_float())

    # compute marginals of states indexed at X
    mu_pk, cov_pk = pk.state_space_model(X).marginals
    mu, cov = base_kernel(1.0, 1.0).state_space_model(X).marginals

    # compare marginals
    np.testing.assert_array_almost_equal(mu.numpy(), mu_pk.numpy())
    np.testing.assert_array_almost_equal(cov.numpy(), cov_pk.numpy())


def test_shared_piecewise_kernel_vs_manual():
    """
    Test that a ssm built out of piecewise kernel with a shared base kernel (different parameters)
    matches that of a ssm built by stiching 2 ssms built separately
    """

    # construct piecewise stationary kernel
    change_points = np.array([-1e-5]).astype(float)
    num_change_points = len(change_points)
    tf_change_points = tf.convert_to_tensor(change_points, dtype=default_float())
    base_kernel = Matern32
    # variance and lengthscale of the kernel
    v, l = 1.0, 1.0
    variances = np.array([v, 2 * v])
    lengthscales = np.array([l, 2 * l])
    ks = [
        base_kernel(variance=variances[l], lengthscale=lengthscales[l])
        for l in range(num_change_points + 1)
    ]
    pk = PiecewiseKernel(ks, tf_change_points)

    # predict at points broader than the range of change points
    N = 5
    xs = [np.linspace(-1, 0, N).reshape(-1,), np.linspace(0, 1, N).reshape(-1,)]
    x = np.concatenate([xs[0], xs[1][1:]])
    X = tf.convert_to_tensor(x, dtype=default_float())
    Xs = [tf.convert_to_tensor(x_, dtype=default_float()) for x_ in xs]

    # build state space models from kernels
    ssm_pk = pk.state_space_model(X)
    ssm_1 = base_kernel(variance=variances[0], lengthscale=lengthscales[0]).state_space_model(Xs[0])
    ssm_2 = base_kernel(variance=variances[1], lengthscale=lengthscales[1]).state_space_model(Xs[1])
    ssm = stich_state_space_models([ssm_1, ssm_2])

    # compute marginals of states indexed at X
    mu_pk, cov_pk = ssm_pk.marginals
    mu, cov = ssm.marginals

    # compare marginals
    np.testing.assert_array_almost_equal(mu.numpy(), mu_pk.numpy(), decimal=4)
    np.testing.assert_array_almost_equal(cov.numpy(), cov_pk.numpy(), decimal=4)


def test_spatio_temporal_kernel():
    """
    Test that the marginal variance obtained from gpflow and markovflow
    spatio temporal kernels match.
    """

    # kernel parameters
    k_space_var = 1.0
    k_space_len = 0.5
    k_time_var = 1.0
    k_time_len = 0.6

    # inputs for evaluation
    N_time, N_space = 3, 3
    input_space = np.linspace(0, 1, N_space).reshape(-1, 1)
    input_time = np.linspace(0, 1, N_time).reshape(-1,)

    # markovflow
    mk_space = gpflow.kernels.Matern12(variance=k_space_var, lengthscales=k_space_len)
    mk_time = Matern12(variance=k_time_var, lengthscale=k_time_len)
    kernel = SparseSpatioTemporalKernel(mk_space, mk_time, input_space)
    ssm = kernel.state_space_model(tf.constant(input_time))
    f_covs = f_covariances(ssm, kernel.generate_emission_model(input_time))

    # gpflow
    gpf_space = gpflow.kernels.Matern12(variance=k_space_var, lengthscales=k_space_len)
    gpf_time = gpflow.kernels.Matern12(variance=k_time_var, lengthscales=k_time_len)
    gpf_covs_space = gpf_space.K(input_space)
    gpf_covs_time = gpf_time(input_time.reshape(-1, 1))
    gpf_covs = kronecker_product([gpf_covs_time, gpf_covs_space])

    # comparing
    np.testing.assert_allclose(gpf_covs, f_covs, rtol=1e-6)


def test_ou_match_matern12():
    """ Testing that the Ornstein-Uhlenbeck implementation matches the Matern1/2 implementation """
    length_scale = 0.9
    amplitude = 1.1
    variance = amplitude * amplitude
    time_points = np.linspace(0, 1, 2).reshape(-1,)

    decay = 1.0 / length_scale
    diffusion = variance * 2 * decay

    mat = Matern12(lengthscale=length_scale, variance=variance)
    ou = OrnsteinUhlenbeck(decay=decay, diffusion=diffusion)
    ssm_ou = ou.state_space_model(time_points)
    ssm_mat = mat.state_space_model(time_points)

    np.testing.assert_allclose(ou.steady_state_covariance, mat.steady_state_covariance, rtol=1e-6)
    np.testing.assert_allclose(ssm_ou.marginal_covariances, ssm_mat.marginal_covariances, rtol=1e-6)
    np.testing.assert_allclose(
        ssm_ou.cholesky_process_covariances, ssm_mat.cholesky_process_covariances, rtol=1e-6
    )
