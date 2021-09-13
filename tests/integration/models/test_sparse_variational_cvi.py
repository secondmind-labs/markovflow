# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""Module containing the integration tests for the `CVIGaussianProcess` class."""
import numpy as np
import tensorflow as tf
from gpflow.likelihoods import Gaussian

from markovflow.kernels import Matern12
from markovflow.models import (
    GaussianProcessRegression,
    SparseCVIGaussianProcess,
    SparseVariationalGaussianProcess,
)
from markovflow.ssm_natgrad import SSMNaturalGradient

OUT_DIM = 1
LENGTH_SCALE = 2.0
VARIANCE = 2.25
NUM_DATA = 6
batch_shape = ()
output_dim = 1


def scvi_gpr_setup(num_inducing):
    """
    Creates a GPR model and a matched Sparse VGP model (z=x),
    and optimize the later (single step)
    """

    time_points, observations, kernel, variance = _setup()

    inducing_points = (
        np.linspace(np.min(time_points.numpy()), np.max(time_points.numpy()), num_inducing) + 1e-10
    )

    chol_obs_covariance = tf.eye(output_dim, dtype=tf.float64) * tf.sqrt(variance)
    input_data = (time_points, observations)
    gpr = GaussianProcessRegression(
        kernel=kernel,
        input_data=input_data,
        chol_obs_covariance=chol_obs_covariance,
        mean_function=None,
    )

    likelihood = Gaussian(variance=variance)
    scvi = SparseCVIGaussianProcess(
        kernel=kernel,
        inducing_points=tf.constant(inducing_points),
        likelihood=likelihood,
        learning_rate=1.0,
    )

    # do not train any hyper-parameters for these tests
    for t in likelihood.trainable_variables + kernel.trainable_variables:
        t._trainable = False

    return scvi, gpr, (time_points, observations)


def _setup():
    """ Data, kernel and likelihood setup """
    time_points = np.linspace(0, 1, NUM_DATA)
    observations = (np.cos(20.0 * time_points) + np.random.randn(*time_points.shape)).reshape(-1, 1)
    time_points = tf.constant(time_points)
    observations = tf.constant(observations)

    kernel = Matern12(lengthscale=LENGTH_SCALE, variance=VARIANCE, output_dim=output_dim)

    observation_noise = 1.0
    variance = tf.constant(observation_noise, dtype=tf.float64)

    return time_points, observations, kernel, variance


def test_scvi_unchanged_at_optimum(with_tf_random_seed):
    """Test that the update does not change sites at the optimum"""
    scvi, _, data = scvi_gpr_setup(NUM_DATA)
    scvi.update_sites(data)

    with tf.GradientTape() as g:
        g.watch(scvi.trainable_variables)
        elbo = scvi.classic_elbo(data)
    grad_elbo = g.gradient(elbo, scvi.trainable_variables)

    for g in grad_elbo:
        np.testing.assert_allclose(g, 0.0, atol=1e-9)


def test_optimal_sites(with_tf_random_seed):
    """Test that the optimal value of the exact sites match the true sites """
    scvi, gpr, data = scvi_gpr_setup(NUM_DATA)
    scvi.update_sites(data)

    sd = scvi.kernel.state_dim

    # for z = x, the sites are 2 sd x 2 sd but half empty
    # one part must match the GPR site
    scvi_nat1 = scvi.nat1.numpy()[:-1, sd:]
    scvi_nat2 = scvi.nat2.numpy()[:-1, sd:, sd:]

    # manually compute the optimal sites
    s2 = gpr._chol_obs_covariance.numpy()
    gpr_nat1 = gpr.observations / s2
    gpr_nat2 = -0.5 / s2 * np.ones_like(scvi_nat2)

    np.testing.assert_allclose(scvi_nat1, gpr_nat1)
    np.testing.assert_allclose(scvi_nat2, gpr_nat2)


def test_elbo_increases_each_step(with_tf_random_seed):
    """Test that the update increases the elbo """
    scvi, _, data = scvi_gpr_setup(NUM_DATA)
    # set learning rate less than 1 so that convergence is not in one step
    scvi.learning_rate = 1.0
    old_elbo = scvi.classic_elbo(data)

    num_steps = 5
    for _ in range(num_steps):
        scvi.update_sites(data)
        new_elbo = scvi.classic_elbo(data)
        assert old_elbo <= new_elbo
        print(old_elbo - new_elbo)
        old_elbo = new_elbo


def test_elbo_reach_opt_in_one_step(with_tf_random_seed):
    """Test that one update with lr = 1 reaches optimal elbo"""
    scvi, gpr, data = scvi_gpr_setup(NUM_DATA)

    # set scvi sites to random values
    state_dim = scvi.kernel.state_dim
    scvi.nat2.assign(scvi.nat2.numpy() - np.eye(state_dim * 2))
    scvi.nat1.assign(np.random.randn(*scvi.nat1.shape))

    # run one update step
    scvi.update_sites(data)

    # compute elbo
    elbo_scvi = scvi.classic_elbo(data).numpy()
    log_lik_gpr = gpr.log_likelihood().numpy()

    np.testing.assert_array_almost_equal(elbo_scvi, log_lik_gpr)


def test_s2vgp_natgrad_and_scvi_equivalent(with_tf_random_seed):
    """
    S2VGP natgrads and SCVI should be equivalent
    This is the only test where z != x
    """

    num_inducing = NUM_DATA // 2
    learning_rate = 0.9
    scvi, _, data = scvi_gpr_setup(num_inducing=num_inducing)
    scvi.learning_rate = learning_rate

    s2vgp = SparseVariationalGaussianProcess(
        kernel=scvi.kernel, inducing_points=scvi.inducing_inputs, likelihood=scvi.likelihood,
    )

    opt_ng = SSMNaturalGradient(learning_rate)

    # scvi is state and converges quick
    for _ in range(10):
        scvi.update_sites(data)

    # s2vgp needs a bit more iterations (and is a bit slower)
    @tf.function
    def s2vgp_step():
        opt_ng.minimize(lambda: -s2vgp.elbo(data), ssm=s2vgp.dist_q)

    _ = [s2vgp_step() for _ in range(100)]
    np.testing.assert_almost_equal(scvi.classic_elbo(data), s2vgp.elbo(data), decimal=3)
