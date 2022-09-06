import numpy as np
import tensorflow as tf
import pytest
from gpflow.config import default_float

from markovflow.kernels.matern import Matern12
from markovflow.mean_function import LinearMeanFunction
from markovflow.models.gaussian_process_regression import GaussianProcessRegression
from markovflow.kalman_filter import KalmanFilterWithSparseSites, UnivariateGaussianSitesNat, KalmanFilterWithSites


@pytest.fixture(
    name="time_step_homogeneous", params=[(0.01, True), (0.01, False), (0.001, True), (0.001, False)],
)
def _time_step_homogeneous_fixture(request):
    return request.param


@pytest.fixture(name="kalman_gpr_setup")
def _setup(batch_shape, time_step_homogeneous):
    """
        Create a Gaussian Process model and an equivalent kalman filter model
        with more latent states than observations.

        Note: Batch shape is ignored as :class:`~markovflow.kalman_filter.KalmanFilterWithSparseSites` currently
        doesn't support batches of sites.
    """
    dt, homogeneous = time_step_homogeneous

    time_grid = np.arange(0.0, 1.0, dt)
    if not homogeneous:
        time_grid = np.sort(np.random.choice(time_grid, 50, replace=False))

    time_points = time_grid[::10]
    observations = np.sin(12 * time_points[..., None]) + np.random.randn(len(time_points), 1) * 0.1

    input_data = (
        tf.constant(time_points, dtype=default_float()),
        tf.constant(observations, dtype=default_float()),
    )

    observation_covariance = 1.0  # Same as GPFlow default
    kernel = Matern12(lengthscale=1.0, variance=1.0, output_dim=observations.shape[-1])
    kernel.set_state_mean(tf.random.normal((1,), dtype=default_float()))
    gpr_model = GaussianProcessRegression(
            input_data=input_data,
            kernel=kernel,
            mean_function=LinearMeanFunction(1.1),
            chol_obs_covariance=tf.constant([[np.sqrt(observation_covariance)]], dtype=default_float()),
        )

    prior_ssm = kernel.state_space_model(time_grid)
    emission_model = kernel.generate_emission_model(time_grid)
    observations_index = tf.where(tf.equal(time_grid[..., None], time_points))[:, 0][..., None]

    observations -= gpr_model.mean_function(time_points)

    nat1 = observations / observation_covariance
    nat2 = (-0.5 / observation_covariance) * tf.ones_like(nat1)[..., None]
    lognorm = tf.zeros_like(nat1)
    sites = UnivariateGaussianSitesNat(nat1=nat1, nat2=nat2, log_norm=lognorm)

    kf_sparse_sites = KalmanFilterWithSparseSites(prior_ssm, emission_model, sites, time_grid.shape[0],
                                                  observations_index, observations)

    return gpr_model, kf_sparse_sites


def test_kalman_loglikelihood(with_tf_random_seed, kalman_gpr_setup):
    """
        Compare Kalman log-likelihood and GPR log-likelihood
    """
    gpr_model, kf_sparse_sites = kalman_gpr_setup

    np.testing.assert_allclose(gpr_model.log_likelihood(), kf_sparse_sites.log_likelihood())


def _get_kf_sites(kf_sparse_sites: KalmanFilterWithSparseSites):
    """
        Get :class:`~markovflow.kalman_filter.KalmanFilterWithSites` from
        :class:`~markovflow.kalman_filter.KalmanFilterWithSparseSites`
    """
    nat1 = kf_sparse_sites.sparse_to_dense(kf_sparse_sites.sites.nat1, kf_sparse_sites.grid_shape)
    nat2 = kf_sparse_sites.sparse_to_dense(kf_sparse_sites.sites.nat2, kf_sparse_sites.grid_shape + (1,)) + 1e-20
    log_norm = kf_sparse_sites.sparse_to_dense(kf_sparse_sites.sites.log_norm, kf_sparse_sites.grid_shape)
    sites = UnivariateGaussianSitesNat(nat1, nat2, log_norm)

    return KalmanFilterWithSites(kf_sparse_sites.prior_ssm,  kf_sparse_sites.emission, sites)


def test_kalman_posterior(with_tf_random_seed, kalman_gpr_setup):
    """
        Compare the marginals of the posterior of :class:`~markovflow.kalman_filter.KalmanFilterWithSites` and
        :class:`~markovflow.kalman_filter.KalmanFilterWithSparseSites`
    """
    _, kf_sparse_sites = kalman_gpr_setup

    kf_sites = _get_kf_sites(kf_sparse_sites)

    kf_m, kf_S = kf_sites.posterior_state_space_model().marginals
    kf_sparse_m, kf_sparse_S = kf_sparse_sites.posterior_state_space_model().marginals

    np.testing.assert_array_almost_equal(kf_m, kf_sparse_m)
    np.testing.assert_array_almost_equal(kf_S, kf_sparse_S)
