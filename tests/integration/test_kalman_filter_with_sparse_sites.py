import numpy as np
import tensorflow as tf
import pytest
from gpflow.config import default_float

from markovflow.kernels.matern import Matern12
from markovflow.mean_function import LinearMeanFunction
from markovflow.models.gaussian_process_regression import GaussianProcessRegression
from markovflow.kalman_filter import KalmanFilterWithSparseSites, UnivariateGaussianSitesNat
from markovflow.likelihoods import MultivariateGaussian


@pytest.fixture(name="kalman_gpr_setup")
def _setup(batch_shape):
    """
    Create a Gaussian Process model and an equivalent kalman filter model
    with more latent states than observations. 
    FIXME: Currently batch_shape isn't used.
    """
    time_grid = np.arange(0.0, 1.0, 0.01)
    time_points = time_grid[::10]
    observations = np.sin(12 * time_points[..., None]) + np.random.randn(len(time_points), 1) * 0.1

    input_data = (
        tf.constant(time_points, dtype=default_float()),
        tf.constant(observations, dtype=default_float()),
    )

    observation_covariance = 1.0  # Same as GPFlow default
    kernel = Matern12(lengthscale=1.0, variance=1.0, output_dim=observations.shape[-1])
    gpr_model = GaussianProcessRegression(
            input_data=input_data,
            kernel=kernel,
            # mean_function=LinearMeanFunction(1.1),
            chol_obs_covariance=tf.constant([[np.sqrt(observation_covariance)]], dtype=default_float()),
        )

    prior_ssm = kernel.state_space_model(time_grid)
    emission_model = kernel.generate_emission_model(time_grid)
    observations_idx = tf.where(tf.equal(time_grid[..., None], time_points))[:, 0][..., None]

    # observations -= gpr_model.mean_function(time_points)

    nat1 = observations / observation_covariance
    nat1 = tf.scatter_nd(observations_idx, nat1, time_grid[..., None].shape)
    nat2 = (-0.5 / observation_covariance) * tf.ones_like(nat1)[..., None]
    lognorm = tf.zeros_like(nat1)
    sites = UnivariateGaussianSitesNat(nat1=nat1, nat2=nat2, log_norm=lognorm)

    kf_sparse_sites = KalmanFilterWithSparseSites(prior_ssm, emission_model, sites, time_grid,
                                                  observations_idx, observations)

    return gpr_model, kf_sparse_sites

def test_kalman_loglikelihood(with_tf_random_seed, kalman_gpr_setup):
    gpr_model, kf_sparse_sites = kalman_gpr_setup

    np.testing.assert_allclose(gpr_model.log_likelihood(), kf_sparse_sites.log_likelihood())