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
"""Module containing the `GaussianProcessRegression` integration tests"""
import gpflow.kernels as kernels_gpf
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from gpflow import default_float
from gpflow.models.gpr import GPR

from markovflow.base import auto_namescope_enabled
from markovflow.kernels import Matern32
from markovflow.models import GaussianProcessRegression
from markovflow.mean_function import MeanFunction, ZeroMeanFunction, LinearMeanFunction
from tests.tools.generate_random_objects import generate_random_time_observations

LENGTH_SCALE = 0.9
AMPLITUDE = 0.3
NUM_DATA = 15
OBSERVATION_NOISE_VARIANCE = 1e-3
VARIABLE_NAME = "test_name"


def create_markovflow_gpr(
    time_points: np.ndarray, observations: np.ndarray, mean_function: MeanFunction = None,
) -> GaussianProcessRegression:
    """Create a MarkovFlow Gaussian Process Regression"""
    # create the MarkovFlow model
    chol_obs_covariance = tf.Variable(
        [[np.sqrt(OBSERVATION_NOISE_VARIANCE)]], dtype=default_float(), name=VARIABLE_NAME
    )
    input_data = (tf.constant(time_points), tf.constant(observations))
    if mean_function is None:
        mean_function = ZeroMeanFunction(obs_dim=observations.shape[-1])
    return GaussianProcessRegression(
        input_data=input_data,
        kernel=Matern32(
            lengthscale=LENGTH_SCALE,
            variance=AMPLITUDE * AMPLITUDE,
            output_dim=observations.shape[-1],
        ),
        mean_function=mean_function,
        chol_obs_covariance=chol_obs_covariance,
    )


@pytest.fixture(name="gpflow_gpr_setup")
def _setup_gp_fixture():
    """Create a GPFlow GPR model."""
    time_points, observations = (
        tf.convert_to_tensor(x)
        for x in generate_random_time_observations(
            obs_dim=1, num_data=NUM_DATA, batch_shape=tuple()
        )
    )

    k_gpf = kernels_gpf.Matern32(variance=AMPLITUDE * AMPLITUDE, lengthscales=LENGTH_SCALE)
    gpf_gpr = GPR(data=(time_points[..., None], observations), kernel=k_gpf)
    gpf_gpr.likelihood.variance.assign(tf.constant(OBSERVATION_NOISE_VARIANCE, dtype=tf.float64))
    return gpf_gpr, create_markovflow_gpr(time_points, observations)


def test_gpr_means(with_tf_random_seed, gpflow_gpr_setup):
    """Test the MarkovFlow GPR produces the correct posterior means compared with GPFlow."""
    gpf_gpr, gpr = gpflow_gpr_setup

    mf_means, _ = gpr.posterior.predict_f(gpr._time_points)
    gpf_means = gpf_gpr.predict_f(gpf_gpr.data[0])[0]

    np.testing.assert_allclose(mf_means, gpf_means, rtol=1e-6)


def test_gpr_covariances(with_tf_random_seed, gpflow_gpr_setup):
    """Test the MarkovFlow GPR produces the correct posterior covariances compared with GPFlow."""
    gpf_gpr, gpr = gpflow_gpr_setup

    _, mf_covs = gpr.posterior.predict_f(gpr._time_points, full_output_cov=True)
    mf_covs = tf.linalg.diag_part(mf_covs)
    gpf_covs = gpf_gpr.predict_f(gpf_gpr.data[0])[1]

    np.testing.assert_allclose(gpf_covs, mf_covs, rtol=1e-6)


def test_gpr_log_likelihood(with_tf_random_seed, gpflow_gpr_setup):
    """Test the MarkovFlow GPR produces the correct log likelihoods compared with GPFlow."""
    gpf_gpr, gpr = gpflow_gpr_setup

    mf_liks, gpf_liks = gpr.log_likelihood(), gpf_gpr.log_marginal_likelihood()

    np.testing.assert_allclose(gpf_liks, mf_liks)


def test_loss(with_tf_random_seed, gpflow_gpr_setup):
    """Test the MarkovFlow GPR loss is equal to the negative log likelihood."""
    _, gpr = gpflow_gpr_setup

    mf_liks, mf_loss = gpr.log_likelihood(), gpr.loss()

    np.testing.assert_allclose(mf_loss, -mf_liks)


def test_grad_gpr_log_likelihood(with_tf_random_seed, gpflow_gpr_setup):
    """Test the MarkovFlow GPR produces the correct log likelihoods compared with GPFlow."""
    gpf_gpr, gpr = gpflow_gpr_setup

    with tf.GradientTape() as tape:
        gpr_ll = gpr.log_likelihood()
    mf_grads = tape.gradient(target=gpr_ll, sources=gpr._kernel._lengthscale.unconstrained_variable)
    with tf.GradientTape() as tape:
        gpf_gpr_ll = gpf_gpr.log_marginal_likelihood()
    gpf_grads = tape.gradient(
        target=gpf_gpr_ll, sources=gpf_gpr.kernel.lengthscales.unconstrained_variable
    )

    np.testing.assert_allclose(mf_grads.numpy(), gpf_grads.numpy(), rtol=1e-6)


def test_gpr_trainable_variables():
    """ Test that the expected trainable variables are present """
    gpr = create_markovflow_gpr(*generate_random_time_observations(1, NUM_DATA, tuple()))
    trainables = gpr.trainable_variables
    assert any(VARIABLE_NAME in v.name for v in trainables)
    # The GPR created has a Matern32 kernel - check for those trainables
    basename = f"{gpr._kernel.__class__.__name__}.__init__/" if auto_namescope_enabled() else ""
    assert any(basename + "lengthscale" in v.name for v in trainables)
    assert any(basename + "variance" in v.name for v in trainables)


def test_gpr_predict_f_methods(with_tf_random_seed, gpflow_gpr_setup):
    """
    Test that model predict_f method work as intended and call the underlying posterior process,
    """
    _, gpr = gpflow_gpr_setup
    time_points = gpr.time_points

    preds = gpr.predict_f(time_points)
    preds_p = gpr.posterior.predict_f(time_points)

    np.testing.assert_allclose(preds, preds_p)


def test_gpr_log_prior_density(with_tf_random_seed):
    """Test that trainable parameters are correctly identified and log_prior_density computed."""
    # gpflow's priors seem to require dtype=float32
    old_default_float = default_float()
    gpflow.config.set_default_float(np.float32)
    mf_coef = gpflow.Parameter(1.5)
    gpr = create_markovflow_gpr(
        *generate_random_time_observations(1, NUM_DATA, tuple()), LinearMeanFunction(mf_coef)
    )
    gpflow.config.set_default_float(old_default_float)
    # get all parameters
    kernel_var = gpr.kernel.variance
    kernel_ls = gpr.kernel.lengthscale
    # set parameters to not trainable
    for t in gpr.trainable_variables:
        t._trainable = False
    assert gpr.trainable_parameters == ()
    assert gpr.log_prior_density() == 0
    # set these 3 parameters to trainable
    for t in [mf_coef, kernel_var, kernel_ls]:
        gpflow.set_trainable(t, True)
    # since they have no priors, the density should be 0
    assert gpr.log_prior_density() == 0
    # add priors and check the prior density is the correct sum
    mf_coef.prior = tfp.distributions.Normal(loc=1.0, scale=10.0)
    kernel_var.prior = tfp.distributions.Gamma(concentration=2.0, rate=3.0)
    kernel_ls.prior = tfp.distributions.Gamma(concentration=0.5, rate=2.0)
    assert gpr.log_prior_density() == tf.add_n(
        [x.log_prior_density() for x in [kernel_var, kernel_ls, mf_coef]]
    )
