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
"""Integration test for the `SSMNaturalGradient`."""

import numpy as np
import pytest
import tensorflow as tf
from gpflow.likelihoods import Gaussian

from markovflow.kernels import Matern52, Sum
from markovflow.models import GaussianProcessRegression, VariationalGaussianProcess
from markovflow.ssm_natgrad import SSMNaturalGradient


@pytest.fixture(name="natgrad_models_setup")
def _natgrad_models_setup_fixture():
    return _setup()


def _setup():
    X = tf.identity(np.linspace(0, 1, 100))
    Y = tf.identity(np.random.randn(100, 1))
    kern_list = [Matern52(lengthscale=0.3, variance=0.1) for _ in range(10)]
    kern = Sum(kern_list)
    lik = Gaussian(variance=0.01)
    gpr = GaussianProcessRegression(
        input_data=(X, Y), kernel=kern, chol_obs_covariance=0.1 * tf.eye(1, dtype=tf.float64)
    )
    vgp = VariationalGaussianProcess(input_data=(X, Y), kernel=kern, likelihood=lik)
    return gpr, vgp


def test_natgrad_gets_the_optimal_ELBO_in_1_iteration(with_tf_random_seed, natgrad_models_setup):
    """Test that VGP with 1 natgrad step can recover the true marginal likelihood of GPR."""

    gpr, vgp = natgrad_models_setup

    log_lik = gpr.log_likelihood()
    print("ground truth gpr: ", log_lik)
    print("initial vgp: ", vgp.elbo())

    nat_grad = SSMNaturalGradient(gamma=1e-0, momentum=False)

    @tf.function
    def opt_step():
        nat_grad.minimize(lambda: -vgp.elbo(), vgp.dist_q)

    opt_step()
    elbo = vgp.elbo()

    print("vgp after 1 natgrad step: ", elbo)

    np.testing.assert_allclose(log_lik, elbo, atol=1e-5, rtol=1e-6)
