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
"""Module containing the integration tests for children of the `SpatioTemporalBase` class."""
from copy import deepcopy
import pytest
import numpy as np
import tensorflow as tf
from gpflow.likelihoods import Gaussian
import gpflow.kernels
from gpflow.models import GPR
import gpflow.mean_functions

from markovflow.kernels import Matern12
from markovflow.models import (
    SpatioTemporalSparseVariational,
    SpatioTemporalSparseCVI,
)
from markovflow.models.spatio_temporal_variational import SpatioTemporalBase


@pytest.fixture(name="st_data_rng")
def _setup_data_rng_fixture():
    rng = np.random.default_rng(42)
    return rng


@pytest.fixture(name="st_data")
def _setup_spatiotemporal_data_fixture(st_data_rng):
    x = np.array([0.0, 1.0])
    t = np.array([2.0, 3.0])
    X = np.array(np.meshgrid(x, t)).T.reshape(-1, 2)
    X = X[np.argsort(X[:, 1]), :]
    Y = st_data_rng.normal(size=(X.shape[0], 1))
    return (X, Y)


@pytest.fixture(name="st_mean_function")
def _setup_spatiotemporal_mean_function_fixture(st_data_rng):
    A = np.array([[1.0, 2.0]]).T
    b = np.array([3.0])
    mean_function = gpflow.mean_functions.Linear(A=A, b=b)
    return mean_function


@pytest.fixture(name="st_model_params")
def _setup_spatiotemporal_params_fixture(st_data, st_mean_function):
    X, _ = st_data
    params = dict(
        inducing_space=np.unique(X[:, 0]).reshape(-1, 1),
        inducing_time=np.unique(X[:, 1]),
        kernel_space=gpflow.kernels.Matern32(lengthscales=1, variance=1, active_dims=[0]),
        kernel_time=Matern12(lengthscale=1, variance=1),
        likelihood=Gaussian(),
        mean_function=st_mean_function,
    )
    return params


@pytest.fixture(name="gpr_model")
def _setup_gpr_model_fixture(st_data, st_mean_function):
    kernel = gpflow.kernels.Product(
        [
            gpflow.kernels.Matern32(lengthscales=1, variance=1, active_dims=[0]),
            gpflow.kernels.Matern12(lengthscales=1, variance=1, active_dims=[1]),
        ]
    )
    model = GPR(data=st_data, kernel=kernel, mean_function=st_mean_function)
    return model


def test_spatiotemporalsparsevariational(with_tf_random_seed, st_model_params, gpr_model, st_data):
    """
    Test that `SpatioTemporalSparseVariational` trained on data at the inducing points
    evaluated at the inducing points gives the same ELBO and predicted mean as `GPR`.
    """
    st_model = SpatioTemporalSparseVariational(**st_model_params)
    assert isinstance(st_model, SpatioTemporalBase)
    for t in (
        st_model.kernel.kernel_space.trainable_variables
        + st_model.kernel.kernel_time.trainable_variables
        + st_model._likelihood.trainable_variables
        + st_model._mean_function.trainable_variables
    ):
        t._trainable = False

    true_likelihood = gpr_model.log_marginal_likelihood()

    opt = tf.optimizers.Adam(learning_rate=1e-1)

    @tf.function
    def opt_step(data):
        opt.minimize(lambda: st_model.loss(data), st_model.trainable_variables)

    ntries = 100
    nsteps = 100
    atol = 1e-4
    rtol = 1e-4
    for _ in range(ntries):  # number of tries
        for _ in range(nsteps):
            opt_step(st_data)
            trained_likelihood = st_model.elbo(st_data)
            if np.allclose(trained_likelihood, true_likelihood, atol=atol, rtol=rtol):
                break

    assert np.allclose(trained_likelihood, true_likelihood, atol=atol, rtol=rtol)
    gpr_mean, _ = gpr_model.predict_f(st_data[0])
    st_mean, _ = st_model.space_time_predict_f(st_data[0])
    assert np.allclose(gpr_mean, st_mean, atol=1e-2, rtol=1e-2)


def test_spatiotemporalsparsecvi(with_tf_random_seed, st_model_params, gpr_model, st_data):
    """
    Test that `SpatioTemporalSparseCVI` trained on data at the inducing points
    evaluated at the inducing points gives the same ELBO and predicted mean as `GPR`.
    """
    st_model = SpatioTemporalSparseCVI(**st_model_params, learning_rate=1.0)
    assert isinstance(st_model, SpatioTemporalBase)
    true_likelihood = gpr_model.log_marginal_likelihood()

    nsteps = 10
    for _ in range(nsteps):
        st_model.update_sites(st_data)

    atol = 1e-6
    rtol = 1e-6
    trained_likelihood = st_model.elbo(st_data)
    assert np.allclose(trained_likelihood, true_likelihood, atol=atol, rtol=rtol)
    gpr_mean, _ = gpr_model.predict_f(st_data[0])
    st_mean, _ = st_model.space_time_predict_f(st_data[0])
    assert np.allclose(gpr_mean, st_mean, atol=atol, rtol=rtol)
