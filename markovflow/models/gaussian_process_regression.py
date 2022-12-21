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
"""Module containing a model for GP regression."""
from typing import Optional, Tuple

import tensorflow as tf
from gpflow.base import TensorType

from markovflow.kalman_filter import KalmanFilter
from markovflow.kernels import SDEKernel
from markovflow.likelihoods import MultivariateGaussian
from markovflow.mean_function import MeanFunction, ZeroMeanFunction
from markovflow.models.models import MarkovFlowModel
from markovflow.posterior import AnalyticPosteriorProcess, PosteriorProcess


class GaussianProcessRegression(MarkovFlowModel):
    """
    Performs GP regression.

    The key reference is Chapter 2 of::

        Gaussian Processes for Machine Learning
        Carl Edward Rasmussen and Christopher K. I. Williams
        The MIT Press, 2006. ISBN 0-262-18253-X.

    This class uses the kernel and the time points to create a state space model.
    GP regression is then a Kalman filter on that state space model using the observations.
    """

    def __init__(
        self,
        input_data: Tuple[tf.Tensor, tf.Tensor],
        kernel: SDEKernel,
        mean_function: Optional[MeanFunction] = None,
        chol_obs_covariance: Optional[TensorType] = None,
    ) -> None:
        """
        :param kernel: A kernel defining a prior over functions.
        :param input_data: A tuple of ``(time_points, observations)`` containing the observed data:
            time points of observations, with shape ``batch_shape + [num_data]``,
            observations with shape ``batch_shape + [num_data, observation_dim]``.
        :param chol_obs_covariance: A :data:`~markovflow.base.TensorType` containing
            the Cholesky factor of the observation noise covariance,
            with shape ``[observation_dim, observation_dim]``.
            a default None value will assume independent likelihood variance of 1.0
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        """
        super().__init__(self.__class__.__name__)
        time_points, observations = input_data
        observation_dim = observations.shape[-1]

        if chol_obs_covariance is None:
            chol_obs_covariance = tf.eye(observation_dim)

        tf.ensure_shape(chol_obs_covariance, [observation_dim, observation_dim])

        # ensure that time_points have the shape: batch_shape + [num_data]
        tf.ensure_shape(time_points, observations.shape[:-1])

        # To collect kernel and mean function gpflow.Module trainable_variables
        self._kernel = kernel
        if mean_function is None:
            mean_function = ZeroMeanFunction(obs_dim=1)
        self._mean_function = mean_function

        self._time_points = time_points
        self._observations = observations

        self._chol_obs_covariance = chol_obs_covariance

    @property
    def time_points(self) -> tf.Tensor:
        """
        Return the time points of observations.

        :return: A tensor with shape ``batch_shape + [num_data]``.
        """
        return self._time_points

    @property
    def observations(self) -> tf.Tensor:
        """
        Return the observations.

        :return: A tensor with shape ``batch_shape + [num_data, observation_dim]``.
        """
        return self._observations

    @property
    def kernel(self) -> SDEKernel:
        """
        Return the kernel of the GP.
        """
        return self._kernel

    @property
    def mean_function(self) -> MeanFunction:
        """
        Return the mean function of the GP.
        """
        return self._mean_function

    @property
    def _kalman(self) -> KalmanFilter:
        # subtract the mean function from the observations, if it exists
        residuals = self._observations
        if self._mean_function is not None:
            residuals -= self._mean_function(self._time_points)
        return KalmanFilter(
            state_space_model=self._kernel.state_space_model(self._time_points),
            emission_model=self._kernel.generate_emission_model(self._time_points),
            observations=residuals,
            chol_obs_covariance=self._chol_obs_covariance,
        )

    def loss(self) -> tf.Tensor:
        """
        Return the loss, which is the negative log likelihood.
        """
        return -self.log_likelihood()

    @property
    def posterior(self) -> PosteriorProcess:
        """
        Obtain a posterior process for inference.

        For this class, this is the :class:`~markovflow.posterior.AnalyticPosteriorProcess`
        built from the Kalman filter.
        """
        return AnalyticPosteriorProcess(
            posterior_dist=self._kalman.posterior_state_space_model(),
            kernel=self._kernel,
            conditioning_time_points=self._time_points,
            likelihood=MultivariateGaussian(self._chol_obs_covariance),
            mean_function=self._mean_function,
        )

    def log_likelihood(self) -> tf.Tensor:
        """
        Calculate the log likelihood of the observations given the kernel parameters.

        In other words, :math:`log p(y_{1...T} | ϑ)` for some parameters :math:`ϑ`.

        :return: A scalar tensor (summed over the batch shape and the whole trajectory).
        """
        return self._kalman.log_likelihood()
