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
"""Module containing the hand-crafted `NumpyKalmanFilter`."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseNumpyKalmanFilter(ABC):
    """
    A base class for hand written Kalman Filters against which to compare MarkovFlow.

    This uses the notation found in 'Applied Stochastic Differential Equations' by Sarkka and Solin.
    """

    def __init__(
        self,
        num_timesteps: int,
        transition_matrix: np.ndarray,
        transition_mean: np.ndarray,
        transition_noise: np.ndarray,
        observation_matrix: np.ndarray,
        initial_state_prior_mean: np.ndarray,
        initial_state_prior_cov: np.ndarray,
    ) -> None:
        self.num_timesteps = num_timesteps

        self.output_dim, self.state_dim = observation_matrix.shape
        self.H = observation_matrix

        assert transition_matrix.shape == (self.state_dim, self.state_dim)
        self.A = transition_matrix

        assert transition_mean.shape == (self.state_dim,)
        self.b = transition_mean

        assert transition_noise.shape == (self.state_dim, self.state_dim)
        self.Q = transition_noise

        assert initial_state_prior_mean.shape == (self.state_dim,)
        self.mu_0 = initial_state_prior_mean

        assert initial_state_prior_cov.shape == (self.state_dim, self.state_dim)
        self.P_0 = initial_state_prior_cov

    @property
    @abstractmethod
    def R(self):
        raise NotImplementedError

    def forward_filter(
        self, observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the forward filter. i.e. return the mean and variance for
            p(xₜ | y₁ ... yₜ₋₁), this is called 'pred' and
            p(xₜ | y₁ ... yₜ) this is called 'filter'
        also return log likelihoods i.e. log p(yₜ | y₁ ... yₜ₋₁)

        :param observations: batch_shape + [num_timesteps, output_dim]
        :return: log_liks: batch_shape + [num_timesteps]
                 filter_mus: batch_shape + [num_timesteps, state_dim]
                 filter_covs: batch_shape + [num_timesteps, state_dim, state_dim]
                 pred_mus: batch_shape + [num_timesteps, state_dim]
                 pred_covs: batch_shape + [num_timesteps, state_dim, state_dim]
        """
        assert observations.shape[-2] == self.num_timesteps
        assert observations.shape[-1] == self.output_dim
        batch_shape = observations.shape[:-2]

        # batch_shape + [0]
        log_liks = np.zeros(batch_shape + (0,))
        # batch_shape + [1, state_dim]
        pred_mus = np.tile(self.mu_0[None, :], reps=batch_shape + (1, 1))
        # [1, state_dim, state_dim]
        pred_ps = self.P_0[None, ...]
        # batch_shape + [0, state_dim]
        filter_mus = np.zeros(batch_shape + (0, self.state_dim))
        # [0, state_dim, state_dim]
        filter_ps = np.zeros((0, self.state_dim, self.state_dim))

        for i in range(self.num_timesteps):
            # batch_shape + [state_dim]
            pred_mu = pred_mus[..., -1, :]
            # [state_dim, state_dim]
            pred_p = pred_ps[-1, :, :]

            # batch_shape + [output_dim]
            v_k = observations[..., i, :] - pred_mu @ self.H.T
            # [output_dim, output_dim]
            inv_obs_p = np.linalg.inv(self.H @ pred_p @ self.H.T + self.R[..., i, :, :])
            # [state_dim, output_dim]
            kalman_gain = pred_p @ self.H.T @ inv_obs_p

            # batch_shape + [1, state_dim]
            filter_mu = (pred_mu + (kalman_gain @ v_k[..., None])[..., 0])[..., None, :]
            # [1, state_dim, state_dim]
            filter_p = ((np.eye(self.state_dim) - kalman_gain @ self.H) @ pred_p)[None, :, :]

            # batch_shape + [i, state_dim]
            filter_mus = np.concatenate([filter_mus, filter_mu], axis=-2)
            # [i, state_dim, state_dim]
            filter_ps = np.concatenate([filter_ps, filter_p], axis=-3)

            # batch_shape + [i, state_dim]
            pred_mus = np.concatenate([pred_mus, filter_mu @ self.A.T + self.b], axis=-2)
            # [i, state_dim, state_dim]
            pred_ps = np.concatenate(
                [pred_ps, self.A @ filter_p @ self.A.T + self.Q[None, :, :]], axis=-3
            )
            # batch_shape
            log_lik = -0.5 * (
                np.einsum("...j,jk,...k->...", v_k, inv_obs_p, v_k)
                + np.log(2.0 * np.pi) * self.output_dim
                - np.linalg.slogdet(inv_obs_p)[1]
            )
            # batch_shape + [i]
            log_liks = np.concatenate([log_liks, log_lik[..., None]], axis=-1)

        return log_liks, filter_mus, filter_ps, pred_mus[..., 1:, :], pred_ps[1:, :, :]

    def backward_smoothing_pass(
        self,
        filtered_means: np.ndarray,
        filtered_covs: np.ndarray,
        predicted_means: np.ndarray,
        predicted_covs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the backwards pass of the Kalman filter to calculate the mean and variance of:
            p(xₖ | y₁ ... y_T)
        i.e. the smoothed values of the latent state given the entire trajectory

        :param filtered_means: batch_shape + [num_timesteps, state_dim]
        :param filtered_covs: [num_timesteps, state_dim, state_dim]
        :param predicted_means: batch_shape + [num_timesteps, state_dim]
        :param predicted_covs: [num_timesteps, state_dim, state_dim]
        :return: smooth_means: batch_shape + [num_timesteps, state_dim]
                 smooth_covs: [num_timesteps, state_dim, state_dim]
        """
        smooth_means = filtered_means[..., -1:, :]  # batch_shape + [1, state_dim]
        smooth_covs = filtered_covs[-1:, :, :]  # [1, state_dim, state_dim]

        for i in reversed(range(self.num_timesteps - 1)):
            # [state_dim, state_dim]
            g_k = filtered_covs[i, :, :] @ self.A.T @ np.linalg.inv(predicted_covs[i, :, :])

            # [num_trajectories, state_dim]
            smooth_mu = (
                filtered_means[..., i, :]
                + (smooth_means[..., 0, :] - predicted_means[..., i, :]) @ g_k.T
            )
            # [state_dim, state_dim]
            smooth_cov = (
                filtered_covs[i, :, :]
                + g_k @ (smooth_covs[0, :, :] - predicted_covs[i, :, :]) @ g_k.T
            )
            # [num_trajectories, i, state_dim]
            smooth_means = np.concatenate([smooth_mu[..., None, :], smooth_means], axis=-2)
            # [i, state_dim, state_dim]
            smooth_covs = np.concatenate([smooth_cov[None, :, :], smooth_covs], axis=-3)

        return smooth_means, smooth_covs

    def generate_trajectories(self, batch_shape: Tuple) -> np.ndarray:
        """
        Generate `batch_shape` trajectories from the Linear Dynamical System.

        :param batch_shape: the shape of the trajectories to create.
        :return: trajectories with shape batch_shape + [num_timesteps, output_dim]
        """
        p_1_sqrt = np.linalg.cholesky(self.P_0)  # [state_dim, state_dim]
        q_sqrt = np.linalg.cholesky(self.Q)  # [state_dim, state_dim]
        r_sqrt = np.linalg.cholesky(self.R)  # [output_dim, output_dim]

        data_dim = batch_shape + (1, self.state_dim)

        mu_0 = np.tile(self.mu_0[None, :], reps=batch_shape + (1, 1))
        # generate the random walk in the latent space
        # batch_shape + [1, state_dim]
        x_s = mu_0 + np.random.normal(size=data_dim) @ p_1_sqrt.T
        for _ in range(self.num_timesteps - 1):
            noise = np.random.normal(size=data_dim) @ q_sqrt.T
            x_s = np.concatenate([x_s, x_s[..., -1:, :] @ self.A.T + noise], axis=-2)

        # create the noise for each latent
        y_noise_dim = batch_shape + (self.num_timesteps, self.output_dim, 1)

        return x_s @ self.H.T + (r_sqrt @ np.random.normal(size=y_noise_dim))[..., 0]


class NumpyKalmanFilter(BaseNumpyKalmanFilter):
    """
    A hand written Kalman Filter with a noise model shared for all observations.

    This uses the notation found in 'Applied Stochastic Differential Equations' by Sarkka and Solin.
    """

    def __init__(
        self,
        num_timesteps: int,
        transition_matrix: np.ndarray,
        transition_mean: np.ndarray,
        transition_noise: np.ndarray,
        observation_matrix: np.ndarray,
        observation_noise: np.ndarray,
        initial_state_prior_mean: np.ndarray,
        initial_state_prior_cov: np.ndarray,
    ) -> None:

        super().__init__(
            num_timesteps,
            transition_matrix,
            transition_mean,
            transition_noise,
            observation_matrix,
            initial_state_prior_mean,
            initial_state_prior_cov,
        )

        assert observation_noise.shape == (self.output_dim, self.output_dim)
        self._R = np.tile(observation_noise, [self.num_timesteps, 1, 1])

    @property
    def R(self):
        return self._R

    def forward_filter(
        self, observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the forward filter. i.e. return the mean and variance for
            p(xₜ | y₁ ... yₜ₋₁), this is called 'pred' and
            p(xₜ | y₁ ... yₜ) this is called 'filter'
        also return log likelihoods i.e. log p(yₜ | y₁ ... yₜ₋₁)

        :param observations: batch_shape + [num_timesteps, output_dim]
        :return: log_liks: batch_shape + [num_timesteps]
                 filter_mus: batch_shape + [num_timesteps, state_dim]
                 filter_covs: batch_shape + [num_timesteps, state_dim, state_dim]
                 pred_mus: batch_shape + [num_timesteps, state_dim]
                 pred_covs: batch_shape + [num_timesteps, state_dim, state_dim]
        """
        assert observations.shape[-2] == self.num_timesteps
        assert observations.shape[-1] == self.output_dim
        batch_shape = observations.shape[:-2]

        # batch_shape + [0]
        log_liks = np.zeros(batch_shape + (0,))
        # batch_shape + [1, state_dim]
        pred_mus = np.tile(self.mu_0[None, :], reps=batch_shape + (1, 1))
        # [1, state_dim, state_dim]
        pred_ps = self.P_0[None, ...]
        # batch_shape + [0, state_dim]
        filter_mus = np.zeros(batch_shape + (0, self.state_dim))
        # [0, state_dim, state_dim]
        filter_ps = np.zeros((0, self.state_dim, self.state_dim))

        for i in range(self.num_timesteps):
            # batch_shape + [state_dim]
            pred_mu = pred_mus[..., -1, :]
            # [state_dim, state_dim]
            pred_p = pred_ps[-1, :, :]

            # batch_shape + [output_dim]
            v_k = observations[..., i, :] - pred_mu @ self.H.T
            # [output_dim, output_dim]
            inv_obs_p = np.linalg.inv(self.H @ pred_p @ self.H.T + self.R[..., i, :, :])
            # [state_dim, output_dim]
            kalman_gain = pred_p @ self.H.T @ inv_obs_p

            # batch_shape + [1, state_dim]
            filter_mu = (pred_mu + (kalman_gain @ v_k[..., None])[..., 0])[..., None, :]
            # [1, state_dim, state_dim]
            filter_p = ((np.eye(self.state_dim) - kalman_gain @ self.H) @ pred_p)[None, :, :]

            # batch_shape + [i, state_dim]
            filter_mus = np.concatenate([filter_mus, filter_mu], axis=-2)
            # [i, state_dim, state_dim]
            filter_ps = np.concatenate([filter_ps, filter_p], axis=-3)

            # batch_shape + [i, state_dim]
            pred_mus = np.concatenate([pred_mus, filter_mu @ self.A.T + self.b], axis=-2)
            # [i, state_dim, state_dim]
            pred_ps = np.concatenate(
                [pred_ps, self.A @ filter_p @ self.A.T + self.Q[None, :, :]], axis=-3
            )
            # batch_shape
            log_lik = -0.5 * (
                np.einsum("...j,jk,...k->...", v_k, inv_obs_p, v_k)
                + np.log(2.0 * np.pi) * self.output_dim
                - np.linalg.slogdet(inv_obs_p)[1]
            )
            # batch_shape + [i]
            log_liks = np.concatenate([log_liks, log_lik[..., None]], axis=-1)

        return log_liks, filter_mus, filter_ps, pred_mus[..., 1:, :], pred_ps[1:, :, :]

    def backward_smoothing_pass(
        self,
        filtered_means: np.ndarray,
        filtered_covs: np.ndarray,
        predicted_means: np.ndarray,
        predicted_covs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the backwards pass of the Kalman filter to calculate the mean and variance of:
            p(xₖ | y₁ ... y_T)
        i.e. the smoothed values of the latent state given the entire trajectory

        :param filtered_means: batch_shape + [num_timesteps, state_dim]
        :param filtered_covs: [num_timesteps, state_dim, state_dim]
        :param predicted_means: batch_shape + [num_timesteps, state_dim]
        :param predicted_covs: [num_timesteps, state_dim, state_dim]
        :return: smooth_means: batch_shape + [num_timesteps, state_dim]
                 smooth_covs: [num_timesteps, state_dim, state_dim]
        """
        smooth_means = filtered_means[..., -1:, :]  # batch_shape + [1, state_dim]
        smooth_covs = filtered_covs[-1:, :, :]  # [1, state_dim, state_dim]

        for i in reversed(range(self.num_timesteps - 1)):
            # [state_dim, state_dim]
            g_k = filtered_covs[i, :, :] @ self.A.T @ np.linalg.inv(predicted_covs[i, :, :])

            # [num_trajectories, state_dim]
            smooth_mu = (
                filtered_means[..., i, :]
                + (smooth_means[..., 0, :] - predicted_means[..., i, :]) @ g_k.T
            )
            # [state_dim, state_dim]
            smooth_cov = (
                filtered_covs[i, :, :]
                + g_k @ (smooth_covs[0, :, :] - predicted_covs[i, :, :]) @ g_k.T
            )
            # [num_trajectories, i, state_dim]
            smooth_means = np.concatenate([smooth_mu[..., None, :], smooth_means], axis=-2)
            # [i, state_dim, state_dim]
            smooth_covs = np.concatenate([smooth_cov[None, :, :], smooth_covs], axis=-3)

        return smooth_means, smooth_covs

    def generate_trajectories(self, batch_shape: Tuple) -> np.ndarray:
        """
        Generate `batch_shape` trajectories from the Linear Dynamical System.

        :param batch_shape: the shape of the trajectories to create.
        :return: trajectories with shape batch_shape + [num_timesteps, output_dim]
        """
        p_1_sqrt = np.linalg.cholesky(self.P_0)  # [state_dim, state_dim]
        q_sqrt = np.linalg.cholesky(self.Q)  # [state_dim, state_dim]
        r_sqrt = np.linalg.cholesky(self.R)  # [output_dim, output_dim]

        data_dim = batch_shape + (1, self.state_dim)

        mu_0 = np.tile(self.mu_0[None, :], reps=batch_shape + (1, 1))
        # generate the random walk in the latent space
        # batch_shape + [1, state_dim]
        x_s = mu_0 + np.random.normal(size=data_dim) @ p_1_sqrt.T
        for _ in range(self.num_timesteps - 1):
            noise = np.random.normal(size=data_dim) @ q_sqrt.T
            x_s = np.concatenate([x_s, x_s[..., -1:, :] @ self.A.T + noise], axis=-2)

        # create the noise for each latent
        y_noise_dim = batch_shape + (self.num_timesteps, self.output_dim, 1)

        return x_s @ self.H.T + (r_sqrt @ np.random.normal(size=y_noise_dim))[..., 0]


class NumpyKalmanFilterWithSites(BaseNumpyKalmanFilter):
    """
    A hand written Kalman Filter against which to compare MarkovFlow.

    This uses the notation found in 'Applied Stochastic Differential Equations' by Sarkka and Solin.
    """

    def __init__(
        self,
        num_timesteps: int,
        transition_matrix: np.ndarray,
        transition_mean: np.ndarray,
        transition_noise: np.ndarray,
        observation_matrix: np.ndarray,
        observation_covariances: np.ndarray,
        observation_means: np.ndarray,
        initial_state_prior_mean: np.ndarray,
        initial_state_prior_cov: np.ndarray,
    ) -> None:

        super().__init__(
            num_timesteps,
            transition_matrix,
            transition_mean,
            transition_noise,
            observation_matrix,
            initial_state_prior_mean,
            initial_state_prior_cov,
        )

        assert observation_covariances.shape[-2:] == (self.output_dim, self.output_dim)
        self._R = observation_covariances

        assert observation_means.shape[-1:] == (self.output_dim,)
        self.m = observation_means

    @property
    def R(self):
        return self._R
