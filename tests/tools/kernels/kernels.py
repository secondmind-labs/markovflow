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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.linalg import expm


@dataclass
class DataShape:
    batch_shape: Tuple[int, ...]
    time_dim: int


class SDEKernelTest(ABC):
    """
    Abstract base class for test objects used in testing the SDE kernels.

    Some unit tests of the kernels will automatically pick up the list of kernels which have been
    implemented. For those tests it is important to implement a test object which inherits from this
    base class.
    """

    @abstractmethod
    def state_transitions(self, time_points, time_deltas):
        pass

    @abstractmethod
    def initial_covariance(self):
        pass

    @abstractmethod
    def initial_mean(self):
        pass

    def process_covariances(self, time_points, time_deltas):
        state_transitions = self.state_transitions(time_points, time_deltas)

        A_P0_A_T = np.einsum(
            "...ij,jk,...lk", state_transitions, self.steady_state_covariance(), state_transitions,
        )
        return self.steady_state_covariance() - A_P0_A_T

    @abstractmethod
    def steady_state_covariance(self):
        pass


class Matern12Test(SDEKernelTest):
    def __init__(self, variance: float, length_scale: float, data_shape: DataShape) -> None:
        self._variance = variance
        self._length_scale = length_scale
        self._data_shape = data_shape

    def initial_covariance(self):
        return self._variance * np.ones(self._data_shape.batch_shape + (1, 1))

    def initial_mean(self):
        return np.zeros(self._data_shape.batch_shape + (1,))

    def state_transitions(self, time_points, time_deltas):
        return np.exp(-time_deltas / self._length_scale)[..., None, None]

    def steady_state_covariance(self):
        return np.array([[self._variance]])


class Matern32Test(SDEKernelTest):
    def __init__(self, variance: float, length_scale: float, data_shape: DataShape) -> None:
        self._data_shape = data_shape
        self._variance = variance
        self._lambda = np.sqrt(3) / length_scale

    def initial_mean(self):
        return np.zeros(self._data_shape.batch_shape + (2,))

    def initial_covariance(self):
        return self.steady_state_covariance() * np.ones(self._data_shape.batch_shape + (2, 2))

    def state_transitions(self, time_points, time_deltas):
        F = np.array([[0, 1], [-np.square(self._lambda), -2 * self._lambda]])
        F_ts = F * time_deltas[..., None, None]

        # The scipy `expm` function does not support broadcasting over batches of matrices.
        state_transitions = np.zeros_like(F_ts)
        for index in np.ndindex(time_deltas.shape):
            state_transitions[index] = expm(F_ts[index])

        return state_transitions

    def steady_state_covariance(self):
        return np.array([[1, 0], [0, self._lambda ** 2]]) * self._variance


class Matern52Test(SDEKernelTest):
    def __init__(self, variance: float, length_scale: float, data_shape: DataShape) -> None:
        self._data_shape = data_shape

        self._variance = variance
        self._lambda = np.sqrt(5) / length_scale

    def initial_mean(self):
        return np.zeros(self._data_shape.batch_shape + (3,))

    def initial_covariance(self):
        return self.steady_state_covariance() * np.ones(self._data_shape.batch_shape + (3, 3))

    def state_transitions(self, time_points, time_deltas):
        F = np.array(
            [
                [0, 1, 0],
                [0, 0, 1],
                [-self._lambda ** 3, -3 * np.square(self._lambda), -3 * self._lambda],
            ]
        )
        F_ts = F * time_deltas[..., None, None]

        # The scipy `expm` function does not support broadcasting over batches of matrices.
        state_transitions = np.zeros_like(F_ts)
        for index in np.ndindex(time_deltas.shape):
            state_transitions[index] = expm(F_ts[index])

        return state_transitions

    def steady_state_covariance(self):
        lambda_23 = np.square(self._lambda) / 3.0
        return (
            np.array([[1, 0, -lambda_23], [0, lambda_23, 0], [-lambda_23, 0, self._lambda ** 4]])
            * self._variance
        )
