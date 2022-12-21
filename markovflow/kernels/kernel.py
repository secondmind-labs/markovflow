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
"""Module containing a base class for kernels."""
import abc
from abc import abstractmethod

import tensorflow as tf
import gpflow

from markovflow.emission_model import EmissionModel
from markovflow.gauss_markov import GaussMarkovDistribution


class Kernel(gpflow.Module, abc.ABC):
    r"""
    Abstract class generating a :class:`~markovflow.state_space_model.StateSpaceModel` for a
    given set of time points.

    For a given set of time points :math:`tₖ`, define a state space model of the form:

    .. math:: xₖ₊₁ = Aₖ xₖ + qₖ

    ...where:

    .. math::
        &qₖ \sim 𝓝(0, Qₖ)\\
        &x₀ \sim 𝓝(μ₀, P₀)\\
        &xₖ ∈ ℝ^d\\
        &Aₖ ∈ ℝ^{d × d}\\
        &Qₖ ∈ ℝ^{d × d}\\
        &μ₀ ∈ ℝ^{d × 1}\\
        &P₀ ∈ ℝ^{d × d}\\
        &d \verb| is the state_dim|

    And an :class:`~markovflow.emission_model.EmissionModel` for a given output dimension:

    .. math:: fₖ = H xₖ

    ...where:

    .. math::
        &x ∈ ℝ^d\\
        &f ∈ ℝ^m\\
        &H ∈ ℝ^{m × d}\\
        &m \verb| is the output_dim|

    .. note:: Implementations of this class should typically avoid performing computation in their
              `__init__` method. Performing computation in the constructor conflicts with
              running in TensorFlow's eager mode.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Return the output dimension of the kernel.
        """
        raise NotImplementedError

    @abstractmethod
    def build_finite_distribution(self, time_points: tf.Tensor) -> GaussMarkovDistribution:
        """
        Return the :class:`~markovflow.gauss_markov.GaussMarkovDistribution` that this kernel
        represents on the provided time points.

        .. note:: Currently the only representation we can use is a
            :class:`~markovflow.state_space_model.StateSpaceModel`.

        :param time_points: The times between which to define the distribution,
            with shape ``batch_shape + [num_data]``.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_emission_model(self, time_points: tf.Tensor) -> EmissionModel:
        """
        Return the :class:`~markovflow.emission_model.EmissionModel` associated with this kernel
        that maps from the latent :class:`~markovflow.gauss_markov.GaussMarkovDistribution`
        to the observations.

        :param time_points: The time points over which the emission model is defined,
            with shape ``batch_shape + [num_data]``.
        """
        raise NotImplementedError
