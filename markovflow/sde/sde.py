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

import tensorflow as tf


class SDE(ABC):
    """
    Abstract class representing Stochastic Differential Equation.

    ..math::
     &dx(t)/dt = f(x(t),t) + l(x(t),t) w(t)

    """

    @abstractmethod
    def drift(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Drift function of the SDE i.e. `f(x(t),t)`

        :param x: state at `t` i.e. `x(t)` with shape ``batch_shape + [state_dim].
        :param t: time `t` with shape ``batch_shape + [state_dim]``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``batch_shape + [state_dim]``.

        :raises NotImplementedError: Must be implemented in derived classes.
        """
        pass

    @abstractmethod
    def diffusion(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Diffusion function of the SDE i.e. `l(x(t),t)`

        :param x: state at `t` i.e. `x(t)` with shape ``batch_shape + [state_dim]``.
        :param t: time `t` with shape ``batch_shape + [state_dim]``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``batch_shape + [state_dim, state_dim]``.

        :raises NotImplementedError: Must be implemented in derived classes.
        """
        pass


class OrnsteinUhlenbeckSDE(SDE):
    """
    Ornstein-Uhlenbeck SDE represented by

    ..math:: dx(t) = -λ x(t) dt + dB(t), the spectral density of the Brownian motion is specified by q.
    """
    def __init__(self, decay: tf.Tensor, q: tf.Tensor = tf.ones((1, 1))):
        """
        Initialize the Ornstein-Uhlenbeck SDE.

        :param decay: λ, a tensor with shape ``[1, 1]``.
        :param q: spectral density of the Brownian motion ``[state_dim, state_dim]``.
        """
        super(OrnsteinUhlenbeckSDE, self).__init__()
        self.decay = decay
        self.q = tf.cast(q, dtype=decay.dtype)
        self.state_dim = q.shape[0]

    def drift(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Drift of the Ornstein-Uhlenbeck process
        ..math:: f(x(t), t) = -λ x(t)

        :param x: state at `t` i.e. `x(t)` with shape ``batch_shape + [state_dim]``.
        :param t: time `t` with shape ``batch_shape + [state_dim]``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``batch_shape + [state_dim]``.
        """
        return -self.decay * x

    def diffusion(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Diffusion of the Ornstein-Uhlenbeck process
        ..math:: l(x(t), t) = sqrt(q)

        :param x: state at `t` i.e. `x(t)` with shape ``batch_shape + [state_dim]``.
        :param t: time `t` with shape ``batch_shape + [state_dim]``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``batch_shape + [state_dim, state_dim]``.
        """
        return tf.sqrt(self.q) * tf.ones((self.state_dim, self.state_dim), dtype=x.dtype)


class DoubleWellSDE(SDE):
    """
    Double-Well SDE represented by

    ..math:: dx(t) = f(x(t)) dt + dB(t),

    where f(x(t)) = 4 x(t) (1 - x(t)^2) and the spectral density of the Brownian motion is specified by q.
    """
    def __init__(self, q: tf.Tensor = tf.ones((1, 1))):
        """
        Initialize the Double-Well SDE.

        :param q: spectral density of the Brownian motion ``[state_dim, state_dim]``.
        """
        super(DoubleWellSDE, self).__init__()
        self.q = q
        self.state_dim = q.shape[0]

    def drift(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Drift of the double-well process
        ..math:: f(x(t), t) = 4 x(t) (1 - x(t)^2)

        :param x: state at `t` i.e. `x(t)` with shape ``batch_shape + [state_dim]``.
        :param t: time `t` with shape ``batch_shape + [state_dim]``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``batch_shape + [state_dim]``.
        """
        return 4. * x * (1. - tf.square(x))

    def diffusion(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Diffusion of the double-well process
        ..math:: l(x(t), t) = sqrt(q)

        :param x: state at `t` i.e. `x(t)` with shape ``batch_shape + [state_dim]``.
        :param t: time `t` with shape ``batch_shape + [state_dim]``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``batch_shape + [state_dim, state_dim]``.
        """
        self.q = tf.cast(self.q, dtype=x.dtype)
        return tf.sqrt(self.q) * tf.ones((self.state_dim, self.state_dim), dtype=x.dtype)
