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
    def drift(self, x, t):
        pass

    @abstractmethod
    def diffusion(self, x, t):
        pass


class OrnsteinUhlenbeckSDE(SDE):
    """
    Ornstein-Uhlenbeck SDE represented by

    ..math:: dx(t) = -λ x(t) dt + dB(t), the spectral density of the Brownian motion is specified by q.
    """
    def __init__(self, decay: tf.Tensor, q: tf.Tensor = tf.ones((1, 1))):
        super(OrnsteinUhlenbeckSDE, self).__init__()
        self.decay = decay
        self.q = tf.cast(q, dtype=decay.dtype)

    def drift(self, x, t):
        """
        Drift of the Ornstein-Uhlenbeck process
        ..math:: f(x(t), t) = -λ x(t)
        """
        return -self.decay * x

    def diffusion(self, x, t):
        """
        Diffusion of the Ornstein-Uhlenbeck process
        ..math:: l(x(t), t) = sqrt(q)
        """
        return tf.sqrt(self.q) * tf.ones_like(x)


class DoubleWellSDE(SDE):
    """
    Double-Well SDE represented by

    ..math:: dx(t) = f(x(t)) dt + dB(t),

    where f(x(t)) = 4 x(t) (1 - x(t)^2) and the spectral density of the Brownian motion is specified by q.
    """
    def __init__(self, q: tf.Tensor = tf.ones((1, 1))):
        super(DoubleWellSDE, self).__init__()
        self.q = q

    def drift(self, x, t):
        """
        Drift of the double-well process
        ..math:: f(x(t), t) = 4 x(t) (1 - x(t)^2)
        """
        return 4 * x * (1 - x**2)

    def diffusion(self, x, t):
        """
        Diffusion of the double-well process
        ..math:: l(x(t), t) = sqrt(q)
        """
        self.q = tf.cast(self.q, dtype=x.dtype)
        return tf.sqrt(self.q) * tf.ones_like(x)
