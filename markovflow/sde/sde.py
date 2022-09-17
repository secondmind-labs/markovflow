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
from gpflow.quadrature import mvnquad


class SDE(ABC):
    """
    Abstract class representing Stochastic Differential Equation.

    ..math::
     &dx(t)/dt = f(x(t),t) + l(x(t),t) w(t)

    """

    def __init__(self, state_dim=1):
        """
        :param state_dim: The output dimension of the kernel.
        """
        self._state_dim = state_dim

    @property
    def state_dim(self) -> int:
        """
        Return the state dimension of the sde.
        """
        return self._state_dim

    @abstractmethod
    def drift(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Drift function of the SDE i.e. `f(x(t),t)`

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, state_dim)``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``(n_batch, state_dim)``.

        :raises NotImplementedError: Must be implemented in derived classes.
        """
        raise NotImplementedError

    @abstractmethod
    def diffusion(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Diffusion function of the SDE i.e. `l(x(t),t)`

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``(n_batch, state_dim, state_dim)``.

        :raises NotImplementedError: Must be implemented in derived classes.
        """
        raise NotImplementedError

    def gradient_drift(self, x: tf.Tensor, t: tf.Tensor = tf.zeros((1, 1))) -> tf.Tensor:
        """
        Calculates the gradient of the drift wrt the states x(t).

        ..math:: df(x(t))/dx(t)

        :param x: states with shape (num_states, state_dim).
        :param t: time of states with shape (num_states, 1), defaults to zero.

        :return: the gradient of the SDE drift with shape (num_states, state_dim).
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            drift_val = self.drift(x, t)
            dfx = tape.gradient(drift_val, x)
        return dfx

    def expected_drift(self, q_mean: tf.Tensor, q_covar: tf.Tensor) -> tf.Tensor:
        """
        Calculates the Expectation of the drift under the provided Gaussian over states.

        ..math:: E_q(x(t))[f(x(t))]

        :param q_mean: mean of Gaussian over states with shape (n_batch, num_states, state_dim).
        :param q_covar: covariance of Gaussian over states with shape (n_batch, num_states, state_dim, state_dim).

        :return: the expectation value with shape (n_batch, num_states, state_dim).
        """
        fx = lambda x: self.drift(x=x, t=tf.zeros(x.shape[0], 1))

        n_batch, n_states, state_dim = q_mean.shape
        q_mean = tf.reshape(q_mean, (-1, state_dim))
        q_covar = tf.reshape(q_covar, (-1, state_dim, state_dim))

        val = mvnquad(fx, q_mean, q_covar, H=10)

        val = tf.reshape(val, (n_batch, n_states, state_dim))
        return val

    def expected_gradient_drift(self, q_mean: tf.Tensor, q_covar: tf.Tensor) -> tf.Tensor:
        """
         Calculates the Expectation of the gradient of the drift under the provided Gaussian over states

        ..math:: E_q(.)[f'(x(t))]

        :param q_mean: mean of Gaussian over states with shape (n_batch, num_states, state_dim).
        :param q_covar: covariance of Gaussian over states with shape (n_batch, num_states, state_dim, state_dim).

        :return: the expectation value with shape (n_batch, num_states, state_dim).
        """
        n_batch, n_states, state_dim = q_mean.shape
        q_mean = tf.reshape(q_mean, (-1, state_dim))
        q_covar = tf.reshape(q_covar, (-1, state_dim, state_dim))
        val = mvnquad(self.gradient_drift, q_mean, q_covar, H=10)

        val = tf.reshape(val, (n_batch, n_states, state_dim))
        return val


class OrnsteinUhlenbeckSDE(SDE):
    """
    Ornstein-Uhlenbeck SDE represented by

    ..math:: dx(t) = -λ x(t) dt + dB(t), the spectral density of the Brownian motion is specified by q.
    """

    def __init__(self, decay: tf.Tensor, q: tf.Tensor = tf.ones((1, 1))):
        """
        Initialize the Ornstein-Uhlenbeck SDE.

        :param decay: λ, a tensor with shape ``(1, 1)``.
        :param q: spectral density of the Brownian motion ``(state_dim, state_dim)``.
        """
        super(OrnsteinUhlenbeckSDE, self).__init__(state_dim=q.shape[0])
        self.decay = decay
        self.q = q

    def drift(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Drift of the Ornstein-Uhlenbeck process
        ..math:: f(x(t), t) = -λ x(t)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``(n_batch, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return -self.decay * x

    def diffusion(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Diffusion of the Ornstein-Uhlenbeck process
        ..math:: l(x(t), t) = sqrt(q)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``(n_batch, state_dim, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return tf.ones_like(x[..., None]) * tf.linalg.cholesky(self.q)


class DoubleWellSDE(SDE):
    """
    Double-Well SDE represented by

    ..math:: dx(t) = f(x(t)) dt + dB(t),

    where f(x(t)) = 4 x(t) (1 - x(t)^2) and the spectral density of the Brownian motion is specified by q.
    """

    def __init__(self, q: tf.Tensor = tf.ones((1, 1))):
        """
        Initialize the Double-Well SDE.

        :param q: spectral density of the Brownian motion ``(state_dim, state_dim)``.
        """
        super(DoubleWellSDE, self).__init__(state_dim=q.shape[0])
        self.q = q

    def drift(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Drift of the double-well process
        ..math:: f(x(t), t) = 4 x(t) (1 - x(t)^2)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``(n_batch, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return 4.0 * x * (1.0 - tf.square(x))

    def diffusion(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Diffusion of the double-well process
        ..math:: l(x(t), t) = sqrt(q)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``(n_batch, state_dim, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return tf.ones_like(x[..., None]) * tf.linalg.cholesky(self.q)
