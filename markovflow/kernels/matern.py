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
"""Module containing the Matern family of kernels."""

import tensorflow as tf
from gpflow import Parameter, default_float
from gpflow.utilities import positive

from markovflow.kernels.sde_kernel import StationaryKernel
from markovflow.utils import tf_scope_class_decorator, tf_scope_fn_decorator


@tf_scope_class_decorator
class Matern12(StationaryKernel):
    r"""
    Represents the Matern1/2 kernel. This kernel has the formula:

    .. math:: C(x, x') = σ² exp(-|x - x'| / ℓ)

    ...where lengthscale :math:`ℓ` and signal variance :math:`σ²` are kernel parameters.

    This defines an SDE where:

    .. math::
        &F = - 1/ℓ\\
        &L = 1

    ...so that :math:`Aₖ = exp(-Δtₖ/ℓ)`.
    """

    def __init__(
        self, lengthscale: float, variance: float, output_dim: int = 1, jitter: float = 0.0
    ) -> None:
        """
        :param lengthscale: A value for the lengthscale parameter.
        :param variance: A value for the variance parameter.
        :param output_dim: The output dimension of the kernel.
        :param jitter: A small non-negative number to add into a matrix's diagonal to
            maintain numerical stability during inversion.
        """
        super().__init__(output_dim, jitter=jitter)

        _check_lengthscale_and_variance(lengthscale, variance)

        self._lengthscale = Parameter(lengthscale, transform=positive(), name="lengthscale")
        self._variance = Parameter(variance, transform=positive(), name="variance")

    @property
    def state_dim(self) -> int:
        """Return the state dimension of the kernel, which is always one."""
        return 1

    def state_transitions(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the state transition matrices kernel.

        The state dimension is one, so the matrix exponential reduces to a standard one:

        .. math:: Aₖ = exp(-Δtₖ/ℓ)

        Because this is a stationary kernel, `transition_times` is ignored.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``. Ignored.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        tf.debugging.assert_rank_at_least(time_deltas, 1, message="time_deltas cannot be a scalar.")
        state_transitions = tf.exp(-time_deltas / self._lengthscale)[..., None, None]
        shape = tf.concat([tf.shape(time_deltas), [self.state_dim, self.state_dim]], axis=0)
        tf.debugging.assert_equal(tf.shape(state_transitions), shape)
        return state_transitions

    @property
    def feedback_matrix(self) -> tf.Tensor:
        """
        Return the feedback matrix :math:`F`. This is where:

        .. math:: dx(t)/dt = F x(t) + L w(t)

        For this kernel, note that :math:`F = - 1 / ℓ`.

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        return tf.identity([[-1.0 / self._lengthscale]])

    @property
    def steady_state_covariance(self) -> tf.Tensor:
        """
        Return the steady state covariance :math:`P∞`. For this kernel,
        this is the variance hyperparameter.

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """

        return tf.identity(tf.reshape(self._variance, (self.state_dim, self.state_dim)))

    @property
    def lengthscale(self) -> Parameter:
        """
        Return the lengthscale parameter. This is a GPflow
        `Parameter <https://gpflow.readthedocs.io/en/master/gpflow/index.html#gpflow-parameter>`_.
        """
        return self._lengthscale

    @property
    def variance(self) -> Parameter:
        """
        Return the variance parameter. This is a GPflow
        `Parameter <https://gpflow.readthedocs.io/en/master/gpflow/index.html#gpflow-parameter>`_.
        """
        return self._variance


@tf_scope_class_decorator
class OrnsteinUhlenbeck(StationaryKernel):
    r"""
    Represents the Ornstein–Uhlenbeck kernel.
    This is an alternative parameterization of the Matern1/2 kernel.
    This kernel has the formula:

    .. math:: C(x, x') = q/2λ exp(-λ|x - x'|)

    ...where decay :math:`λ` and diffusion coefficient :math:`q` are kernel parameters.

    This defines an SDE where:

    .. math::
        &F = - λ\\
        &L = q

    ...so that :math:`Aₖ = exp(-λ Δtₖ)`.
    """

    def __init__(
        self, decay: float, diffusion: float, output_dim: int = 1, jitter: float = 0.0
    ) -> None:
        """
        :param decay: A value for the decay parameter.
        :param diffusion: A value for the diffusion parameter.
        :param output_dim: The output dimension of the kernel.
        :param jitter: A small non-negative number to add into a matrix's diagonal to
            maintain numerical stability during inversion.
        """
        super().__init__(output_dim, jitter=jitter)

        _check_lengthscale_and_variance(decay, diffusion)

        self._decay = Parameter(decay, transform=positive(), name="decay")
        self._diffusion = Parameter(diffusion, transform=positive(), name="diffusion")

    @property
    def state_dim(self) -> int:
        """Return the state dimension of the kernel, which is always one."""
        return 1

    def state_transitions(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the state transition matrices kernel.

        The state dimension is one, so the matrix exponential reduces to a standard one:

        .. math:: Aₖ = exp(-λ Δtₖ)

        Because this is a stationary kernel, `transition_times` is ignored.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``. Ignored.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        tf.debugging.assert_rank_at_least(time_deltas, 1, message="time_deltas cannot be a scalar.")
        state_transitions = tf.exp(-time_deltas * self._decay)[..., None, None]
        shape = tf.concat([tf.shape(time_deltas), [self.state_dim, self.state_dim]], axis=0)
        tf.debugging.assert_equal(tf.shape(state_transitions), shape)
        return state_transitions

    @property
    def feedback_matrix(self) -> tf.Tensor:
        """
        Return the feedback matrix :math:`F`. This is where:

        .. math:: dx(t)/dt = F x(t) + L w(t)

        For this kernel, note that :math:`F = -λ`.

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        return tf.identity([[-self._decay]])

    @property
    def steady_state_covariance(self) -> tf.Tensor:
        """
        Return the steady state covariance :math:`P∞`. For this kernel,
        this is q/2λ.

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """

        return tf.identity(
            tf.reshape(0.5 * self._diffusion / self._decay, (self.state_dim, self.state_dim))
        )

    @property
    def decay(self) -> Parameter:
        """
        Return the decay parameter. This is a GPflow
        `Parameter <https://gpflow.readthedocs.io/en/master/gpflow/index.html#gpflow-parameter>`_.
        """
        return self._decay

    @property
    def diffusion(self) -> Parameter:
        """
        Return the diffusion parameter. This is a GPflow
        `Parameter <https://gpflow.readthedocs.io/en/master/gpflow/index.html#gpflow-parameter>`_.
        """
        return self._diffusion


@tf_scope_class_decorator
class Matern32(StationaryKernel):
    r"""
    Represents the Matern3/2 kernel. This kernel has the formula:

    .. math:: C(x, x') = σ² (1 + λ|x - x'|) exp(λ|x - x'|)

    ...where :math:`λ = √3 / ℓ`, and lengthscale :math:`ℓ` and signal variance :math:`σ²`
    are kernel parameters.

    The transition matrix :math:`F` in the SDE form for this kernel is:

    .. math::
        F = &[[0, 1]\\
            &[[-λ², -2λ]]

    Covariance for the initial state is:

    .. math::
        P∞ = [&[1, 0],\\
              &[0, λ²]] * \verb|variance|

    ...where `variance` is a kernel parameter.

    Since the characteristic equation for the feedback matrix :math:`F` for this kernel
    is :math:`(λI + F)² = 0`, the state transition matrix is:

    .. math::
        Aₖ &= expm(FΔtₖ)\\
           &= exp(-λΔtₖ) expm((λI + F)Δtₖ)\\
           &= exp(-λΔtₖ) (I + (λI + F)Δtₖ)

    ...where :math:`expm` is the matrix exponential operator. Note that all higher order terms of
    :math:`expm((λI + F)Δtₖ)` disappear.
    """

    def __init__(
        self, lengthscale: float, variance: float, output_dim: int = 1, jitter: float = 0.0
    ) -> None:
        """
        :param lengthscale: A value for the lengthscale parameter.
        :param variance: A value for the variance parameter.
        :param output_dim: The output dimension of the kernel.
        :param jitter: A small non-negative number to add into a matrix's diagonal to
            maintain numerical stability during inversion.
        """
        super().__init__(output_dim, jitter=jitter)

        _check_lengthscale_and_variance(lengthscale, variance)

        self._lengthscale = Parameter(lengthscale, transform=positive(), name="lengthscale")
        self._variance = Parameter(variance, transform=positive(), name="variance")

    @property
    def _lambda(self) -> tf.Tensor:
        """ λ the scalar used elsewhere in the docstrings """
        return tf.math.sqrt(tf.constant(3.0, dtype=default_float())) / self._lengthscale

    @property
    def state_dim(self) -> int:
        """Return the state dimension of the kernel, which is always two."""
        return 2

    def state_transitions(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the state transition matrices for the kernel.

        Because this is a stationary kernel, `transition_times` is ignored.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``. Ignored.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        tf.debugging.assert_rank_at_least(time_deltas, 1, message="time_deltas cannot be a scalar.")
        # [state_dim, state_dim]
        I = tf.eye(self.state_dim, dtype=default_float())
        # [..., num_transitions, 1, 1]
        extended_time_deltas = time_deltas[..., None, None]
        # (λI + F)t [..., num_transitions, state_dim, state_dim]
        F_lambda_I_t = (self.feedback_matrix + self._lambda * I) * extended_time_deltas

        # expm(-λΔtₖ)(I + (λI + F)Δtₖ) [... num_transitions, state_dim, state_dim]
        result = tf.exp(-self._lambda * extended_time_deltas) * (I + F_lambda_I_t)

        shape = tf.concat([tf.shape(time_deltas), [self.state_dim, self.state_dim]], axis=0)
        tf.debugging.assert_equal(tf.shape(result), shape)
        return result

    @property
    def feedback_matrix(self) -> tf.Tensor:
        r"""
        Return the feedback matrix :math:`F`. This is where:

        .. math:: dx(t)/dt = F x(t) + L w(t)

        For this kernel, note that:

        .. math::
            F = &[0    &1]\\
                &[-λ²  &-2λ]

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        return tf.identity([[0, 1], [-tf.square(self._lambda), -2.0 * self._lambda]])

    @property
    def steady_state_covariance(self) -> tf.Tensor:
        r"""
        Return the steady state covariance :math:`P∞`. This is given by:

        .. math::
            P∞ = σ² [&[1, 0],\\
                     &[0, λ²]]

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        return self._variance * tf.convert_to_tensor(
            value=[[1.0, 0], [0, tf.square(self._lambda)]], dtype=default_float()
        )

    @property
    def lengthscale(self) -> Parameter:
        """
        Return the lengthscale parameter. This is a GPflow
        `Parameter <https://gpflow.readthedocs.io/en/master/gpflow/index.html#gpflow-parameter>`_.
        """
        return self._lengthscale

    @property
    def variance(self) -> Parameter:
        """
        Return the variance parameter. This is a GPflow
        `Parameter <https://gpflow.readthedocs.io/en/master/gpflow/index.html#gpflow-parameter>`_.
        """
        return self._variance


@tf_scope_class_decorator
class Matern52(StationaryKernel):
    r"""
    Represents the Matern5/2 kernel. This kernel has the formula:

    .. math:: C(x, x') = σ² (1 + λ|x - x'| + λ²|x - x'|²/3) exp(λ|x - x'|)

    ...where :math:`λ = √5 / ℓ`, and lengthscale :math:`ℓ` and signal variance :math:`σ²`
    are kernel parameters.

    The transition matrix :math:`F` in the SDE form for this kernel is::

        F = [  0,    1,   0]
            [  0,    0,   1]
            [-λ³, -3λ², -3λ]

    Covariance for the initial state is::

        P∞ = σ² [    1,    0, -λ²/3]
                [    0, λ²/3,     0]
                [-λ²/3,    0,    λ⁴]

    Since the characteristic equation for the feedback matrix :math:`F` for this kernel
    is :math:`(λI + F)³ = 0`, the state transition matrix is:

    .. math::
        Aₖ &= expm(FΔtₖ)\\
           &= exp(-λΔtₖ) expm((λI + F)Δtₖ)\\
           &= exp(-λΔtₖ) (I + (λI + F)Δtₖ + (λI + F)²Δtₖ²/2)

    ...where :math:`expm` is the matrix exponential operator. Note that all
    higher order terms disappear.
    """

    def __init__(
        self, lengthscale: float, variance: float, output_dim: int = 1, jitter: float = 0.0
    ) -> None:
        """
        :param lengthscale: A value for the lengthscale parameter.
        :param variance: A value for the variance parameter.
        :param output_dim: The output dimension of the kernel.
        :param jitter: A small non-negative number to add into a matrix's diagonal to
            maintain numerical stability during inversion.
        """
        super().__init__(output_dim, jitter=jitter)
        _check_lengthscale_and_variance(lengthscale, variance)
        self._lengthscale = Parameter(lengthscale, transform=positive(), name="lengthscale")
        self._variance = Parameter(variance, transform=positive(), name="variance")

    @property
    def _lambda(self) -> tf.Tensor:
        """ λ the scalar used elsewhere in the docstrings """
        return tf.math.sqrt(tf.constant(5.0, dtype=default_float())) / self._lengthscale

    @property
    def state_dim(self) -> int:
        """Return the state dimension of the kernel, which is always three."""
        return 3

    def state_transitions(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the state transition matrices for the kernel.

        Because this is a stationary kernel, `transition_times` is ignored.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``. Ignored.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        tf.debugging.assert_rank_at_least(time_deltas, 1, message="time_deltas cannot be a scalar.")
        # [state_dim, state_dim]
        I = tf.eye(self.state_dim, dtype=default_float())
        extended_time_deltas = time_deltas[..., None, None]
        # (λI + F)t [..., num_transitions, state_dim, state_dim]
        F_lambda_I_t = (self.feedback_matrix + self._lambda * I) * extended_time_deltas

        # expm(-λΔtₖ)(I + (λI + F)Δtₖ + (λI + F)²Δtₖ²/2) [... num_transitions, state_dim, state_dim]
        result = tf.exp(-self._lambda * extended_time_deltas) * (
            I + F_lambda_I_t + F_lambda_I_t @ F_lambda_I_t / 2.0
        )

        shape = tf.concat([tf.shape(time_deltas), [self.state_dim, self.state_dim]], axis=0)
        tf.debugging.assert_equal(tf.shape(result), shape)
        return result

    @property
    def feedback_matrix(self) -> tf.Tensor:
        r"""
        Return the feedback matrix :math:`F`. This is where:

        .. math:: dx(t)/dt = F x(t) + L w(t)

        For this kernel, note that::

            F = [[  0,    1,   0]
                 [  0,    0,   1]
                 [-λ³, -3λ², -3λ]]

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        return tf.identity(
            [
                [0, 1, 0],
                [0, 0, 1],
                [-self._lambda ** 3, -3.0 * tf.square(self._lambda), -3.0 * self._lambda],
            ]
        )

    @property
    def steady_state_covariance(self) -> tf.Tensor:
        r"""
        Return the steady state covariance :math:`P∞`. This is given by::

            P∞ = σ² [    1,    0, -λ²/3]
                    [    0, λ²/3,     0]
                    [-λ²/3,    0,    λ⁴]

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """

        lambda_23 = tf.square(self._lambda) / 3.0
        return self._variance * tf.convert_to_tensor(
            value=[[1, 0, -lambda_23], [0, lambda_23, 0], [-lambda_23, 0, self._lambda ** 4]],
            dtype=default_float(),
        )

    @property
    def lengthscale(self) -> Parameter:
        """
        Return the lengthscale parameter. This is a GPflow
        `Parameter <https://gpflow.readthedocs.io/en/master/gpflow/index.html#gpflow-parameter>`_.
        """
        return self._lengthscale

    @property
    def variance(self) -> Parameter:
        """
        Return the variance parameter. This is a GPflow
        `Parameter <https://gpflow.readthedocs.io/en/master/gpflow/index.html#gpflow-parameter>`_.
        """
        return self._variance


@tf_scope_fn_decorator
def _check_lengthscale_and_variance(lengthscale: float, variance: float) -> None:
    """Verify that the lengthscale and variance are positive"""
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be positive.")
    if variance <= 0.0:
        raise ValueError("variance must be positive.")
