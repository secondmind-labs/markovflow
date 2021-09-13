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
"""Module containing a natural gradient optimiser."""

from typing import Callable, List, Optional

import tensorflow as tf

from markovflow.ssm_gaussian_transformations import (
    expectations_to_ssm_params,
    naturals_to_ssm_params,
    ssm_to_expectations,
    ssm_to_naturals,
)
from markovflow.state_space_model import StateSpaceModel
from markovflow.utils import tf_scope_class_decorator


@tf_scope_class_decorator
class SSMNaturalGradient(tf.optimizers.Optimizer):
    r"""
    Represents a natural gradient optimiser. It is also capable of updating
    parameters with momentum, as per the :class:`~tf.keras.optimizers.Adam` optimiser.

    To account for momentum we keep track of a running moving average for :math:`g̃` and
    the Fisher norm of :math:`g̃`, where :math:`g̃ = F⁻¹g` is the natural gradient. This is
    defined as the Euclidean gradient preconditioned by the inverse Fisher information matrix.

    The Fisher norm of the natural gradient is given by:

    .. math:: |g̃|_F = g̃ᵀFg̃ = gᵀF⁻¹Fg̃ = gᵀg̃

    ...which is the inner product between the natural gradient and the Euclidean gradient.

    The moving average for the natural gradient and its norm are given by:

    .. math::
        &mₖ₊₁ = β₁ mₖ + (1 - β₁)g̃ₖ\\
        &vₖ₊₁ = β₂ vₖ + (1 - β₂)|g̃|ₖ

    The final update is given by:

    .. math::
        &θ̃ₖ₊₁ = θ̃ₖ - γ mₖ / (√vₖ + ε) \verb|(in the momentum case)|\\
        &θ̃ₖ₊₁ = θ̃ₖ - γ g̃ₖ \verb|(if we don't have momentum)|
    """

    def __init__(
        self,
        gamma: float = 0.1,
        momentum: bool = True,
        beta1: float = 0.9,
        beta2: float = 0.99,
        epsilon: float = 1e-08,
        name: Optional[str] = None,
    ) -> None:
        """
        :param gamma: The learning rate of the optimiser.
        :param momentum: Whether to update with momentum or not.
        :param beta1: The momentum parameter for the moving average of the natural gradient.
        :param beta2: The momentum parameter for the moving average of the norm of natural gradient.
        :param epsilon: A small constant to make sure we do not divide by :math:`0`
            in the momentum term.
        :param name: Optional name to give the optimiser.
        """
        name = self.__class__.__name__ if name is None else name
        super().__init__(name)

        self.gamma = gamma
        self._momentum = momentum
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        # Create the moving average buffers
        self._ms: List[tf.Variable] = []
        self._v = tf.Variable(0.0, dtype=tf.float64, trainable=False)
        self._step_counter = tf.Variable(1, dtype=tf.float64, trainable=False)
        self._effective_lr = tf.Variable(gamma, trainable=False)

    # pylint: disable=arguments-differ
    def minimize(self, loss_fn: Callable, ssm: StateSpaceModel) -> None:
        """
        Minimise the objective function of the model.

        Note the natural gradient optimiser works with variational parameters only.

        :param loss_fn: The Loss function.
        :param ssm: A state space model that represents our variational posterior.
        """
        if not self._ms:
            self._ms = [
                tf.Variable(tf.zeros(tf.shape(x), dtype=tf.float64), trainable=False)
                for x in ssm_to_expectations(ssm)
            ]
        self._natgrad_steps(loss_fn, ssm)

    def _natgrad_steps(self, loss_fn: Callable, ssm: StateSpaceModel):
        """
        Call a natgrad step after wrapping it in a name scope.

        :param loss_fn: A Loss function.
        :param ssm: A state space model that represents our variational posterior.
        """
        with tf.name_scope(f"{self._name}/natural_gradient_steps"):
            self._natgrad_step(loss_fn, ssm)

    def _natgrad_step(self, loss_fn: Callable, ssm: StateSpaceModel):
        """
        Implements equation [10] from

        @inproceedings{salimbeni18,
            title={Natural Gradients in Practice: Non-Conjugate  Variational Inference in
                   Gaussian Process Models},
            author={Salimbeni, Hugh and Eleftheriadis, Stefanos and Hensman, James},
            booktitle={AISTATS},
            year={2018}

        In addition, for convenience with the rest of MarkovFlow, this code computes ∂L/∂η using
        the chain rule:

        g̃ = ∂L/∂η = [(∂L / ∂[ssm_variables])(∂[ssm_variables] / ∂η)]ᵀ

        In the code η = eta, θ = theta.

        :param loss_fn: Loss function.
        :param ssm: A state space model that represents our variational posterior.
        """
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(ssm.trainable_variables)

            loss = loss_fn()
            # the expectation parameterization as function of the ssm parameters
            etas = ssm_to_expectations(ssm)
            # we need these to calculate the relevant gradients
            ssm_params = expectations_to_ssm_params(*etas)

            # the natural parameterization as function of the ssm parameters
            thetas = ssm_to_naturals(ssm)

            if self._momentum:
                # Get ∂L/∂θ via the chain rule. We need that to compute the norm of the natgrad
                ssm_params_2 = naturals_to_ssm_params(*thetas)

        # 1) the oridinary gradient
        dL_dAs, dL_doffsets, dL_dchol_P0, dL_dchol_Qs, dL_dmu0 = tape.gradient(
            loss, ssm.trainable_variables
        )
        dL_dchol_P0 = ssm.cholesky_initial_covariance.transform.forward(dL_dchol_P0)
        dL_dchol_Qs = ssm.cholesky_process_covariances.transform.forward(dL_dchol_Qs)

        dL_dssm = (dL_dAs, dL_doffsets, dL_dchol_P0, dL_dchol_Qs, dL_dmu0)

        # 2) the chain rule to get ∂L/∂η, where η (eta) are the expectation parameters
        dL_detas = tape.gradient(ssm_params, etas, output_gradients=dL_dssm)

        if self._momentum:
            # the chain rule to get ∂L/∂θ, where θ (theta) are the natural parameters
            dL_dthetas = tape.gradient(ssm_params_2, thetas, output_gradients=dL_dssm)

        del tape  # Remove "persitent" tape

        # momentum
        if self._momentum:

            # adjust learning rate to debias momemtum...
            lr = (
                self.gamma
                * tf.sqrt(1.0 - self._beta2 ** self._step_counter)
                / (1.0 - self._beta1 ** self._step_counter)
            )

            # get the moving average for the natural gradients
            ms_new = [m * self._beta1 + (1.0 - self._beta1) * g for m, g in zip(self._ms, dL_detas)]

            # get moving average for the norm of the natural gradients
            natgrad_norm_components = [tf.reduce_sum(g * gt) for g, gt in zip(dL_detas, dL_dthetas)]
            natgrad_norm_components[-1] = natgrad_norm_components[-1] * 2.0
            natgrad_norm = tf.reduce_sum(natgrad_norm_components)

            v_new = self._v * self._beta2 + (1.0 - self._beta2) * natgrad_norm

            # perform natural gradient descent on the θ parameters
            thetas_new = [
                theta - lr * m / (tf.sqrt(v_new) + self._epsilon)
                for theta, m in zip(thetas, ms_new)
            ]

            _ = [
                mov.assign(mov_new) for mov, mov_new in zip(self._ms + [self._v], ms_new + [v_new])
            ]
            self._step_counter.assign_add(1)

            effective_lr = lr / (tf.sqrt(v_new) + self._epsilon)
            self._effective_lr = effective_lr
        else:
            thetas_new = [theta - self.gamma * dL_deta for theta, dL_deta in zip(thetas, dL_detas)]

        # Transform back to the model ssm parameters
        As_new, offsets_new, chol_P0_new, chol_Qs_new, mu0_new = naturals_to_ssm_params(*thetas_new)
        chol_P0_new = ssm.cholesky_initial_covariance.transform.inverse(chol_P0_new)
        chol_Qs_new = ssm.cholesky_process_covariances.transform.inverse(chol_Qs_new)
        ssm_params_new = (As_new, offsets_new, chol_P0_new, chol_Qs_new, mu0_new)

        _ = [v.assign(var_new) for v, var_new in zip(ssm.trainable_variables, ssm_params_new)]

    def get_config(self):
        """Return a Python dictionary containing the configuration of the optimiser."""
        config = super().get_config()
        config.update(
            {
                "gamma": self._serialize_hyperparameter("gamma"),
                "beta1": self._serialize_hyperparameter("beta1"),
                "beta2": self._serialize_hyperparameter("beta2"),
                "epsilon": self._serialize_hyperparameter("epsilon"),
                "momentum": self._serialize_hyperparameter("momentum"),
            }
        )
        return config

    def _resource_apply_dense(self, grad, handle, apply_state):
        # After bumping to TF2.2, PyLint started thinking that this method was abstract in the
        # Optimizers class (despite the fact that it has a trivial implementation). We reproduce
        # the trivial implementation to get around this.
        raise NotImplementedError()

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        # After bumping to TF2.2, PyLint started thinking that this method was abstract in the
        # Optimizers class (despite the fact that it has a trivial implementation). We reproduce
        # the trivial implementation to get around this.
        raise NotImplementedError()
