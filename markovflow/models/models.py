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
"""
Module containing base classes for models.

.. note:: Markovflow models are intended to work with eager mode in TensorFlow. Therefore
   models (and their collaborating objects) should typically avoid performing any
   computation in their `__init__` methods. Because models and other objects are typically
   initialised outside of an optimisation loop, performing computation in the constructor
   means that this computation is performed 'too early', and optimisation is not possible.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import tensorflow as tf
import gpflow

from markovflow.posterior import PosteriorProcess
from markovflow.utils import tf_scope_class_decorator


class MarkovFlowModel(gpflow.Module, ABC):
    """
    Abstract class representing Markovflow models that depend on input data.

    All Markovflow models are :class:`GPflow Modules <gpflow.Module>`, so it is possible to obtain
    trainable variables via the :attr:`trainable_variables` attribute and trainable parameters via
    the :attr:`trainable_parameters` attribute. You can combine this with the :meth:`loss` method
    to train the model. For example::

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
        for i in range(iterations):
            model.optimization_step(optimizer)

    Call the :meth:`predict_f` method to predict marginal function values at future time points.
    For example::

        mean, variance = model.predict_f(validation_data_tensor)

    .. note:: Markovflow models that extend this class must implement the :meth:`loss`
       method and :attr:`posterior` attribute.
    """

    def log_prior_density(self) -> tf.Tensor:
        """
        Sum of the log prior probability densities of all (constrained) variables in this model.
        """
        if self.trainable_parameters:
            return tf.add_n([p.log_prior_density() for p in self.trainable_parameters])
        else:
            return tf.convert_to_tensor(0.0, gpflow.default_float())

    @abstractmethod
    def loss(self) -> tf.Tensor:
        """
        Obtain the loss, which you can use to train the model.
        It should always return a scalar.

        :raises NotImplementedError: Must be implemented in derived classes.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def posterior(self) -> PosteriorProcess:
        """
        Return a posterior process from the model, which can be used for inference.

        :raises NotImplementedError: Must be implemented in derived classes.
        """
        raise NotImplementedError()

    def predict_state(self, new_time_points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict state at `new_time_points`. Note these time points should be sorted.

        :param new_time_points: Time points to generate observations for, with shape
            ``batch_shape + [num_new_time_points,]``.
        :return: Predicted mean and covariance for the new time points, with respective shapes
            ``batch_shape + [num_new_time_points, state_dim]``
            ``batch_shape + [num_new_time_points, state_dim, state_dim]``.
        """
        return self.posterior.predict_state(new_time_points)

    def predict_f(
        self, new_time_points: tf.Tensor, full_output_cov: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict marginal function values at `new_time_points`. Note these
        time points should be sorted.

        :param new_time_points: Time points to generate observations for, with shape
            ``batch_shape + [num_new_time_points]``.
        :param full_output_cov: Either full output covariance (`True`) or marginal
            variances (`False`).
        :return: Predicted mean and covariance for the new time points, with respective shapes
            ``batch_shape + [num_new_time_points, output_dim]`` and either
            ``batch_shape + [num_new_time_points, output_dim, output_dim]`` or
            ``batch_shape + [num_new_time_points, output_dim]``.
        """
        return self.posterior.predict_f(new_time_points, full_output_cov)


@tf_scope_class_decorator
class MarkovFlowSparseModel(gpflow.Module, ABC):
    """
    Abstract class representing Markovflow models that do not need to store the training
    data (:math:`X, Y`) in the model to approximate the
    posterior predictions :math:`p(f*|X, Y, x*)`.

    This currently applies only to sparse variational models.

    The `optimization_step` method should typically be used to train the model. For example::

        input_data = (tf.constant(time_points), tf.constant(observations))
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
        for i in range(iterations):
            model.optimization_step(input_data, optimizer)

    Call the :meth:`predict_f` method to predict marginal function values at future time points.
    For example::

        mean, variance = model.predict_f(validation_data_tensor)

    .. note:: Markovflow models that extend this class must implement the :meth:`loss`
       method and :attr:`posterior` attribute.
    """

    def log_prior_density(self) -> tf.Tensor:
        """
        Sum of the log prior probability densities of all (constrained) variables in this model.
        """
        if self.trainable_parameters:
            return tf.add_n([p.log_prior_density() for p in self.trainable_parameters])
        else:
            return tf.convert_to_tensor(0.0, gpflow.default_float())

    @abstractmethod
    def loss(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Obtain the loss, which can be used to train the model.

        :param input_data: A tuple of time points and observations containing the data at which
            to calculate the loss for training the model:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``

        :raises NotImplementedError: Must be implemented in derived classes.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def posterior(self) -> PosteriorProcess:
        """
        Obtain a posterior process from the model, which can be used for inference.

        :raises NotImplementedError: Must be implemented in derived classes.
        """
        raise NotImplementedError()

    def predict_state(self, new_time_points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict state at `new_time_points`. Note these time points should be sorted.

        :param new_time_points: Time points to generate observations for, with shape
            ``batch_shape + [num_new_time_points,]``.
        :return: Predicted mean and covariance for the new time points, with respective shapes
            ``batch_shape + [num_new_time_points, state_dim]``
            ``batch_shape + [num_new_time_points, state_dim, state_dim]``.
        """
        return self.posterior.predict_state(new_time_points)

    def predict_f(
        self, new_time_points: tf.Tensor, full_output_cov: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict marginal function values at `new_time_points`. Note these
        time points should be sorted.

        :param new_time_points: Time points to generate observations for, with shape
            ``batch_shape + [num_new_time_points]``.
        :param full_output_cov: Either full output covariance (`True`) or marginal
            variances (`FalseF`).
        :return: Predicted mean and covariance for the new time points, with respective shapes
            ``batch_shape + [num_new_time_points, output_dim]`` and either
            ``batch_shape + [num_new_time_points, output_dim, output_dim]`` or
            ``batch_shape + [num_new_time_points, output_dim]``.
        """
        return self.posterior.predict_f(new_time_points, full_output_cov)

    def predict_log_density(
        self, input_data: Tuple[tf.Tensor, tf.Tensor], full_output_cov: bool = False
    ) -> tf.Tensor:
        """
        Compute the log density of the data. That is:

        .. math:: log ∫ p(yᵢ=Yᵢ|Fᵢ)q(Fᵢ) dFᵢ

        :param input_data: A tuple of time points and observations containing the data at which
            to calculate the loss for training the model:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``

        :param full_output_cov: Either full output covariance (`True`) or marginal
            variances (`False`).
        :return: Predicted log density at input time points, with shape
            ``batch_shape + [num_data]``.
        """
        X, Y = input_data
        f_mean, f_var = self.predict_f(X, full_output_cov=full_output_cov)
        return self.likelihood.predict_log_density(f_mean, f_var, Y)
