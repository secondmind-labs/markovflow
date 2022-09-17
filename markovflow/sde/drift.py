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
"""Module containing classes for different drifts."""
import tensorflow as tf
from gpflow.base import TensorType

from markovflow.state_space_model import StateSpaceModel
import markovflow.sde.sde_utils as utils


class LinearDrift:
    """
    A linear drift of the form, f(x_t, t) = A_t * x_t + b_t.
    """
    def __init__(self, A: TensorType = None, b: TensorType = None):
        """
        Initialize the linear drift with the parameters A and b defining it as f(x_t, t) = A_t * x_t + b_t.

        :param A: `A` parameter of the linear drift with shape `batch_shape + [num_transitions, state_dim, state_dim]``
        :param b: `b` parameter of the linear drift with shape `batch_shape + [num_transitions, state_dim]`
        """
        self.A = A
        self.b = b

    def set_from_ssm(self, ssm: StateSpaceModel, dt: float):
        """
        Approximately converts a StateSpaceModel into a linear drift of the form: f(x(t), t) = A(t) * x(t) + b(t),

        Given the linear drift f(x(t), t) = A(t) * x(t) + b(t), of a continuous time SDE. The conditional distribution
        of two  states  x(t_{k+1}), x(t_{k}) is given by
            x(t_{k+1}) | x(t_{k}) = C_k x(t_{k}) +  b_k + n_k,

         with C_k = exp( \int_{t_{k} < t < t_{k+1}}  A(t) dt) and b_k = [C_k - I] A(t)^{-1} b(t).

        Taking the limit of small-time intervals ( dt = t_{k+1} - t_{k} -> 0). The first order Taylor expansion of
        the exponential  for small-time intervals ( dt = t_{k+1} - t_{k} -> 0), gives the approximation:
                        C_k ~= I + A(t_{k}) dt     and     b_k ~=  b(t_{k}) dt

        Therefore,

        A = (SSM.state_transitions - I)/dt
        b = SSM.state_offsets / dt
        """
        state_transitions = utils.handle_tensor_shape(ssm.state_transitions, desired_dimensions=3)
        state_offsets = utils.handle_tensor_shape(ssm.state_offsets, desired_dimensions=2)

        self.A = (state_transitions - tf.eye(ssm.state_dim, dtype=state_offsets.dtype)) / dt
        self.b = state_offsets / dt

    def to_ssm(self, q: TensorType, transition_times: TensorType, initial_mean: TensorType,
               initial_chol_covariance: TensorType):
        """
        Approximately converts a linear drift SDE of the form: f(x(t), t) = A(t) * x(t) + b(t) to a StateSpaceModel.

        For details, look into the docstring of `set_from_ssm` function.

        SSM.state_transitions = (A + I) * dt
        SSM.state_offsets = b * dt

        """
        if self.A is None or self.b is None:
            raise Exception("Linear drift is empty so SSM can't be created!")

        self.A = utils.handle_tensor_shape(self.A, desired_dimensions=4)  # (B, N, D, D)
        self.b = utils.handle_tensor_shape(self.b, desired_dimensions=3)  # (B, N, D)
        transition_times = utils.handle_tensor_shape(transition_times, desired_dimensions=1)  # (N+1, )
        q = utils.handle_tensor_shape(q, desired_dimensions=4)  # (B, N, D, D)
        initial_mean = utils.handle_tensor_shape(initial_mean, desired_dimensions=2)  # (B, D, )
        initial_chol_covariance = utils.handle_tensor_shape(initial_chol_covariance,
                                                            desired_dimensions=3)  # (B, D, D)

        B, N, D = self.b .shape
        assert self.A.shape == (B, N, D, D)
        assert self.b.shape == (B, N, D)
        assert transition_times.shape == (N+1, )
        assert initial_mean.shape == (B, D,)
        assert initial_chol_covariance.shape == (B, D, D)
        assert q.shape == (B, N, D, D)

        transition_deltas = tf.reshape(transition_times[1:] - transition_times[:-1], (1, -1, 1))
        state_transitions = self.A * tf.expand_dims(transition_deltas, -1) + tf.eye(self.A.shape[-1],
                                                                                    dtype=self.A.dtype)

        state_offsets = self.b * transition_deltas
        chol_process_covariances = q * tf.expand_dims(
            tf.sqrt(transition_deltas), axis=-1
        )

        return StateSpaceModel(
            initial_mean=initial_mean,
            chol_initial_covariance=initial_chol_covariance,
            state_transitions=state_transitions,
            state_offsets=state_offsets,
            chol_process_covariances=chol_process_covariances,
        )
