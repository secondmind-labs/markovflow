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
from typing import Callable, Sequence

import tensorflow as tf


def minimize(optimizer: tf.optimizers.Optimizer, loss_fn: Callable[[], tf.Tensor], variables: Sequence):
    """Apply one optimizer step for TensorFlow versions without ``Optimizer.minimize``."""
    with tf.GradientTape() as tape:
        loss = loss_fn()
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        (gradient, variable)
        for gradient, variable in zip(gradients, variables)
        if gradient is not None
    )
    return loss
