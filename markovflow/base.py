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
Module containing type aliases, attributes and helper functions.

.. autodata:: SampleShape
"""
import os
from typing import List, Tuple, Union

import tensorflow_probability as tfp


def ordered():
    """
    A bijector to be used when parameterising inducing points so that we ensure they remain ordered.
    TensorFlow Probability unfortunately got their naming the wrong way around, see
    https://github.com/tensorflow/probability/issues/765 - we need to construct
    Invert(Ordered()) to obtain a Parameter that is always ordered!

    Note: when using TensorFlow Probability >= 0.12.0 this can be replaced with
    `tfp.bijectors.Ascending`.
    """
    return tfp.bijectors.Invert(tfp.bijectors.Ordered())


SampleShape = Union[Tuple, List, int]
"""
A type that is either a ``Tuple``, a ``List``, or an ``int``.
"""


APPROX_INF = 1e10
"""
Constant defining an appropriate level of approximate inference.
"""

AUTO_NAMESCOPE = "AUTO_NAMESCOPE"
"""
Name of environmental variable which if set enables a well structured TensorFlow graph for
tensorboard debugging. In rare cases may cause issues, and will clutter the stacktrace so off by
default.
"""


def auto_namescope_enabled() -> bool:
    """ Return `True` if autonamescoping is enabled. See the description of AUTO_NAMESCOPE."""
    return True if os.environ.get(AUTO_NAMESCOPE) else False
