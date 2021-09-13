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
"""Module to create different Markovflow and GPflow kernels"""
from typing import List, Type, Union

import gpflow
import tensorflow as tf

from markovflow.emission_model import EmissionModel
from markovflow.gauss_markov import GaussMarkovDistribution
from markovflow.kernels import Kernel, Matern12, Matern32, Matern52, Product, SDEKernel, Sum
from markovflow.kernels.constant import Constant

LENGTH_SCALE = 2.0
VARIANCE1 = 2.25
VARIANCE2 = 3.0
VARIANCE3 = 4.0
VARIANCE4 = 5.0
JITTER = 0.0
OUTPUT_DIM = 1


class DummyNonSDEKernel(Kernel):
    """Dummy class used to test initialization of combination kernels, such as Sum and Product."""

    def output_dim(self):
        pass

    def build_finite_distribution(self, time_points: tf.Tensor) -> GaussMarkovDistribution:
        pass

    def generate_emission_model(self, time_points: tf.Tensor) -> EmissionModel:
        pass


def _create_markovflow_primitive_kernel(c: Type) -> SDEKernel:
    """
    Helper function to return a kernel object from its type name.
    :param c: type name of the kernel, can be Constant, Matern12 or Matern32.
    :return: kernel object of the given type.
    """
    if c == Constant:
        return Constant(variance=VARIANCE1, output_dim=OUTPUT_DIM, jitter=JITTER)
    elif c == Matern12:
        return Matern12(
            lengthscale=LENGTH_SCALE, variance=VARIANCE2, output_dim=OUTPUT_DIM, jitter=JITTER
        )
    elif c == Matern32:
        return Matern32(
            lengthscale=LENGTH_SCALE, variance=VARIANCE3, output_dim=OUTPUT_DIM, jitter=JITTER
        )
    elif c == Matern52:
        return Matern52(
            lengthscale=LENGTH_SCALE, variance=VARIANCE4, output_dim=OUTPUT_DIM, jitter=JITTER
        )
    else:
        raise AssertionError("Given kernel type is not supported.")


def _create_gpflow_primitive_kernel(c: Type) -> gpflow.kernels.Kernel:
    """
    Create GPflow kernel from markovflow kernel type.
    :param c: class of a markovflow kernel.
    :return: Corresponding GPflow kernel.
    """
    if c == Constant:
        return gpflow.kernels.Constant(variance=VARIANCE1)
    elif c == Matern12:
        return gpflow.kernels.Matern12(lengthscales=LENGTH_SCALE, variance=VARIANCE2)
    elif c == Matern32:
        return gpflow.kernels.Matern32(lengthscales=LENGTH_SCALE, variance=VARIANCE3)
    elif c == Matern52:
        return gpflow.kernels.Matern52(lengthscales=LENGTH_SCALE, variance=VARIANCE4)
    else:
        raise AssertionError("Given kernel type is not supported.")


def _create_markovflow_combination_kernel(
    c: Type, children_types: List[Type]
) -> Union[Sum, Product]:
    """
    Create combination kernel of given type.
    :param c: Type of the combination kernel. Value can be Sum or Product.
    :param children_types: the list of children kernels.
    :return: created combination kernel.
    """
    children = [
        _create_markovflow_primitive_kernel(child_type)
        for child_type in children_types
        if child_type is not None
    ]

    if c == Sum:
        return Sum(children, jitter=JITTER)
    elif c == Product:
        return Product(children, jitter=JITTER)
    else:
        raise AssertionError("Given combination kernel type is not supported.")


def _create_gpflow_combination_kernel(
    c: Type, children_types: List[Type]
) -> Union[gpflow.kernels.Sum, gpflow.kernels.Product]:
    """
    Create GPflow combination kernel, either gpflow.kernels.Sum or gpflow.kernels.Product.
    :param c: The Markovflow combination kernel type.
    :param children_types: List of Markovflow primitive kernel types.
    :return: the created GPflow combination kernel.
    """
    children = [_create_gpflow_primitive_kernel(k) for k in children_types]
    if c == Sum:
        return gpflow.kernels.Sum(children)
    elif c == Product:
        return gpflow.kernels.Product(children)
    else:
        raise AssertionError("Given combination kernel type is not supported.")


def _create_markovflow_kernel(kernel: SDEKernel) -> SDEKernel:
    """
    Recreate a markovflow kernel (either primitive or combination).
    This is a walkaround for the issue that tensors are not in the same tensorflow graph.
    :param kernel: the markovflow kernel.
    :return: the re-created markovflow kernel.
    """
    kernel_type = type(kernel)
    if kernel_type in [Sum, Product]:
        combination_kernel_type = type(kernel)
        children_types = [type(c) for c in kernel.kernels]
        result = _create_markovflow_combination_kernel(combination_kernel_type, children_types)
    else:
        result = _create_markovflow_primitive_kernel(kernel_type)
    return result


def _create_gpflow_kernel(kernel: SDEKernel) -> gpflow.kernels.Kernel:
    """
    Create the corresponding gpflow kernel to the given markovflow kernel.
    :param kernel: the markovflow kernel.
    :return: Corresponding gpflow kernel.
    """
    kernel_type = type(kernel)
    if kernel_type in [Sum, Product]:
        children_types = [type(c) for c in kernel.kernels]
        result = _create_gpflow_combination_kernel(kernel_type, children_types)
    else:
        result = _create_gpflow_primitive_kernel(kernel_type)
    return result
