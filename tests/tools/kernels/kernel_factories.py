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
from typing import Optional

from gpflow import default_float
from tensorflow import DType

from markovflow.kernels import Matern12, Matern32, Matern52, SDEKernel
from tests.tools.kernels.kernels import (
    DataShape,
    Matern12Test,
    Matern32Test,
    Matern52Test,
    SDEKernelTest,
)


class SDEKernelTestFactory(ABC):
    @abstractmethod
    def create_kernel(self) -> SDEKernel:
        pass

    @abstractmethod
    def create_kernel_test(self) -> SDEKernelTest:
        pass


class Matern12TestFactory(SDEKernelTestFactory):
    def __init__(
        self,
        lengthscale: float,
        variance: float,
        data_shape: DataShape,
        output_dim: int,
        float_type: Optional[DType] = None,
    ) -> None:
        self._lengthscale = lengthscale
        self._variance = variance
        self._output_dim = output_dim
        self._data_shape = data_shape
        self._float_type = float_type if float_type else default_float()

    def create_kernel(self) -> SDEKernel:
        return Matern12(
            lengthscale=self._lengthscale, variance=self._variance, output_dim=self._output_dim
        )

    def create_kernel_test(self) -> SDEKernelTest:
        return Matern12Test(self._variance, self._lengthscale, self._data_shape)


class Matern32TestFactory(SDEKernelTestFactory):
    def __init__(
        self,
        lengthscale: float,
        variance: float,
        data_shape: DataShape,
        output_dim: int,
        float_type: Optional[DType] = None,
    ) -> None:
        self._lengthscale = lengthscale
        self._variance = variance
        self._output_dim = output_dim
        self._data_shape = data_shape
        self._float_type = float_type if float_type else default_float()

    def create_kernel(self) -> SDEKernel:
        return Matern32(
            lengthscale=self._lengthscale, variance=self._variance, output_dim=self._output_dim
        )

    def create_kernel_test(self) -> SDEKernelTest:
        return Matern32Test(self._variance, self._lengthscale, self._data_shape)


class Matern52TestFactory(SDEKernelTestFactory):
    def __init__(
        self,
        lengthscale: float,
        variance: float,
        data_shape: DataShape,
        output_dim: int,
        float_type: Optional[DType] = None,
    ) -> None:
        self._lengthscale = lengthscale
        self._variance = variance
        self._output_dim = output_dim
        self._data_shape = data_shape
        self._float_type = float_type if float_type else default_float()

    def create_kernel(self) -> SDEKernel:
        return Matern52(
            lengthscale=self._lengthscale, variance=self._variance, output_dim=self._output_dim
        )

    def create_kernel_test(self) -> SDEKernelTest:
        return Matern52Test(self._variance, self._lengthscale, self._data_shape)
