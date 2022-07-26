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
Package containing ready-to-use GP models.
"""
from .gaussian_process_regression import GaussianProcessRegression
from .iwvi import ImportanceWeightedVI
from .models import MarkovFlowModel, MarkovFlowSparseModel
from .pep import PowerExpectationPropagation
from .sparse_pep import SparsePowerExpectationPropagation
from .sparse_variational import SparseVariationalGaussianProcess
from .sparse_variational_cvi import SparseCVIGaussianProcess
from .spatio_temporal_variational import SpatioTemporalSparseVariational, SpatioTemporalSparseCVI
from .variational import VariationalGaussianProcess
from .variational_cvi import CVIGaussianProcess
