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

import numpy as np


def assert_samples_close_in_expectation(samples_1, samples_2, sigma=3):
    """
    raises an error if the samples are not within the expected deviation of the sample mean.

    Remember to choose sigma such that it's very likely ALL your tests pass. If you run 10 tests
    testing 10 data points the probability >1 data point (and thus >1 test) failing at
    3 sigma is 1 - 0.997^100 = 26%
    """
    assert samples_1.shape == samples_2.shape
    num_samples, *_ = samples_1.shape

    sample_mean_1 = np.mean(samples_1, axis=0)
    sample_mean_2 = np.mean(samples_2, axis=0)
    diff = np.abs(sample_mean_1 - sample_mean_2)

    all_samples = np.concatenate([samples_1, samples_2])
    var = np.var(all_samples, axis=0)
    sample_mean_std = np.sqrt(var / num_samples)
    normalised_diff = diff / (np.sqrt(2) * sample_mean_std)

    # check that the difference is within sigma error of the standard deviation of the sample mean
    np.testing.assert_array_less(normalised_diff, sigma)


def assert_samples_close_to_mean_in_expectation(samples, true_mean, true_variance=None, sigma=3):
    """
    raises an error if the samples are not within the expected deviation of the true mean. Will
    use the true variance if provided otherwise will estimate it from the samples

    Remember to choose sigma such that it's very likely ALL your tests pass. If you run 10 tests
    testing 10 data points the probability >1 data point (and thus >1 test) failing at
    3 sigma is 1 - 0.997^100 = 26%
    """
    num_samples, *_ = samples.shape
    sample_mean = np.mean(samples, axis=0)
    assert sample_mean.shape == true_mean.shape

    diff = np.abs(sample_mean - true_mean)
    var = true_variance if true_variance else np.var(samples, axis=0)
    sample_mean_std = np.sqrt(var / num_samples)
    normalised_diff = diff / sample_mean_std

    # check that the difference is within sigma error of the standard deviation of the sample mean
    np.testing.assert_array_less(normalised_diff, sigma)
