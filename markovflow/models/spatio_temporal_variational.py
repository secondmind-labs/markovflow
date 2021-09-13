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
"""Module containing a model for sparse spatio temporal variational inference"""
from abc import ABC, abstractmethod
from typing import Tuple

import gpflow
import gpflow.kernels as gpfk
import tensorflow as tf

import markovflow.kernels as mfk
from markovflow.emission_model import EmissionModel
from markovflow.kernels import IndependentMultiOutput, SDEKernel
from markovflow.models.models import MarkovFlowSparseModel
from markovflow.posterior import AnalyticPosteriorProcess, ConditionalProcess, PosteriorProcess
from markovflow.utils import batch_base_conditional


class SparseSpatioTemporalKernel(IndependentMultiOutput):
    """
    A spatio-temporal kernel  k(s,t) can be built from the product of a spatial kernel kₛ(s)
    and a Markovian temporal kernel kₜ(t), i.e. k(s,t) = kₛ(s) kₜ(t)

    A GP f(.)∈ ℝ^m  with kernel  k(Z,.)  [with space marginalized to locations Z]
    can be build as f(.) = chol(Kₛ(Z, Z)) @ [H s₁(.),..., H sₘ(.)],

    where s₁(.),...,sₘ(.) are iid SDEs from the equivalent representation of markovian kernel kₜ(t)
    """

    def __init__(self, kernel_space: gpflow.kernels.Kernel, kernel_time: SDEKernel, inducing_space):
        """
        :param kernel_space: spatial kernel
        :param kernel_time: temporal kernel
        :param inducing_space: spatial inducing points
        """

        self.kernel_space = kernel_space
        self.kernel_time = kernel_time
        self.inducing_space = inducing_space

        # the output dim of the ssm is the number of spatial inducing points
        self._output_dim = inducing_space.shape[-2]

        super().__init__([kernel_time for _ in range(self._output_dim)])

    def generate_emission_model(self, time_points: tf.Tensor) -> EmissionModel:
        r"""
        Generate the emission matrix :math:`H`.
        This is the direct sum of the shared m child emission matrices H,
        pre-multiplied by the Cholesky factor of the spatial kernel evaluated at Z.
            chol(Kₛ(Z, Z)) @ [H,..., H]

        :param time_points: The time points over which the emission model is defined, with shape
                        ``batch_shape + [num_data]``.
        :return: The emission model associated with this kernel.
        """
        H = super().generate_emission_model(time_points).emission_matrix
        L = tf.linalg.cholesky(self.kernel_space(self.inducing_space))
        return EmissionModel(emission_matrix=tf.matmul(L, H))


class SpatioTemporalBase(MarkovFlowSparseModel, ABC):
    """
    Model for Spatio-temporal GP regression using a factor kernel
    k_space_time((s,t),(s',t')) = k_time(t,t') * k_space(s,s')

    where k_time is a Markovian kernel.

    The following notation is used:
        * X=(x,t) - the space-time points of the training data.
        * zₛ - the space inducing/pseudo points.
        * zₜ - the time inducing/pseudo points.
        * y - observations corresponding to points X.
        * f(.,.) the spatio-temporal process
        * x(.,.) the SSM formulation of the spatio-temporal process
        * u(.) = x(zₛ,.) - the spatio-temporal SSM marginalized at zₛ
        * p(y | f) - the likelihood
        * p(.) the prior distribution
        * q(.) the variational distribution

    This can be seen as the temporal extension of gpflow.SVGP,
    where instead of fixed inducing variables u, they are now time dependent u(t)
    and follow a Markov chain.

    for a fixed set of space points zₛ
    p(x(zₛ, .)) is a continuous time process of state dimension Mₛd
    for a fixed time slice t, p(x(.,t)) ~ GP(0, kₛ)

    The following conditional independence holds:
    p(x(s,t) | x(zₛ, .)) = p(x(s,t) | s(zₛ, t)), i.e.,
    prediction at a new point at time t given x(zₛ, .) only depends on s(zₛ, t)

    This builds a spatially sparse process as
    q(x(.,.)) = q(x(zₛ, .)) p(x(.,.) |x(zₛ, .)),
    where the multi-output temporal process q(x(zₛ, .)) is also sparse
    q(x(zₛ, .)) = q(x(zₛ, zₜ)) p(x(zₛ,.) |x(zₛ,  zₜ))

    the marginal q(x(zₛ, zₜ)) is parameterized as a multivariate Gaussian distribution
    """

    def __init__(
        self,
        inducing_space,
        kernel_space: gpfk.Kernel,
        kernel_time: mfk.SDEKernel,
        likelihood: gpflow.likelihoods.Likelihood,
    ):
        """
        :param inducing_space: inducing space points [Ms, D]
        :param kernel_space: Gpflow space kernel
        :param kernel_time: Markovflow time kernel
        :param likelihood: a likelihood object
        """
        super().__init__(self.__class__.__name__)

        self.kernel_space = kernel_space
        self.kernel_time = kernel_time
        self.inducing_space = inducing_space
        self.likelihood = likelihood

        self.num_inducing_space = inducing_space.shape[0]

        self._kernel = SparseSpatioTemporalKernel(
            kernel_time=kernel_time, kernel_space=kernel_space, inducing_space=inducing_space
        )

    def space_time_predict_f(self, X):
        """
        Predict marginal function values at `X`. Note the
        time points should be sorted.

        :param X: Time point and associated spatial dimension to generate observations for,
         with shape
            ``batch_shape + [space_dim + 1, num_new_time_points]``.

        :return: Predicted mean and covariance for the new time points, with respective shapes
            ``batch_shape + [num_new_time_points, output_dim]`` and either
            ``batch_shape + [num_new_time_points, output_dim, output_dim]`` or
            ``batch_shape + [num_new_time_points, output_dim]``.
        """
        # split space and time dimension
        x, t = X[..., :-1], X[..., -1]

        # predict ssm at time points. latent ssm has state dimension : state_dim x Ms
        # and is projected to an output dimesion : Ms
        mean_u, cov_u = self.posterior.predict_f(new_time_points=t, full_output_cov=True)
        chol_cov_u = tf.linalg.cholesky(cov_u)  # Ms x Ms

        # use space conditional
        Kmn = self.kernel_space(self.inducing_space, x)  # Ms x N
        Kmm = self.kernel_space(self.inducing_space)  # Ms x Ms
        Knn = self.kernel_space(x, full_cov=False)  # N x 1

        mean_f, var_f = batch_base_conditional(
            Kmn, Kmm, Knn, tf.linalg.matrix_transpose(mean_u), q_sqrt=chol_cov_u
        )
        return mean_f[..., None], var_f[..., None]

    @abstractmethod
    def loss(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Obtain a `Tensor` representing the loss, which can be used to train the model.

        :param input_data: A tuple of time points and observations containing the data at which
            to calculate the loss for training the model.

        :raises NotImplementedError: Must be implemented in derived classes.
        """
        raise NotImplementedError()

    @property
    def posterior(self) -> PosteriorProcess:
        """ Posterior """
        raise NotImplementedError()


class SparseSpatioTemporalVariational(SpatioTemporalBase):
    """
    Model for Spatio-temporal GP regression using a factor kernel
    k_space_time((s,t),(s',t')) = k_time(t,t') * k_space(s,s')

    where k_time is a Markovian kernel.
    """

    def __init__(
        self,
        inducing_space,
        inducing_time,
        kernel_space: gpfk.Kernel,
        kernel_time: mfk.SDEKernel,
        likelihood: gpflow.likelihoods.Likelihood,
        num_data=None,
    ):
        """
        :param inducing_space: inducing space points [Ms, D]
        :param inducing_time: inducing time points [Mt,]
        :param kernel_space: Gpflow space kernel
        :param kernel_time: Markovflow time kernel
        :param likelihood: a likelihood object
        :param num_data: number of observations
        """

        super().__init__(inducing_space, kernel_space, kernel_time, likelihood)

        self.num_data = num_data
        self.inducing_time = inducing_time
        self.num_inducing_time = inducing_time.shape[0]
        self.ssm_p = self._kernel.state_space_model(self.inducing_time)
        self.ssm_q = self.ssm_p.create_trainable_copy()

        self._posterior = ConditionalProcess(
            posterior_dist=self.ssm_q,
            kernel=self._kernel,
            conditioning_time_points=self.inducing_time,
        )

    def elbo(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Calculates the evidence lower bound (ELBO) log p(y)

        :param input_data: A tuple of space-time points and observations containing data at which
            to calculate the loss for training the model.
        :return: A scalar tensor (summed over the batch_shape dimension) representing the ELBO.
        """
        X, Y = input_data

        # predict the variational posterior at the original time points.
        # calculate sₓ ~ q(sₓ) given that we know q(s(z)), x, z,
        # and then project to function space fₓ = H*sₓ ~ q(fₓ).
        fx_mus, fx_covs = self.space_time_predict_f(X)

        # VE(fₓ) = Σᵢ ∫ log(p(yᵢ | fₓ)) q(fₓ) dfx
        ve_fx = tf.reduce_sum(self.likelihood.variational_expectations(fx_mus, fx_covs, Y))
        # KL[q(s(z))|| p(s(z))]
        kl_fz = tf.reduce_sum(self.ssm_q.kl_divergence(self.ssm_p))
        # Return ELBO({fₓ, fz}) = VE(fₓ) - KL[q(s(z)) || p(s(z))]

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl_fz.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl_fz.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl_fz.dtype)
        return ve_fx * scale - kl_fz

    def loss(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Return the loss, which is the negative evidence lower bound (ELBO).

        :param input_data: A tuple of space-time points and observations containing the data
            at which to calculate the loss for training the model.
        """
        return -self.elbo(input_data)

    @property
    def posterior(self) -> PosteriorProcess:
        """ Posterior process """
        return self._posterior

    def predict_log_density(
        self, input_data: Tuple[tf.Tensor, tf.Tensor], full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Compute the log density of the data at the new data points.
        """
        X, Y = input_data
        f_mean, f_var = self.space_time_predict_f(X)
        return self.likelihood.predict_log_density(f_mean, f_var, Y)
