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
from typing import Optional, Tuple, Union

import gpflow
import gpflow.kernels as gpfk
import tensorflow as tf
from gpflow import default_float
from gpflow.base import Parameter
import gpflow.mean_functions

import markovflow.kernels as mfk
from markovflow.conditionals import conditional_statistics
from markovflow.emission_model import EmissionModel
from markovflow.kernels import IndependentMultiOutput
from markovflow.kernels import SDEKernel
import markovflow.mean_function
from markovflow.models.models import MarkovFlowSparseModel
from markovflow.models.variational_cvi import (
    back_project_nats,
    gradient_transformation_mean_var_to_expectation,
)
from markovflow.posterior import ConditionalProcess
from markovflow.posterior import PosteriorProcess
from markovflow.ssm_gaussian_transformations import naturals_to_ssm_params
from markovflow.state_space_model import StateSpaceModel
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
        pre-multiplied by the Cholesky factor of the spatial kernel evaluated at Zₛ.
            chol(Kₛ(Zₛ, Zₛ)) @ [H,..., H]

        :param time_points: The time points over which the emission model is defined, with shape
                        ``batch_shape + [num_data]``.
        :return: The emission model associated with this kernel.
        """
        H = super().generate_emission_model(time_points).emission_matrix
        L = tf.linalg.cholesky(self.kernel_space(self.inducing_space))
        return EmissionModel(emission_matrix=tf.matmul(L, H))

    def state_to_space_conditional_projection(self, inputs):
        r"""
        Generates the matrix P, in the conditional mean E[f(x,t)|s(t)] = P s(t)
        It is given by combining
        E[f(x,t)|f(Zₛ)] =  Kₛ(x, Zₛ)Kₛ(Zₛ, Zₛ)⁻¹f(Zₛ)
        E[f(Zₛ)|s(t)] =  chol(Kₛ(Zₛ, Zₛ)) @ [H,..., H] s(t)
        leading to
        E[f(x,t)|s(t)] = Kₛ(x, Zₛ)Kₛ(Zₛ, Zₛ)⁻¹ chol(Kₛ(Zₛ, Zₛ)) @ [H,..., H] s(t)
            =  Kₛ(x, Zₛ) chol(Kₛ(Zₛ, Zₛ))⁻ᵀ @ [H,..., H] s(t)
        :param inputs: Time point and associated spatial dimension to generate observations for,
         with shape ``batch_shape + [space_dim + 1, num_new_time_points]``.
        :return: The projection tensor with shape
            ``batch_shape + [num_new_time_points, obs_dim, state_dim]``.
        """
        space_points, time_points = inputs[..., :-1], inputs[..., -1]
        H = super().generate_emission_model(time_points).emission_matrix
        chol_Kmm = tf.linalg.cholesky(self.kernel_space(self.inducing_space))
        C = tf.linalg.triangular_solve(chol_Kmm, H, adjoint=True)  # N x Ms x sd
        Knm = self.kernel_space(space_points, self.inducing_space)  # N x Ms
        return tf.reduce_sum(Knm[..., None] * C, axis=-2, keepdims=True)  # N x 1 x sd


class SpatioTemporalBase(MarkovFlowSparseModel, ABC):
    """
    Base class for Spatio-temporal GP regression using a factor kernel
    k_space_time((s,t),(s',t')) = k_time(t,t') * k_space(s,s')

    where k_time is a Markovian kernel.
    """

    def __init__(
        self,
        inducing_space,
        kernel_space: gpfk.Kernel,
        kernel_time: mfk.SDEKernel,
        likelihood: gpflow.likelihoods.Likelihood,
        mean_function: Optional[
            Union[markovflow.mean_function.MeanFunction, gpflow.mean_functions.MeanFunction]
        ] = None,
    ):
        """
        :param inducing_space: inducing space points [Ms, D]
        :param kernel_space: Gpflow space kernel
        :param kernel_time: Markovflow time kernel
        :param likelihood: a likelihood object
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        """
        super().__init__(self.__class__.__name__)

        self._mean_function = mean_function

        self._kernel_space = kernel_space
        self._kernel_time = kernel_time
        self._inducing_space = inducing_space
        self._likelihood = likelihood

        self.num_inducing_space = inducing_space.shape[0]

        self._kernel = SparseSpatioTemporalKernel(
            kernel_time=kernel_time, kernel_space=kernel_space, inducing_space=inducing_space
        )

    def space_time_predict_f(self, inputs):
        """
        Predict marginal function values at `inputs`. Note the
        time points should be sorted.

        :param inputs: Time point and associated spatial dimension to generate observations for,
         with shape
            ``batch_shape + [space_dim + 1, num_new_time_points]``.

        :return: Predicted mean and covariance for the new time points, with respective shapes
            ``batch_shape + [num_new_time_points, output_dim]`` and either
            ``batch_shape + [num_new_time_points, output_dim, output_dim]`` or
            ``batch_shape + [num_new_time_points, output_dim]``.
        """
        # split space and time dimension
        x, t = inputs[..., :-1], inputs[..., -1]

        # predict ssm at time points. latent ssm has state dimension : state_dim x Ms
        # and is projected to an output dimesion : Ms
        mean_u, cov_u = self.posterior.predict_f(new_time_points=t, full_output_cov=True)
        chol_cov_u = tf.linalg.cholesky(cov_u)  # Ms x Ms

        # use space conditional
        Kmn = self._kernel_space(self._inducing_space, x)  # Ms x N
        Kmm = self._kernel_space(self._inducing_space)  # Ms x Ms
        Knn = self._kernel_space(x, full_cov=False)  # N x 1

        mean_f, var_f = batch_base_conditional(
            Kmn, Kmm, Knn, tf.linalg.matrix_transpose(mean_u), q_sqrt=chol_cov_u
        )
        mean_f, var_f = mean_f[..., None], var_f[..., None]
        if self._mean_function is not None:
            mean_f += self._mean_function(inputs)

        return mean_f, var_f

    def loss(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Return the loss, which is the negative evidence lower bound (ELBO).

        :param input_data: A tuple of space-time points and observations containing the data
            at which to calculate the loss for training the model.
        """
        return -self.elbo(input_data)

    @property
    def posterior(self) -> PosteriorProcess:
        """ Posterior """
        raise NotImplementedError()

    @property
    def dist_q(self) -> StateSpaceModel:
        """ Posterior state space model on inducing states """
        raise NotImplementedError()

    @property
    def dist_p(self) -> StateSpaceModel:
        """ Prior state space model on inducing states """
        raise NotImplementedError()

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
        ve_fx = tf.reduce_sum(self._likelihood.variational_expectations(fx_mus, fx_covs, Y))
        # KL[q(s(z))|| p(s(z))]
        kl_fz = tf.reduce_sum(self.dist_q.kl_divergence(self.dist_p))
        # Return ELBO({fₓ, fz}) = VE(fₓ) - KL[q(s(z)) || p(s(z))]

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl_fz.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl_fz.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl_fz.dtype)
        return ve_fx * scale - kl_fz

    def predict_log_density(
        self, input_data: Tuple[tf.Tensor, tf.Tensor], full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Compute the log density of the data at the new data points.
        """
        X, Y = input_data
        f_mean, f_var = self.space_time_predict_f(X)
        return self._likelihood.predict_log_density(f_mean, f_var, Y)

    @property
    def kernel(self) -> SDEKernel:
        """
        Return the kernel of the GP.
        """
        return self._kernel

    @property
    def inducing_time(self) -> tf.Tensor:
        """
        Return the temporal inducing inputs of the model.
        """
        return self._inducing_time

    @property
    def inducing_space(self) -> tf.Tensor:
        """
        Return the spatial inducing inputs of the model.
        """
        return self._inducing_space


class SpatioTemporalSparseVariational(SpatioTemporalBase):
    """
    Model for Variational Spatio-temporal GP regression using a factor kernel
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

    for a fixed set of spatial inducing inputs zₛ
    p(x(zₛ, .)) is a continuous time process of state dimension Mₛd
    for a fixed time slice t, p(x(.,t)) ~ GP(0, kₛ)

    The following conditional independence holds:
    p(x(s,t) | x(zₛ, .)) = p(x(s,t) | s(zₛ, t)), i.e.,
    prediction at a new point at time t given x(zₛ, .) only depends on s(zₛ, t)

    This builds a spatially sparse process as
    q(x(.,.)) = q(x(zₛ, .)) p(x(.,.) |x(zₛ, .)),
    where the multi-output temporal process q(x(zₛ, .)) is also sparse
    q(x(zₛ, .)) = q(x(zₛ, zₜ)) p(x(zₛ,.) |x(zₛ,  zₜ))

    the marginal q(x(zₛ, zₜ)) is a multivariate Gaussian distribution
    parameterized as a state space model.
    """

    def __init__(
        self,
        inducing_space,
        inducing_time,
        kernel_space: gpfk.Kernel,
        kernel_time: mfk.SDEKernel,
        likelihood: gpflow.likelihoods.Likelihood,
        mean_function: Optional[
            Union[markovflow.mean_function.MeanFunction, gpflow.mean_functions.MeanFunction]
        ] = None,
        num_data=None,
    ):
        """
        :param inducing_space: inducing space points [Ms, D]
        :param inducing_time: inducing time points [Mt,]
        :param kernel_space: Gpflow space kernel
        :param kernel_time: Markovflow time kernel
        :param likelihood: a likelihood object
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        :param num_data: number of observations
        """

        super().__init__(inducing_space, kernel_space, kernel_time, likelihood, mean_function)

        self.num_data = num_data
        self._inducing_time = inducing_time
        self.num_inducing_time = inducing_time.shape[0]
        self._dist_p = self._kernel.state_space_model(self._inducing_time)
        self._dist_q = self.dist_p.create_trainable_copy()

        self._posterior = ConditionalProcess(
            posterior_dist=self.dist_q,
            kernel=self.kernel,
            conditioning_time_points=self.inducing_time,
        )

    @property
    def dist_q(self) -> StateSpaceModel:
        return self._dist_q

    @property
    def dist_p(self) -> StateSpaceModel:
        return self._dist_p

    @property
    def posterior(self) -> PosteriorProcess:
        """ Posterior process """
        return self._posterior


class SpatioTemporalSparseCVI(SpatioTemporalBase):
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

    This can be seen as the spatial extension of markovflow's SparseCVIGaussianProcess
    for temporal (only) Gaussian Processes.
    The inducing variables u(x,t) are now space and time dependent.

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

    the marginal q(x(zₛ, zₜ)) is parameterized as the product
    q(x(zₛ, zₜ)) = p(x(zₛ, zₜ)) t(x(zₛ, zₜ))
    where p(x(zₛ, zₜ)) is a state space model and t(x(zₛ, zₜ)) are sites.
    """

    def __init__(
        self,
        inducing_space,
        inducing_time,
        kernel_space: gpfk.Kernel,
        kernel_time: mfk.SDEKernel,
        likelihood: gpflow.likelihoods.Likelihood,
        mean_function: Optional[
            Union[markovflow.mean_function.MeanFunction, gpflow.mean_functions.MeanFunction]
        ] = None,
        num_data=None,
        learning_rate=0.1,
    ) -> None:
        """
        :param inducing_space: inducing space points [Ms, D]
        :param inducing_time: inducing time points [Mt,]
        :param kernel_space: Gpflow space kernel
        :param kernel_time: Markovflow time kernel
        :param likelihood: a likelihood object
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        :param num_data: The total number of observations.
            (relevant when feeding in external minibatches).
        :param learning_rate: the learning rate.
        """

        super().__init__(inducing_space, kernel_space, kernel_time, likelihood, mean_function)

        self.num_data = num_data
        self._inducing_time = inducing_time
        self.num_inducing_time = inducing_time.shape[0]

        self.learning_rate = learning_rate
        # initialize sites
        num_inducing = inducing_time.shape[0]
        state_dim = self._kernel.state_dim
        zeros1 = tf.zeros((num_inducing + 1, 2 * state_dim), dtype=default_float())
        zeros2 = tf.zeros((num_inducing + 1, 2 * state_dim, 2 * state_dim), dtype=default_float())
        self.nat1 = Parameter(zeros1)
        self.nat2 = Parameter(zeros2)

    @property
    def posterior(self) -> PosteriorProcess:
        """ Posterior object to predict outside of the training time points """
        return ConditionalProcess(
            posterior_dist=self.dist_q,
            kernel=self.kernel,
            conditioning_time_points=self.inducing_time,
        )

    @property
    def dist_q(self) -> StateSpaceModel:
        """
        Computes the variational posterior distribution on the vector of inducing states
        """
        # get prior precision
        prec = self.dist_p.precision

        # [..., num_transitions + 1, state_dim, state_dim]
        prec_diag = prec.block_diagonal
        # [..., num_transitions, state_dim, state_dim]
        prec_subdiag = prec.block_sub_diagonal

        sd = self.kernel.state_dim

        summed_lik_nat1_diag = self.nat1[..., 1:, :sd] + self.nat1[..., :-1, sd:]

        summed_lik_nat2_diag = self.nat2[..., 1:, :sd, :sd] + self.nat2[..., :-1, sd:, sd:]
        summed_lik_nat2_subdiag = self.nat2[..., 1:-1, sd:, :sd]

        # conjugate update of the natural parameter: post_nat = prior_nat + lik_nat
        theta_diag = -0.5 * prec_diag + summed_lik_nat2_diag
        theta_subdiag = -prec_subdiag + summed_lik_nat2_subdiag * 2.0

        post_ssm_params = naturals_to_ssm_params(
            theta_linear=summed_lik_nat1_diag, theta_diag=theta_diag, theta_subdiag=theta_subdiag
        )

        post_ssm = StateSpaceModel(
            state_transitions=post_ssm_params[0],
            state_offsets=post_ssm_params[1],
            chol_initial_covariance=post_ssm_params[2],
            chol_process_covariances=post_ssm_params[3],
            initial_mean=post_ssm_params[4],
        )
        return post_ssm

    @property
    def dist_p(self) -> StateSpaceModel:
        """
        Computes the prior distribution on the vector of inducing states
        """
        return self._kernel.state_space_model(self._inducing_time)

    def projection_inducing_states_to_observations(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        Compute the projection matrix from of the conditional mean of f(x,t) | s(t)
        :param input_data: Time point and associated spatial dimension to generate observations for,
         with shape ``batch_shape + [space_dim + 1, num_time_points]``.
        :return: The projection matrix with shape [num_time_points, obs_dim, num_inducing_time x state_dim ]
        """
        inputs, _ = input_data
        _, inputs_time = inputs[..., :-1], inputs[..., -1]
        # projection from nearby inducing states to current state (temporal)
        P, _ = conditional_statistics(inputs_time, self._inducing_time, self._kernel)
        # projection to state space to observation space (spatial)
        A = self._kernel.state_to_space_conditional_projection(inputs)
        return tf.einsum("ncs,nfc->nfs", P, A)

    def update_sites(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> None:
        """
        Perform one joint update of the Gaussian sites
                𝜽ₘ ← ρ𝜽ₘ + (1-ρ)𝐠ₘ

        Here 𝐠ₘ are the sum of the gradient of the variational expectation for each data point
        indexed k, projected back to the site vₘ = [uₘ, uₘ₊₁], through the conditional p(fₖ|vₘ)
        :param input_data: A tuple of time points and observations
        """
        inputs, observations = input_data
        _, inputs_time = inputs[..., :-1], inputs[..., -1]

        fx_mus, fx_covs = self.space_time_predict_f(inputs)

        # get gradient of variational expectations wrt mu, sigma
        _, grads = self.local_objective_and_gradients(fx_mus, fx_covs, inputs, observations)

        P = self.projection_inducing_states_to_observations(input_data)
        theta_linear, lik_nat2 = back_project_nats(grads[0], grads[1], P)

        # sum sites together
        indices = tf.searchsorted(self._inducing_time, inputs_time)
        num_partition = self._inducing_time.shape[0] + 1

        summed_theta_linear = tf.stack(
            [
                tf.reduce_sum(l, axis=0)
                for l in tf.dynamic_partition(theta_linear, indices, num_partitions=num_partition)
            ]
        )
        summed_lik_nat2 = tf.stack(
            [
                tf.reduce_sum(l, axis=0)
                for l in tf.dynamic_partition(lik_nat2, indices, num_partitions=num_partition)
            ]
        )

        # update
        lr = self.learning_rate
        new_nat1 = (1 - lr) * self.nat1 + lr * summed_theta_linear
        new_nat2 = (1 - lr) * self.nat2 + lr * summed_lik_nat2

        self.nat2.assign(new_nat2)
        self.nat1.assign(new_nat1)

    def local_objective_and_gradients(
        self, Fmu: tf.Tensor, Fvar: tf.Tensor, X: tf.Tensor, Y: tf.Tensor
    ) -> tf.Tensor:
        """
        Returs the local_objective and its gradients wrt to the expectation parameters
        :param Fmu: means μ [..., latent_dim]
        :param Fvar: variances σ² [..., latent_dim]
        :param X: inputs X [..., space_dim + 1]
        :param Y: observations Y [..., observation_dim]
        :return: local objective and gradient wrt [μ, σ² + μ²]
        """

        with tf.GradientTape() as g:
            g.watch([Fmu, Fvar])
            local_obj = tf.reduce_sum(input_tensor=self.local_objective(Fmu, Fvar, Y))
        grads = g.gradient(local_obj, [Fmu, Fvar])

        # turn into gradient wrt μ, σ² + μ²
        if self._mean_function is not None:
            Fmu -= self._mean_function(X)
        grads = gradient_transformation_mean_var_to_expectation([Fmu, Fvar], grads)

        return local_obj, grads

    def local_objective(self, Fmu: tf.Tensor, Fvar: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        """
        local loss in CVI
        :param Fmu: means [..., latent_dim]
        :param Fvar: variances [..., latent_dim]
        :param Y: observations [..., observation_dim]
        :return: local objective [...]
        """
        return self._likelihood.variational_expectations(Fmu, Fvar, Y)
