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
"""Module containing a model for CVI."""
from typing import Optional, Tuple

import tensorflow as tf
from gpflow.base import Parameter
from gpflow.likelihoods import Likelihood

from markovflow.kalman_filter import KalmanFilterWithSites, UnivariateGaussianSitesNat
from markovflow.kernels import SDEKernel
from markovflow.mean_function import MeanFunction, ZeroMeanFunction
from markovflow.models.models import MarkovFlowModel
from markovflow.posterior import ConditionalProcess
from markovflow.ssm_gaussian_transformations import naturals_to_ssm_params
from markovflow.state_space_model import StateSpaceModel


class GaussianProcessWithSitesBase(MarkovFlowModel):
    """
    Base class for site-based Gaussian Process approximation such as EP and CVI.

    The following notation is used:

        * :math:`x` - the time points of the training data
        * :math:`y` - observations corresponding to time points :math:`x`
        * :math:`s(.)` - the latent state of the Markov chain
        * :math:`f(.)` - the noise free predictions of the model
        * :math:`p(y | f)` - the likelihood
        * :math:`t(f)` - a site (indices will refer to the associated data point)
        * :math:`p(.)` - the prior distribution
        * :math:`q(.)` - the variational distribution

    We use the state space formulation of Markovian Gaussian Processes that specifies:

    * The conditional density of neighbouring latent states :math:`p(sₖ₊₁| sₖ)`
    * How to read out the latent process from these states :math:`fₖ = H sₖ`

    The likelihood links data to the latent process and :math:`p(yₖ | fₖ)`.
    We would like to approximate the posterior over the latent state space model of this model.

    We parameterize the approximate posterior using sites :math:`tₖ(fₖ)`:

    .. math:: q(s) = p(s) ∏ₖ tₖ(fₖ)

    ...where :math:`tₖ(fₖ)` are univariate Gaussian sites parameterized in the natural form:

    .. math:: t(f) = exp(𝜽ᵀφ(f) - A(𝜽))

    ...and where :math:`𝜽=[θ₁,θ₂]` and :math:`𝛗(f)=[f,f²]`.

    Here, :math:`𝛗(f)` are the sufficient statistics and :math:`𝜽` are the natural parameters.

    """

    def __init__(
        self,
        input_data: Tuple[tf.Tensor, tf.Tensor],
        kernel: SDEKernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
    ) -> None:
        """
        :param input_data: A tuple containing the observed data:

            * Time points of observations with shape ``batch_shape + [num_data]``
            * Observations with shape ``batch_shape + [num_data, observation_dim]``

        :param kernel: A kernel that defines a prior over functions.
        :param likelihood: A likelihood with shape ``batch_shape + [num_inducing]``.
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        """
        super().__init__(self.__class__.__name__)
        time_points, observations = input_data

        self._kernel = kernel
        if mean_function is None:
            mean_function = ZeroMeanFunction(obs_dim=1)
        self._mean_function = mean_function

        self._likelihood = likelihood
        self._time_points = time_points
        self._observations = observations

        # initialize sites
        self.sites = UnivariateGaussianSitesNat(
            nat1=Parameter(tf.zeros_like(observations)),
            nat2=Parameter(tf.ones_like(observations)[..., None] * -1e-10),
            log_norm=Parameter(tf.zeros_like(observations)),
        )

    @property
    def dist_q(self) -> StateSpaceModel:
        """
        Construct the :class:`~markovflow.state_space_model.StateSpaceModel` representation of
        the posterior process indexed at the time points.
        """
        # get prior precision
        prec = self.dist_p.precision

        # [..., num_transitions + 1, state_dim, state_dim]
        prec_diag = prec.block_diagonal
        # [..., num_transitions, state_dim, state_dim]
        prec_subdiag = prec.block_sub_diagonal

        H = self.kernel.generate_emission_model(self.time_points).emission_matrix
        bp_nat1, bp_nat2 = back_project_nats(self.sites.nat1, self.sites.nat2[..., 0], H)
        # conjugate update of the natural parameter: post_nat = prior_nat + lik_nat
        theta_diag = -0.5 * prec_diag + bp_nat2
        theta_subdiag = -prec_subdiag

        post_ssm_params = naturals_to_ssm_params(
            theta_linear=bp_nat1, theta_diag=theta_diag, theta_subdiag=theta_subdiag
        )

        return StateSpaceModel(
            state_transitions=post_ssm_params[0],
            state_offsets=post_ssm_params[1],
            chol_initial_covariance=post_ssm_params[2],
            chol_process_covariances=post_ssm_params[3],
            initial_mean=post_ssm_params[4],
        )

    @property
    def posterior_kalman(self) -> KalmanFilterWithSites:
        """Build the Kalman filter object from the prior state space models and the sites."""
        return KalmanFilterWithSites(
            state_space_model=self.dist_p,
            emission_model=self._kernel.generate_emission_model(self._time_points),
            sites=self.sites,
        )

    @property
    def posterior(self):
        """ Posterior object to predict outside of the training time points """
        return ConditionalProcess(
            posterior_dist=self.dist_q,
            kernel=self.kernel,
            conditioning_time_points=self.conditioning_points,
        )

    def log_likelihood(self) -> tf.Tensor:
        """
        Calculate :math:`log p(y)`.

        :return: A scalar tensor representing the ELBO.
        """
        return self.posterior_kalman.log_likelihood()

    @property
    def time_points(self) -> tf.Tensor:
        """
        Return the time points of the observations.

        :return: A tensor with shape ``batch_shape + [num_data]``.
        """
        return self._time_points

    @property
    def conditioning_points(self) -> tf.Tensor:
        """
        Return the time points of the observations.

        :return: A tensor with shape ``batch_shape + [num_data]``.
        """
        return self._time_points

    @property
    def observations(self) -> tf.Tensor:
        """
        Return the observations.

        :return: A tensor with shape ``batch_shape + [num_data, observation_dim]``.
        """
        return self._observations

    @property
    def kernel(self) -> SDEKernel:
        """
        Return the kernel.
        """
        return self._kernel

    @property
    def likelihood(self) -> Likelihood:
        """
        Return the likelihood.
        """
        return self._likelihood

    @property
    def mean_function(self) -> MeanFunction:
        """
        Return the mean function.
        """
        return self._mean_function

    @property
    def dist_p(self) -> StateSpaceModel:
        """
        Return the prior Gauss-Markov distribution.
        """
        return self._kernel.state_space_model(self.conditioning_points)

    def loss(self) -> tf.Tensor:
        """
        Return the loss, which is the negative ELBO.
        """
        return -self.log_likelihood()


class CVIGaussianProcess(GaussianProcessWithSitesBase):
    """
    Provides an alternative parameterization to a
    :class:`~markovflow.models.variational.VariationalGaussianProcess`.

    This class approximates the posterior of a model with a GP prior and a general likelihood
    using a Gaussian posterior parameterized with Gaussian sites.

    The following notation is used:

        * :math:`x` - the time points of the training data
        * :math:`y` - observations corresponding to time points :math:`x`
        * :math:`s(.)` - the latent state of the Markov chain
        * :math:`f(.)` - the noise free predictions of the model
        * :math:`p(y | f)` - the likelihood
        * :math:`t(f)` - a site (indices will refer to the associated data point)
        * :math:`p(.)` the prior distribution
        * :math:`q(.)` the variational distribution

    We use the state space formulation of Markovian Gaussian Processes that specifies:

    * The conditional density of neighbouring latent states :math:`p(sₖ₊₁| sₖ)`
    * How to read out the latent process from these states :math:`fₖ = H sₖ`

    The likelihood links data to the latent process and :math:`p(yₖ | fₖ)`.
    We would like to approximate the posterior over the latent state space model of this model.

    To approximate the posterior, we maximise the evidence lower bound (ELBO) :math:`ℒ` with
    respect to the parameters of the variational distribution, since:

    .. math:: log p(y) = ℒ(q) + KL[q(s) ‖ p(s | y)]

    ...where:

    .. math:: ℒ(q) = ∫ log(p(s, y) / q(s)) q(s) ds

    We parameterize the variational posterior through sites :math:`tₖ(fₖ)`:

    .. math:: q(s) = p(s) ∏ₖ tₖ(fₖ)

    ...where :math:`tₖ(fₖ)` are univariate Gaussian sites parameterized in the natural form:

    .. math:: t(f) = exp(𝜽ᵀφ(f) - A(𝜽))

    ...and where :math:`𝜽=[θ₁,θ₂]` and :math:`𝛗(f)=[f,f²]`.

    Here, :math:`𝛗(f)` are the sufficient statistics and :math:`𝜽` are the natural parameters.
    Note that the subscript :math:`k` has been omitted for simplicity.

    The natural gradient update of the sites can be shown to be the gradient of the
    variational expectations:

    .. math:: 𝐠 = ∇[𝞰][∫ log(p(y=Y|f)) q(f) df]

    ...with respect to the expectation parameters:

    .. math:: 𝞰 = E[𝛗(f)] = [μ, σ² + μ²]

    That is, :math:`𝜽 ← ρ𝜽 + (1-ρ)𝐠`, where :math:`ρ` is the learning rate.

    The key reference is::

      @inproceedings{khan2017conjugate,
        title={Conjugate-Computation Variational Inference: Converting Variational Inference
               in Non-Conjugate Models to Inferences in Conjugate Models},
        author={Khan, Mohammad and Lin, Wu},
        booktitle={Artificial Intelligence and Statistics},
        pages={878--887},
        year={2017}
      }
    """

    def __init__(
        self,
        input_data: Tuple[tf.Tensor, tf.Tensor],
        kernel: SDEKernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        learning_rate=0.1,
    ) -> None:
        """
        :param input_data: A tuple containing the observed data:

            * Time points of observations with shape ``batch_shape + [num_data]``
            * Observations with shape ``batch_shape + [num_data, observation_dim]``

        :param kernel: A kernel that defines a prior over functions.
        :param likelihood: A likelihood with shape ``batch_shape + [num_inducing]``.
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        :param learning_rate: The learning rate of the algorithm.
        """
        super().__init__(
            input_data=input_data, kernel=kernel, likelihood=likelihood, mean_function=mean_function
        )
        self.learning_rate = learning_rate

    def local_objective(self, Fmu: tf.Tensor, Fvar: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        """
        Calculate local loss in CVI.

        :param Fmu: Means with shape ``[..., latent_dim]``.
        :param Fvar: Variances with shape ``[..., latent_dim]``.
        :param Y: Observations with shape ``[..., observation_dim]``.
        :return: A local objective with shape ``[...]``.
        """
        return self._likelihood.variational_expectations(Fmu, Fvar, Y)

    def local_objective_and_gradients(self, Fmu: tf.Tensor, Fvar: tf.Tensor) -> tf.Tensor:
        """
        Return the local objective and its gradients with regard to the expectation parameters.

        :param Fmu: Means :math:`μ` with shape ``[..., latent_dim]``.
        :param Fvar: Variances :math:`σ²` with shape ``[..., latent_dim]``.
        :return: A local objective and gradient with regard to :math:`[μ, σ² + μ²]`.
        """
        with tf.GradientTape() as g:
            g.watch([Fmu, Fvar])
            local_obj = tf.reduce_sum(
                input_tensor=self.local_objective(Fmu, Fvar, self._observations)
            )
        grads = g.gradient(local_obj, [Fmu, Fvar])
        # turn into gradient wrt μ, σ² + μ²
        grads = gradient_transformation_mean_var_to_expectation((Fmu, Fvar), grads)

        return local_obj, grads

    def update_sites(self) -> None:
        """
        Perform one joint update of the Gaussian sites. That is:

        .. math:: 𝜽 ← ρ𝜽 + (1-ρ)𝐠
        """
        fx_mus, fx_covs = self.posterior.predict_f(self.time_points)

        # get gradient of variational expectations wrt the expectation parameters μ, σ² + μ²
        _, grads = self.local_objective_and_gradients(fx_mus, fx_covs)

        # update
        lr = self.learning_rate
        new_nat1 = (1 - lr) * self.sites.nat1 + lr * grads[0]
        new_nat2 = (1 - lr) * self.sites.nat2 + lr * grads[1][..., None]

        self.sites.nat2.assign(new_nat2)
        self.sites.nat1.assign(new_nat1)

    def elbo(self) -> tf.Tensor:
        """
        Calculate the evidence lower bound (ELBO) :math:`log p(y)`.

        This is done by computing the marginal of the model in which the likelihood terms were
        replaced by the Gaussian sites.

        :return: A scalar tensor representing the ELBO.
        """
        return self.log_likelihood()

    def classic_elbo(self) -> tf.Tensor:
        """
        Compute the ELBO the classic way. That is:

        .. math:: ℒ(q) = Σᵢ ∫ log(p(yᵢ | f)) q(f) df - KL[q(f) ‖ p(f)]

        .. note:: This is mostly for testing purposes and should not be used for optimization.

        :return: A scalar tensor representing the ELBO.
        """
        # s ~ q(s) = N(μ, P)
        # Project to function space, fₓ = H*s ~ q(fₓ)
        fx_mus, fx_covs = self.posterior.predict_f(self.time_points)
        # VE(fₓ) = Σᵢ ∫ log(p(yᵢ | fₓ)) q(fₓ) dfₓ
        ve_fx = tf.reduce_sum(
            input_tensor=self._likelihood.variational_expectations(
                fx_mus, fx_covs, self._observations
            )
        )
        # KL[q(sₓ) || p(sₓ)]

        kl_fx = tf.reduce_sum(self.dist_q.kl_divergence(self.dist_p))
        # Return ELBO(fₓ) = VE(fₓ) - KL[q(sₓ) || p(sₓ)]
        return ve_fx - kl_fx

    def predict_log_density(
        self, input_data: Tuple[tf.Tensor, tf.Tensor], full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Compute the log density of the data at the new data points.
        :param input_data: A tuple of time points and observations containing the data at which
            to calculate the loss for training the model:
            a tensor of inputs with shape ``batch_shape + [num_data]``,
            a tensor of observations with shape ``batch_shape + [num_data, observation_dim]``.
        :param full_output_cov: Either full output covariance (`True`) or marginal
            variances (`False`).
        """
        X, Y = input_data
        f_mean, f_var = self.posterior.predict_f(X, full_output_cov)
        return self.likelihood.predict_log_density(f_mean, f_var, Y)


def back_project_nats(nat1, nat2, C):
    """
    Transforms the natural parameters 𝜽f of a Gaussian with sufficient statistics 𝛗(f)=[f,f²]
    into equivalent rank one natural parameters 𝜽g of a (thus degenerate) Gaussian
    with sufficient statistics 𝛗(g)=[g,ggᵀ] where f = Cg.

    In practice  [θg₁, θg₂] = [θf₁ C,θf₂ CᵀC]

    :param nat1: natural parameters with size (num_time_points, 1)
    :param nat2: natural parameters with size (num_time_points, 1)
    :param C: projection with size (num_time_points, 1, project_dim)
    :return: natural parameters with size
            (num_time_points, project_dim)
            (num_time_points, project_dim, project_dim)

    """
    shape_constraints = [(nat1, ["N", 1]), (nat2, ["N", 1]), (C, ["N", 1, "D"])]
    tf.debugging.assert_shapes(shape_constraints)
    # [..., num_transitions, project_dim], performing CT @ nat1f
    bp_nat1 = tf.reduce_sum(C * nat1[..., None], axis=-2)
    # [..., num_transitions, project_dim, project_dim], performing CT @ nat2f @ C
    bp_nat2 = tf.reduce_sum(nat2[..., None, None] * C[..., None] * C[..., None, :], axis=-3)
    return bp_nat1, bp_nat2


def gradient_transformation_mean_var_to_expectation(
    inputs: Tuple[tf.Tensor, tf.Tensor], grads: Tuple[tf.Tensor, tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Transform gradients.

    This is from :math:`𝐠` of a function with regard to :math:`[μ, σ²]` into its gradients
    with regard to :math:`[μ, σ² + μ²]`.

    :param inputs: Means and variances :math:`[μ, σ²]`.
    :param grads: Gradients :math:`𝐠`.
    """
    return grads[0] - 2.0 * grads[1] * inputs[0], grads[1]
