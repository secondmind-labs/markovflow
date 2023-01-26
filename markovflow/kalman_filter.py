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
"""Module containing a Kalman filter."""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import gpflow
from gpflow import default_float
from gpflow.base import TensorType

from markovflow.block_tri_diag import SymmetricBlockTriDiagonal
from markovflow.emission_model import EmissionModel
from markovflow.state_space_model import StateSpaceModel
from markovflow.utils import tf_scope_class_decorator


class BaseKalmanFilter(gpflow.Module, ABC):
    r"""
    Performs a Kalman filter on a :class:`~markovflow.state_space_model.StateSpaceModel` and
    :class:`~markovflow.emission_model.EmissionModel`, with given observations.

    The key reference is::

        @inproceedings{grigorievskiy2017parallelizable,
            title={Parallelizable sparse inverse formulation Gaussian processes (SpInGP)},
            author={Grigorievskiy, Alexander and Lawrence, Neil and S{\"a}rkk{\"a}, Simo},
            booktitle={Int'l Workshop on Machine Learning for Signal Processing (MLSP)},
            pages={1--6},
            year={2017},
            organization={IEEE}
        }

    The following notation from the above paper is used:

        * :math:`G = I_N ⊗ H`, where :math:`⊗` is the Kronecker product
        * :math:`R` is the observation covariance
        * :math:`Σ = I_N ⊗ R`
        * :math:`K⁻¹ = A⁻ᵀQ⁻¹A⁻¹` is the precision, where :math:`A⁻ᵀ =  [Aᵀ]⁻¹ = [A⁻¹]ᵀ`
        * :math:`L` is the Cholesky of :math:`K⁻¹ + GᵀΣ⁻¹G`. That is, :math:`LLᵀ = K⁻¹ + GᵀΣ⁻¹G`
        * :math:`y` is the observation matrix
    """

    def __init__(self, state_space_model: StateSpaceModel, emission_model: EmissionModel,) -> None:
        """
        :param state_space_model: Parametrises the latent chain.
        :param emission_model: Maps the latent chain to the observations.
        """
        super().__init__(self.__class__.__name__)
        # verify observation shape
        self.prior_ssm = state_space_model
        self.emission = emission_model

    @property
    @abstractmethod
    def _r_inv(self):
        """ Precision of observation model """
        raise NotImplementedError

    @property
    @abstractmethod
    def observations(self):
        """ Observation vector """
        raise NotImplementedError

    @property
    def _k_inv_prior(self):
        """ Prior precision """
        return self.prior_ssm.precision

    @property
    def _k_inv_post(self):
        """ Posterior precision """

        # construct the likelihood precision: Gᵀ Σ⁻¹ G
        # HᵀR⁻¹H [state_dim, state_dim]
        h_t_r_h = tf.einsum(
            "...ji,...jk,...kl->...il",
            self.emission.emission_matrix,
            self._r_inv,
            self.emission.emission_matrix,
        )
        # The emission matrix is tiled across the time_points, so for a time invariant matrix
        # this is equivalent to Gᵀ Σ⁻¹ G = (I_N ⊗ HᵀR⁻¹H),
        likelihood_precision = SymmetricBlockTriDiagonal(h_t_r_h)
        # K⁻¹ + GᵀΣ⁻¹G
        return self._k_inv_prior + likelihood_precision

    @property
    def _log_det_observation_precision(self):
        """ Sum of log determinant of the precisions of the observation model """
        num_data = self.prior_ssm.num_transitions + 1
        return tf.cast(num_data, default_float()) * tf.linalg.logdet(self._r_inv)

    def posterior_state_space_model(self) -> StateSpaceModel:
        r"""
        Return the posterior as a state space model.

        The marginal means and covariances are given by:

        .. math::
            &μ(Χ) = (K⁻¹ + GᵀΣ⁻¹G)⁻¹[GᵀΣ⁻¹y + K⁻¹μ]\\
            &P(X) = K⁻¹ + GᵀΣ⁻¹G

        ...where :math:`μ` is a block vector of the marginal means.

        We can derive the state transitions :math:`aₖ` and process noise covariances :math:`qₖ`
        from the block tridiagonal matrix (see
        :meth:`~markovflow.block_tri_diag.SymmetricBlockTriDiagonal.upper_diagonal_lower`).
        Lower case is used to attempt to distinguish the posterior and prior parameters.

        We then need to calculate :math:`μ₀` and :math:`bₖ` (this is what most of the code in
        this function does). This can be calculated from:

        .. math:: K⁻¹ₚₒₛₜμₚₒₛₜ = GᵀΣ⁻¹y + K⁻¹ₚᵣᵢₒᵣμₚᵣᵢₒᵣ

        Firstly, we use that for any :class:`~markovflow.state_space_model.StateSpaceModel`:

        .. math:: K⁻¹μ = A⁻ᵀ Q⁻¹ m

        ...where :math:`m = [μ₀, b₁,... bₙ]` and::

            A⁻¹ =  [ I             ]      Q⁻¹ =  [ P₀⁻¹          ]
                   [-A₁, I         ]            [    Q₁⁻¹       ]
                   [    -A₂, I     ]            [       ᨞      ]
                   [         ᨞  ᨞  ]            [         ᨞    ]
                   [         -Aₙ, I]            [           Qₙ⁻¹]

        So:

        .. math:: mₚₒₛₜ = Qₚₒₛₜ Aₚₒₛₜᵀ [GᵀΣ⁻¹y + Kₚᵣᵢₒᵣ⁻¹mₚᵣᵢₒᵣ]

        :return: The posterior as a state space model.
        """
        a_inv_post, chol_q_inv_post = self._k_inv_post.upper_diagonal_lower()
        assert a_inv_post.block_sub_diagonal is not None

        # (GᵀΣ⁻¹)y [..., num_transitions + 1, state_dim]
        obs_proj = self._back_project_y_to_state(self.observations)

        # Kₚᵣᵢₒᵣ⁻¹μₚᵣᵢₒᵣ (prior parameters) [..., num_transitions + 1, state_dim]
        k_inv_mu_prior = self._k_inv_prior.dense_mult(self.prior_ssm.marginal_means)

        # mₚₒₛₜ = Qₚₒₛₜ Aₚₒₛₜᵀ [GᵀΣ⁻¹y + Kₚᵣᵢₒᵣ⁻¹mₚᵣᵢₒᵣ] [..., num_transitions + 1, state_dim]
        m_post = chol_q_inv_post.solve(
            chol_q_inv_post.solve(a_inv_post.solve(obs_proj + k_inv_mu_prior, transpose_left=True)),
            transpose_left=True,
        )

        # [..., num_transitions + 1, state_dim, state_dim]
        batch_shape = tf.concat(
            [self.prior_ssm.batch_shape, tf.TensorShape([self.prior_ssm.num_transitions + 1])],
            axis=0,
        )
        identities = tf.eye(self.prior_ssm.state_dim, dtype=m_post.dtype, batch_shape=batch_shape)

        # cholesky of [P₀, Q₁, Q₂, ....] [..., num_transitions + 1, state_dim, state_dim]
        concatted_qs = tf.linalg.cholesky(
            tf.linalg.cholesky_solve(chol_q_inv_post.block_diagonal, identities)
        )

        return StateSpaceModel(
            initial_mean=m_post[..., 0, :],
            chol_initial_covariance=concatted_qs[..., 0, :, :],
            state_transitions=-a_inv_post.block_sub_diagonal,
            state_offsets=m_post[..., 1:, :],
            chol_process_covariances=concatted_qs[..., 1:, :, :],
        )

    def log_likelihood(self) -> tf.Tensor:
        r"""
        Construct a TensorTlow function to compute the likelihood.

        We set :math:`y = obs - Hμ` (where :math:`μ` is the vector of marginal state means):

        .. math::
            log p(obs|params) = &- ᴺ⁄₂log(2π) - ½(log |K⁻¹ + GᵀΣ⁻¹G| - log |K⁻¹| - log |Σ⁻¹|)\\
                                &- ½ yᵀ(Σ⁻¹ - Σ⁻¹G(K⁻¹ + GᵀΣ⁻¹G)⁻¹GᵀΣ⁻¹)y

        ...where :math:`N` is the dimensionality of the precision object, that is
        ``state_dim * (num_transitions + 1)``.

        We break up the log likelihood as: cst + term1 + term2 + term3. That is, as:

            * cst: :math:`- ᴺ⁄₂log(2π)`
            * term 1: :math:`- ½ yᵀΣ⁻¹y`
            * term 2:

              .. math::
                 ½ yᵀΣ⁻¹G(K⁻¹ + GᵀΣ⁻¹G)⁻¹GᵀΣ⁻¹)y = ½ yᵀΣ⁻¹G(LLᵀ)⁻¹GᵀΣ⁻¹)y = ½|L⁻¹(GᵀΣ⁻¹)y|²

            * term 3:

              .. math::
                 - ½(log |K⁻¹ + GᵀΣ⁻¹G| - log |K⁻¹| - log |Σ⁻¹|) = ½log |K⁻¹| - log |L| + ½log |Σ⁻¹|

        Note that there are a couple of mistakes in the SpinGP paper for this formula (18):

            * They have :math:`- ½(... + log |Σ⁻¹|)`. It should be :math:`- ½(... - log |Σ⁻¹|)`
            * They have :math:`- ½ yᵀ(... Σ⁻¹G(K⁻¹ + GᵀΣ⁻¹G)⁻¹)y`. It should
              be :math:`- ½ yᵀ(... Σ⁻¹G(K⁻¹ + GᵀΣ⁻¹G)⁻¹GᵀΣ⁻¹)y`

        :return: The likelihood as a scalar tensor (we sum over the `batch_shape`).
        """
        # K⁻¹ + GᵀΣ⁻¹G = LLᵀ.
        l_post = self._k_inv_post.cholesky
        num_data = self.prior_ssm.num_transitions + 1
        # Hμ [..., num_transitions + 1, output_dim]
        marginal = self.emission.project_state_to_f(self.prior_ssm.marginal_means)

        # y = obs - Hμ [..., num_transitions + 1, output_dim]
        disp = self.observations - marginal

        # cst is the constant term for a gaussian log likelihood
        cst = (
            -0.5 * np.log(2 * np.pi) * tf.cast(self.emission.output_dim * num_data, default_float())
        )

        # term1 is: - ½ yᵀΣ⁻¹y shape [...]
        term1 = -0.5 * tf.reduce_sum(
            input_tensor=tf.einsum("...op,...p,...o->...o", self._r_inv, disp, disp), axis=[-1, -2]
        )

        # term 2 is: ½|L⁻¹(GᵀΣ⁻¹)y|²
        # (GᵀΣ⁻¹)y [..., num_transitions + 1, state_dim]
        obs_proj = self._back_project_y_to_state(disp)

        # ½|L⁻¹(GᵀΣ⁻¹)y|² [...]
        term2 = 0.5 * tf.reduce_sum(
            input_tensor=tf.square(l_post.solve(obs_proj, transpose_left=False)), axis=[-1, -2]
        )

        ## term 3 is: ½log |K⁻¹| - log |L| + ½ log |Σ⁻¹|
        # where log |Σ⁻¹| = num_data * log|R⁻¹|
        term3 = (
            0.5 * self.prior_ssm.log_det_precision()
            - l_post.abs_log_det()
            + 0.5 * self._log_det_observation_precision
        )

        return tf.reduce_sum(cst + term1 + term2 + term3)

    def _back_project_y_to_state(self, observations: tf.Tensor) -> tf.Tensor:
        """
        Back project from the observation space to the state_space, i.e. calculate (GᵀΣ⁻¹)y.

        :param observations: a tensor y of shape
                    batch_shape + [num_data, output_dim]
        :return: a tensor (GᵀΣ⁻¹)y of shape
                    batch_shape + [num_data, state_dim]
        """
        # GᵀΣ⁻¹, batch_shape + [num_data, output_dim, state_dim]
        back_projection = tf.einsum(
            "...ij,...ki->...kj", self.emission.emission_matrix, self._r_inv
        )
        # (GᵀΣ⁻¹) y
        return tf.einsum("...ij,...i->...j", back_projection, observations)


@tf_scope_class_decorator
class KalmanFilter(BaseKalmanFilter):
    r"""
    Performs a Kalman filter on a :class:`~markovflow.state_space_model.StateSpaceModel` and
    :class:`~markovflow.emission_model.EmissionModel`, with given observations.

    The key reference is::

        @inproceedings{grigorievskiy2017parallelizable,
            title={Parallelizable sparse inverse formulation Gaussian processes (SpInGP)},
            author={Grigorievskiy, Alexander and Lawrence, Neil and S{\"a}rkk{\"a}, Simo},
            booktitle={Int'l Workshop on Machine Learning for Signal Processing (MLSP)},
            pages={1--6},
            year={2017},
            organization={IEEE}
        }

    The following notation from the above paper is used:

        * :math:`G = I_N ⊗ H`, where :math:`⊗` is the Kronecker product
        * :math:`R` is the observation covariance
        * :math:`Σ = I_N ⊗ R`
        * :math:`K⁻¹ = A⁻ᵀQ⁻¹A⁻¹` is the precision, where :math:`A⁻ᵀ =  [Aᵀ]⁻¹ = [A⁻¹]ᵀ`
        * :math:`L` is the Cholesky of :math:`K⁻¹ + GᵀΣ⁻¹G`. That is, :math:`LLᵀ = K⁻¹ + GᵀΣ⁻¹G`
        * :math:`y` is the observation matrix
    """

    def __init__(
        self,
        state_space_model: StateSpaceModel,
        emission_model: EmissionModel,
        observations: tf.Tensor,
        chol_obs_covariance: TensorType,
    ) -> None:
        """
        :param state_space_model: Parametrises the latent chain.
        :param emission_model: Maps the latent chain to the observations.
        :param observations: Data with shape ``[num_transitions + 1, output_dim]``.
        :param chol_obs_covariance: A :data:`~markovflow.base.TensorType` with shape
            ``[output_dim, output_dim]`` for the Cholesky factor of the covariance to be
            applied to :math:`f` from `emission_model`.
        """
        super().__init__(state_space_model, emission_model)

        assert isinstance(observations, tf.Tensor)

        # verify observation covariance shape
        shape = tf.convert_to_tensor([emission_model.output_dim, emission_model.output_dim])
        message = """The shape of the observation covaraiance matrix and the emission
                     matrix are not compatible"""
        tf.debugging.assert_equal(tf.shape(chol_obs_covariance), shape, message=message)

        # verify observation shape
        message = """The shape of the observations and the state-space-model parameters
                     are not compatible"""
        shape = tf.concat(
            [
                state_space_model.batch_shape,
                [state_space_model.num_transitions + 1, emission_model.output_dim],
            ],
            axis=0,
        )
        tf.debugging.assert_equal(tf.shape(observations), shape, message=message)

        self._chol_obs_covariance = chol_obs_covariance  # To collect gpflow.Module trainables
        self._observations = observations  # batch_shape + [num_transitions + 1, output_dim]

    @property
    def _r_inv(self):
        """ Precision of the observation model """
        # [output_dim, output_dim]
        return tf.linalg.cholesky_solve(
            self._chol_obs_covariance,
            tf.eye(self.emission.output_dim, dtype=self._chol_obs_covariance.dtype),
        )

    @property
    def observations(self):
        """ Observation vector """
        return self._observations


class GaussianSites(gpflow.Module, ABC):
    """
    This class is a wrapper around the parameters specifying multiple independent
    Gaussian distributions.
    """

    @property
    def means(self):
        """
        Return the means of the Gaussians.
        """
        raise NotImplementedError

    @property
    def precisions(self):
        """
        Return the precisions of the Gaussians.
        """
        raise NotImplementedError

    @property
    def log_det_precisions(self):
        """ Return the sum of the log determinant of the observation precisions."""
        raise NotImplementedError


class UnivariateGaussianSitesNat(GaussianSites):
    """
    This class is a wrapper around parameters of univariate Gaussian distributions
    in the natural form. That is:

    .. math:: p(f) = exp(𝞰ᵀφ(f) - A(𝞰))

    ...where :math:`𝞰=[η₁,η₂]` and :math:`𝛗(f)=[f,f²]`.

    The mean :math:`μ` and variance :math:`σ²` parameterization is such that:

    .. math:: μ = -½η₁/η₂, σ²=-½η₂⁻¹
    """

    def __init__(self, nat1, nat2, log_norm=None):
        """
        :param nat1: first natural parameter [N, D]
        :param nat2: second natural parameter [N, D, D]
        :param log_norm: normalizer parameter [N, D]
        """
        super().__init__()
        shape_constraints = [
            (nat1, ["N", 1]),
            (nat2, ["N", 1, 1]),
        ]
        if log_norm is not None:
            shape_constraints += [(log_norm, ["N", 1])]
        tf.debugging.assert_shapes(shape_constraints)

        self.num_data, self.output_dim = nat1.shape
        self.nat1 = nat1
        self.nat2 = nat2
        self.log_norm = log_norm

    @property
    def means(self):
        """
        Return the means of the Gaussians.
        """
        return -0.5 * self.nat1 / self.nat2[..., 0]

    @property
    def precisions(self):
        """
        Return the precisions of the Gaussians.
        """
        return -2 * self.nat2

    @property
    def log_det_precisions(self):
        """ Return the sum of the log determinant of the observation precisions. """
        return tf.math.log(-2 * self.nat2)


@tf_scope_class_decorator
class KalmanFilterWithSites(BaseKalmanFilter):

    r"""
    Performs a Kalman filter on a :class:`~markovflow.state_space_model.StateSpaceModel` and
    :class:`~markovflow.emission_model.EmissionModel`, with Gaussian sites,
    that is time dependent Gaussian Likelihood terms.

    The key reference is::

        @inproceedings{grigorievskiy2017parallelizable,
            title={Parallelizable sparse inverse formulation Gaussian processes (SpInGP)},
            author={Grigorievskiy, Alexander and Lawrence, Neil and S{\"a}rkk{\"a}, Simo},
            booktitle={Int'l Workshop on Machine Learning for Signal Processing (MLSP)},
            pages={1--6},
            year={2017},
            organization={IEEE}
        }

    The following notation from the above paper is used:

        * :math:`G = I_N ⊗ H`, where :math:`⊗` is the Kronecker product
        * :math:`R = [R₁, R₂, ... Rₙ]` is the observation covariance
        * :math:`Σ = blockdiag[R]`
        * :math:`K⁻¹ = A⁻ᵀQ⁻¹A⁻¹` is the precision, where :math:`A⁻ᵀ =  [Aᵀ]⁻¹ = [A⁻¹]ᵀ`
        * :math:`L` is the Cholesky of :math:`K⁻¹ + GᵀΣ⁻¹G`. That is, :math:`LLᵀ = K⁻¹ + GᵀΣ⁻¹G`
        * :math:`y` is the observation matrix
    """

    def __init__(
        self,
        state_space_model: StateSpaceModel,
        emission_model: EmissionModel,
        sites: GaussianSites,
    ) -> None:
        """
        :param state_space_model: Parametrises the latent chain.
        :param emission_model: Maps the latent chain to the observations.
        :param sites: Gaussian sites parameterizing the Gaussian likelihoods.
        """
        # verify site shape
        message = """The shape of the site matrices and the emission
                     matrix are not compatible"""
        tf.debugging.assert_equal(sites.output_dim, emission_model.output_dim, message=message)
        self.sites = sites

        super().__init__(state_space_model, emission_model)

    @property
    def _r_inv(self):
        """ Precisions of the observation model """
        return self.sites.precisions

    @property
    def _log_det_observation_precision(self):
        """ Sum of log determinant of the precisions of the observation model """
        return tf.reduce_sum(tf.linalg.logdet(self._r_inv), axis=-1)

    @property
    def observations(self):
        """ Observation vector """
        return self.sites.means


@tf_scope_class_decorator
class KalmanFilterWithSparseSites(BaseKalmanFilter):
    r"""
    Performs a Kalman filter on a :class:`~markovflow.state_space_model.StateSpaceModel`
    and :class:`~markovflow.emission_model.EmissionModel`, with Gaussian sites, over a time grid.
    """

    def __init__(self, state_space_model: StateSpaceModel, emission_model: EmissionModel, sites: GaussianSites,
                 num_grid_points: int, observations_index: tf.Tensor, observations: tf.Tensor):
        """
        :param state_space_model: Parameterises the latent chain.
        :param emission_model: Maps the latent chain to the observations.
        :param sites: Gaussian sites over the observations.
        :param num_grid_points: number of grid points.
        :param observations_index: Index of the observations in the time grid with shape (N,).
        :param observations: Sparse observations with shape [n_batch] + (N, output_dim).
        """
        self.sites = sites
        self.observations_index = observations_index
        self.sparse_observations = self._drop_batch_shape(observations)
        self.grid_shape = tf.TensorShape((num_grid_points, 1))
        super().__init__(state_space_model, emission_model)

    @property
    def _r_inv(self):
        """
        Precisions of the observation model over the time grid.
        """
        data_sites_precision = self.sites.precisions
        return self.sparse_to_dense(data_sites_precision, output_shape=self.grid_shape + (1,))

    def _drop_batch_shape(self, tensor: tf.Tensor):
        """
        Check the batch, if present, is equal to 1, and drop it.
        """
        tensor_shape = tensor._shape_as_list()
        if len(tensor_shape) < 3: return tensor
        if tensor_shape[0] != 1: raise Exception("KalmanFilterWithSparseSites doesn't support batches")

        return tf.squeeze(tensor, axis=0)

    @property
    def _log_det_observation_precision(self):
        """
        Sum of log determinant of the precisions of the observation model. It only calculates for the data_sites as
        other sites precision is anyways zero.
        """
        return tf.reduce_sum(tf.linalg.logdet(self._r_inv_data), axis=-1)

    @property
    def observations(self):
        """ Sparse observation vector """
        return self.sparse_to_dense(self.sparse_observations, self.grid_shape)

    @property
    def _r_inv_data(self):
        """
        Precisions of the observation model for only the data sites.
        """
        return self.sites.precisions

    def sparse_to_dense(self, tensor: tf.Tensor, output_shape: tf.TensorShape) -> tf.Tensor:
        """
        Convert a sparse tensor to a dense one on the basis of observations index, output tensor is of the output_shape.
        """
        return tf.scatter_nd(self.observations_index, tensor, output_shape)

    def dense_to_sparse(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Convert a dense tensor to a sparse one on the basis of observations index.
        """
        tensor_shape = tensor.shape
        expand_dims = len(tensor_shape) == 3

        tensor = tf.gather_nd(tf.reshape(tensor, (-1, 1)), self.observations_index)
        if expand_dims:
            tensor = tf.expand_dims(tensor, axis=-1)
        return tensor

    def log_likelihood(self) -> tf.Tensor:
        r"""
        Construct a TensorFlow function to compute the likelihood.

        For more mathematical details, look at the log_likelihood function of the parent class.
        The main difference from the parent class are that the vector of observations is now sparse.        

        :return: The likelihood as a scalar tensor (we sum over the `batch_shape`).
        """
        # K⁻¹ + GᵀΣ⁻¹G = LLᵀ.
        l_post = self._k_inv_post.cholesky
        num_data = self.observations_index.shape[0]

        # Hμ [..., num_transitions + 1, output_dim]
        marginal = self.emission.project_state_to_f(self.prior_ssm.marginal_means)
        marginal = self._drop_batch_shape(marginal)

        # y = obs - Hμ [..., num_transitions + 1, output_dim]
        disp = self.observations - marginal
        disp_data = self.sparse_observations - self.dense_to_sparse(marginal)

        # cst is the constant term for a gaussian log likelihood
        cst = (
                -0.5 * np.log(2 * np.pi) * tf.cast(self.emission.output_dim * num_data, default_float())
        )

        term1 = -0.5 * tf.reduce_sum(
                input_tensor=tf.einsum("...op,...p,...o->...o", self._r_inv_data, disp_data, disp_data), axis=[-1, -2]
            )

        # term 2 is: ½|L⁻¹(GᵀΣ⁻¹)y|²
        # (GᵀΣ⁻¹)y [..., num_transitions + 1, state_dim]
        obs_proj = self._back_project_y_to_state(disp)

        # ½|L⁻¹(GᵀΣ⁻¹)y|² [...]
        term2 = 0.5 * tf.reduce_sum(
            input_tensor=tf.square(l_post.solve(obs_proj, transpose_left=False)), axis=[-1, -2]
        )

        ## term 3 is: ½log |K⁻¹| - log |L| + ½ log |Σ⁻¹|
        # where log |Σ⁻¹| = num_data * log|R⁻¹|
        term3 = (
                0.5 * self.prior_ssm.log_det_precision()
                - l_post.abs_log_det()
                + 0.5 * self._log_det_observation_precision
        )

        return tf.reduce_sum(cst + term1 + term2 + term3)
