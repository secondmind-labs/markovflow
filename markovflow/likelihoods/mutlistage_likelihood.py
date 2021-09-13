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

""" Module containing the MultiStageLikelihood """

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.likelihoods import Bernoulli, MultiLatentLikelihood, Poisson
from gpflow.likelihoods.utils import inv_probit


class MultiStageLikelihood(MultiLatentLikelihood):
    r"""
    The Multistage Likelihood as described in

    @inproceedings{seeger2016bayesian,
      title={Bayesian intermittent demand forecasting for large inventories},
      author={Seeger, Matthias W and Salinas, David and Flunkert, Valentin},
      booktitle={Advances in Neural Information Processing Systems},
      pages={4646--4654},
      year={2016}
    }

    This relates scalar data y to variables F = [F0, F1, F2]
    through the log-conditional density

    log p(Y|F) = δ(Y=0) * log σ(F0)
                + δ(Y=1) * (log(1 - σ(F0)) + log σ(F1))
                + δ(Y>1) * (log(1 - σ(F0)) + log(1-σ(F1)) + log Poisson(Y-2|λ(F2)))

    This can be interpreted as a decision tree
          σ(F0) -> Y = 0
         /
    root -> 1-σ(F0) -> 1-σ(F1) -> Y ~ Poisson(λ(F2)) + 2
                  \
                   σ(F1) -> Y = 1
    """

    def __init__(self, invlink_bernoulli=inv_probit, invlink_poisson=tf.exp, **kwargs):
        super().__init__(latent_dim=3, **kwargs)
        self.invlink_bernoulli = invlink_bernoulli
        self.invlink_poisson = invlink_poisson

    def _split_f(self, F):
        """
        Splits the input tensor F into 3 tensors along the last dimension
        :param F: tensor of shape [..., 3]
        :return: tuple of 3 tensors of shape [..., 1]
        """
        F0 = F[..., 0:1]
        F1 = F[..., 1:2]
        F2 = F[..., 2:3]
        return (F0, F1, F2)

    def _log_prob(self, F, Y):
        """
        Return the log-conditional density
        log p(Y|F) = δ(Y=0) * log σ(F0)
                + δ(Y=1) * (log(1 - σ(F0)) + log σ(F1))
                + δ(Y>1) * (log(1 - σ(F0)) + log(1-σ(F1)) + log Poisson(Y-2|λ(F2)))

        :param F: tensor of shape [..., 3]
        :param Y: tensor of shape [..., 1]
        :return:  tensor of shape [...]
        """
        F0, F1, F2 = self._split_f(F)

        # flags
        true = tf.ones_like(Y)
        false = tf.zeros_like(Y)

        bern = Bernoulli(invlink=self.invlink_bernoulli)
        poisson = Poisson(invlink=self.invlink_poisson)

        lp0 = bern.log_prob(F0, true)[:, None]  # log σ(F0)
        lpn0 = bern.log_prob(F0, false)[:, None]  # log(1 - σ(F0))
        lp1 = bern.log_prob(F1, true)[:, None]  # log σ(F1)
        lpn1 = bern.log_prob(F1, false)[:, None]  # log(1 - σ(F1))
        lp2 = poisson.log_prob(F2, Y - 2)[:, None]  # log Poisson(Y-2|λ(F2))

        zeros = tf.zeros_like(Y)
        logp = (
            tf.where(tf.equal(Y, 0), lp0, zeros)
            + tf.where(tf.equal(Y, 1), lpn0 + lp1, zeros)
            + tf.where(tf.greater_equal(Y, 2), lpn0 + lpn1 + lp2, zeros)
        )
        return tf.squeeze(logp, axis=-1)

    def _variational_expectations(self, Fmu, Fvar, Y):
        """
        Returns E_q(F) log p(Y|F) under the factored distribution
        q(F) = ∏ₖ q(Fₖ) = ∏ₖ 𝓝(Fmuₖ, Fvarₖ)

        E_q(F) log p(Y|F) = δ(Y=0) * E_q(F0) log σ(F0)
                + δ(Y=1) * (E_q(F0) log(1 - σ(F0)) + E_q(F1) log σ(F1))
                + δ(Y>1) * (E_q(F0) log(1 - σ(F0)) + E_q(F1) log(1-σ(F1))
                            + E_q(F2) log Poisson(Y-2|λ(F2)))

        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :returns: variational expectations, with shape [...]

        """
        Fmu0, Fmu1, Fmu2 = self._split_f(Fmu)
        Fvar0, Fvar1, Fvar2 = self._split_f(Fvar)

        # flags
        true = tf.ones_like(Y)
        false = tf.zeros_like(Y)

        bern = Bernoulli(invlink=self.invlink_bernoulli)
        poisson = Poisson(invlink=self.invlink_poisson)

        ve0 = bern.variational_expectations(Fmu0, Fvar0, true)[:, None]  # E_q(F0) log σ(F0)
        ven0 = bern.variational_expectations(Fmu0, Fvar0, false)[:, None]  # E_q(F0) log(1 - σ(F0))
        ve1 = bern.variational_expectations(Fmu1, Fvar1, true)[:, None]  # E_q(F1) log σ(F1)
        ven1 = bern.variational_expectations(Fmu1, Fvar1, false)[:, None]  # E_q(F1) log(1 - σ(F1))
        # E_q(F2) log Poisson(Y-2|λ(F2))
        ve2 = poisson.variational_expectations(Fmu2, Fvar2, Y - 2)[:, None]
        zeros = tf.zeros_like(Y)
        return tf.squeeze(
            (
                tf.where(tf.equal(Y, 0), ve0, zeros)
                + tf.where(tf.equal(Y, 1), ven0 + ve1, zeros)
                + tf.where(tf.greater_equal(Y, 2), ven0 + ven1 + ve2, zeros)
            ),
            axis=-1,
        )

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError

    def sample_y(self, F_samples: tf.Tensor):
        """
        Given values of the latent processes F,
        samples observations from P(Y|F)
        :param F_samples: batch_shape + [3]
        :return: batch_shape + [1]
        """

        bern = Bernoulli(invlink=self.invlink_bernoulli)
        poisson = Poisson(invlink=self.invlink_poisson)

        F0, F1, F2 = self._split_f(F_samples)

        # Calculate rates for the 3 levels
        bern_rate0 = bern.invlink(F0)
        bern_rate1 = bern.invlink(F1)
        poiss_rate = poisson.invlink(F2)

        # Sample bernoullis and poisson
        eta0 = tfp.distributions.Bernoulli(probs=bern_rate0).sample()
        eta1 = tfp.distributions.Bernoulli(probs=bern_rate1).sample()
        lmbda = tfp.distributions.Poisson(rate=poiss_rate).sample()

        # Mask
        zeros = tf.zeros_like(eta0, dtype=F_samples.dtype)
        ones = tf.ones_like(eta0, dtype=F_samples.dtype)

        output_ones = tf.logical_and(tf.equal(eta0, 0), tf.equal(eta1, 1))
        output_larger = tf.logical_and(tf.equal(eta0, 0), tf.equal(eta1, 0))
        return tf.where(output_ones, ones, zeros) + tf.where(output_larger, lmbda + 2, zeros)
