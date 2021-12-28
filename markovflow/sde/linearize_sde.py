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
"""Module to linearize a non-linear SDE."""

import tensorflow as tf
from gpflow.quadrature import mvnquad
from gpflow.base import TensorType

from markovflow.base import APPROX_INF
from markovflow.sde import SDE


class LinearizeSDE(SDE):
    """
    Linearizes a non-linear sde (with fixed diffusion) by using a piece-wise linear SDE.

    Non-linear SDE:
    .. math::
        & dx(t)/dt = f(x(t)) + L w(t), \\
        & L = \Sigma^\top \Sigma.

    Linear SDE:
    .. math::
        & dx(t)/dt = (A_i x(t) + b_i) + L w(t), \\
        & L = \Sigma^\top \Sigma ,

    where $i$ are linearization points over a time interval (0, T).

    For computing the optimal parameters $A^{*}_{i}$, $b^{*}_{i}$, Girsanov theorem is used to compare the two drifts.

    Diffusion of the linear SDE is same as the non-linear SDE.
    """
    def __init__(self, sde: SDE, linearize_points: TensorType, dim: int = 1):
        """
        :param sde: non-linear sde to be linearized.
        :param linearize_points: points used for the piecewise linearization ``[N,]``.
        """
        super(LinearizeSDE, self).__init__()
        self.sde = sde
        self.dim = dim
        self.linearize_points = tf.sort(linearize_points)
        self.n = self.linearize_points.shape[0] + 1
        self.A = tf.random.normal((self.n, self.dim, self.dim))
        self.b = tf.random.normal((self.n, self.dim))
        self.quadrature = mvnquad
        self.quadrature_pnts = 10

    def _get_linearization_idx_for_grid(self, t):
        """
        Get the indexes of the linearization parameters i.e. A_i and b_i for the given grid t.
        """
        inf = APPROX_INF * tf.ones_like(self.linearize_points[..., -1:])
        linearize_points_augmented = tf.concat([-inf, self.linearize_points, inf], axis=-1)

        return tf.searchsorted(linearize_points_augmented, t, "right") - 1

    def drift(self, x: TensorType, t: TensorType):
        """Drift of the linearize SDE at x(t)."""
        i = self._get_linearization_idx_for_grid(t)
        A = tf.gather(self.A, i)
        b = tf.gather(self.b, i)
        x = tf.expand_dims(x, -1)
        return tf.squeeze(tf.matmul(A, x), -1) + b

    def diffusion(self, x, t):
        """Diffusion of the linearize SDE"""
        return self.sde.diffusion(x, t)

    def _E_x2(self, q_mean: TensorType, q_covar: TensorType) -> TensorType:
        """E_q(x(t))[x(t)^2]"""
        m = tf.expand_dims(q_mean, axis=-1)
        mmT = m @ tf.transpose(m, [0, 2, 1])
        return mmT + q_covar

    def _E_f(self, q_mean: TensorType, q_covar: TensorType) -> TensorType:
        """E_q(x(t))[f(x(t))]"""
        fx = lambda x: self.sde.drift(x=x, t=0)
        val = self.quadrature(fx, q_mean, q_covar, H=self.quadrature_pnts, Din=self.dim)
        return val

    def _E_f_x(self, q_mean: TensorType, q_covar: TensorType) -> TensorType:
        """E_q(x(t))[x(t) f(x(t))]"""
        def xfx(x):
            x = tf.expand_dims(x, axis=-1)
            return self.sde.drift(x=x, t=0) @ tf.transpose(x, [0, 2, 1])

        val = self.quadrature(xfx, q_mean, q_covar, H=self.quadrature_pnts, Din=self.dim)
        return val

    def _dfx(self, x, t=0):
        """df(x(t)) / dx(t)"""
        with tf.GradientTape() as tape:
            tape.watch(x)
            drift_val = self.sde.drift(x, t)
            dfx = tape.gradient(drift_val, x)
        return dfx

    def _E_df(self, q_mean: TensorType, q_covar: TensorType) -> TensorType:
        """E_q(.)[f'(x(t))]"""
        val = self.quadrature(self._dfx, q_mean, q_covar, H=self.quadrature_pnts)
        return val

    def update_linearization_parameters(self, q_mean: TensorType, q_covar: TensorType, grid: TensorType = None):
        """
        Update the linearization parameters i.e. $A_i$ and $b_i$ on the basis of the current optimal approximate
        Gaussian posterior $q(\cdot) \sim N(q_{mean}, q_{covar})$ specified for every linearization point.

        if grid is None or grid is same as linearization points then non-sparse update is done otherwise sparse.
        """
        if (grid is None) or (q_mean.shape[0] == self.n):
            return self._update_linearization_parameters_non_sparse(q_mean, q_covar)

        else:
            return self._update_linearization_parameters_sparse(q_mean, q_covar, grid)

    def _update_linearization_parameters_non_sparse(self, q_mean: TensorType, q_covar: TensorType):
        """
        Update the linearization parameters i.e. $A_i$ and $b_i$ on the basis of the current optimal approximate
        Gaussian posterior $q(\cdot) \sim N(q_{mean}, q_{covar})$ specified for every linearization point.

        & A_{i}^{*} = E_{q(.)}[d f(x)/ dx]

        & b_{i}^{*} = E_{q(.)}[f(x)] - A_{i}^{*}  E_{q(.)}[x]

        :param q_mean: mean of Gaussian posterior, (N, D)
        :param q_covar: covariance of Gaussian posterior, (N, D, D)

        """

        E_f = self._E_f(q_mean, q_covar)
        E_x = q_mean

        self.A = self._E_df(q_mean, q_covar)
        self.b = E_f - self.A * E_x
        self.A = tf.expand_dims(self.A, axis=-1)

    def _update_linearization_parameters_sparse(self, q_mean: TensorType, q_covar: TensorType, grid: TensorType):
        """
        Update the linearization parameters i.e. $A_i$ and $b_i$ on the basis of the current optimal approximate
        Gaussian posterior $q(\cdot) \sim N(q_{mean}, q_{covar})$ specified over the grid.

        ..math::

        & A_{i}^{*} = (\int E_{q(.)}[f(x_t) x_t] dt - \int E_{q(.)}[f(x_t)] dt \int E_{q(.)}[x_t] dt) *
                      inverse{(\int E_{q(.)}[x_t^2] dt) - \int E_{q(.)}[x_t] dt) \int E_{q(.)}[x_t] dt))}

        & b_{i}^{*} = \int E_{q(.)}[f(x_t)] dt - A_{i}^{*}  \int E_{q(.)}[x_t] dt

        :param q_mean: mean of Gaussian posterior, (N, D)
        :param q_covar: covariance of Gaussian posterior, (N, D, D)
        :param grid: homogeneous grid points, (N-1,)
        """

        dt = float(grid[1] - grid[0])

        E_f_x = self._E_f_x(q_mean, q_covar)
        E_f = self._E_f(q_mean, q_covar)
        E_x = q_mean
        E_x2 = self._E_x2(q_mean, q_covar)

        idx = self._get_linearization_idx_for_grid(grid)

        A = None
        b = None
        for i in range(self.n):
            index = tf.reshape(tf.where(idx == i), (-1))
            E_f_x_i = tf.reduce_sum(tf.gather(E_f_x, index), axis=0) * dt
            E_f_i = tf.expand_dims(tf.reduce_sum(tf.gather(E_f, index), axis=0), axis=0) * dt
            E_x_i = tf.expand_dims(tf.reduce_sum(tf.gather(E_x, index), axis=0), axis=0) * dt
            E_x2_i = tf.reduce_sum(tf.gather(E_x2, index), axis=0) * dt

            val = (E_x2_i - tf.transpose(E_x_i) * E_x_i)
            val = tf.math.abs(val)
            inv_val = tf.linalg.cholesky_solve(tf.linalg.cholesky(val), tf.eye(self.dim, dtype=val.dtype))

            A_i = (E_f_x_i - tf.transpose(E_f_i) @ E_x_i) * inv_val

            b_i = E_f_i - E_x_i @ A_i

            A_i = tf.reshape(A_i, (1, self.dim, self.dim))
            b_i = tf.reshape(b_i, (1, self.dim))

            A = A_i if A is None else tf.concat([A, A_i], axis=0)
            b = b_i if b is None else tf.concat([b, b_i], axis=0)

        self.A = A
        self.b = b
