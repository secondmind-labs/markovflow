"""Euler-Maruyama vs SSM marginals"""

import numpy as np
from markovflow.state_space_model import StateSpaceModel

# Suppose we have a SDE with drift f(x) = A_t x_t + b_t

time_grid = np.arange(0, 10, 0.01).reshape((-1, 1))
A = np.random.random(time_grid.shape[0]-1).reshape((-1, 1, 1))
b = np.random.random(time_grid.shape[0]-1).reshape((-1, 1))

q = 1.5
m = np.zeros(time_grid.shape[0]).reshape((-1, 1))
S = np.zeros(time_grid.shape[0]).reshape((-1, 1, 1))
m[0] = 1.
S[0] = 1.5

dt = 0.01
N = time_grid.shape[0]

"""Euler-Maruyama of the posterior moments"""
for i in range(N - 1):
    dmdt = A[i] * m[i] + b[i]
    dSdt = A[i] * S[i] + S[i] * A[i] + q  # + A[i] * A[i] * dt
    m[i + 1] = m[i] + dt * dmdt
    S[i + 1] = S[i] + dt * dSdt

"""SSM"""
state_transition = np.eye(1, dtype=A.dtype) + A * dt
state_offset = b * dt

q = np.repeat(np.array(q * dt).reshape((1, 1, 1)), state_transition.shape[0], axis=0)

ssm = StateSpaceModel(initial_mean=m[0],
                      chol_initial_covariance=np.linalg.cholesky(S[0]),
                      state_transitions=state_transition,
                      state_offsets=state_offset,
                      chol_process_covariances=np.linalg.cholesky(q)
                      )
ssm_m, ssm_S = ssm.marginals

np.testing.assert_array_almost_equal(m, ssm_m)
np.testing.assert_array_almost_equal(S, ssm_S, decimal=2)
