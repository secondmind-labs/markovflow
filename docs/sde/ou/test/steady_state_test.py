import tensorflow as tf
from gpflow.config import default_float

from markovflow.sde.sde import OrnsteinUhlenbeckSDE
from docs.sde.sde_exp_utils import get_steady_state
from gpflow.likelihoods import Gaussian

from markovflow.models.cvi_sde import SDESSM

DTYPE = default_float()

if __name__ == '__main__':
    decay = 0.5
    Q = 1
    noise_var = 0.1
    dt = 0.001
    t0 = 0.
    t1 = 1.

    time_grid = tf.cast(tf.linspace(t0, t1, int((t1 - t0)/dt)), dtype=DTYPE)

    sde = OrnsteinUhlenbeckSDE(decay * tf.ones((1, 1), dtype=DTYPE), Q * tf.ones((1, 1), dtype=DTYPE))

    likelihood_ssm = Gaussian(noise_var)
    ssm_model = SDESSM(input_data=(tf.ones((10, ), dtype=DTYPE), tf.ones((10, 1), dtype=DTYPE)),
                       prior_sde=sde, grid=time_grid, likelihood=likelihood_ssm)
    # ssm_model.initial_chol_cov = tf.linalg.cholesky(tf.reshape(initial_cov, ssm_model.initial_chol_cov.shape))
    # ssm_model.fx_covs = ssm_model.initial_chol_cov.numpy().item() ** 2 + 0 * ssm_model.fx_covs
    ssm_model._linearize_prior()

    A = tf.squeeze(ssm_model.dist_p_ssm.state_transitions, axis=0)
    b = tf.squeeze(ssm_model.dist_p_ssm.state_offsets, axis=0)
    A = (A - tf.eye(1, dtype=A.dtype)) / dt
    b = b / dt
    q = ssm_model.dist_p_ssm.cholesky_process_covariances

    steady_m, steady_P = get_steady_state(A, b, tf.square(q)/dt)

    print(steady_m[0][0])
    print(steady_P[0][0])

    print(Q / (2 * decay))
