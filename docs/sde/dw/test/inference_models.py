"""
Test the expansion of KL
   KL[q_{L} || p] = KL[q_{L} || p_{L}] + u(q_{L}, p, p_{L})

using t-VGP and VGP models.
"""
import wandb
import tensorflow as tf
import numpy as np
from gpflow.config import default_float
from gpflow.likelihoods import Gaussian
import matplotlib.pyplot as plt

from docs.sde.sde_exp_utils import generate_dw_data
from markovflow.sde.sde import PriorDoubleWellSDE
from markovflow.models.vi_sde import VariationalMarkovGP

from docs.sde.t_vgp_trainer import tVGPTrainer


DTYPE = default_float()
wandb.init()

tf.random.set_seed(33)
np.random.seed(33)


def setup():
    t0 = 0
    t1 = 10
    dt = 0.001
    noise_var = 0.01

    # Define q
    q = tf.ones((1, 1), dtype=DTYPE)

    # Generate observations.
    # Observations and likelihoods are not really required but need to pass them
    obs_val, obs_t, _, t, _, _, _ = generate_dw_data(q=q, x0=1., t0=t0, t1=t1, simulation_dt=dt,
                                                     noise_stddev=np.sqrt(noise_var), n_observations=20, dtype=DTYPE)
    obs_val = tf.reshape(obs_val, (-1, 1))
    observations = (obs_t, obs_val)
    likelihood = Gaussian(variance=noise_var)

    # DW non-linear SDE, p
    sde_p = PriorDoubleWellSDE(q=q, initial_a_val=3.0, initial_c_val=1.0)

    return sde_p, observations, t, likelihood, dt


if __name__ == '__main__':

    sde_p, observations, t, likelihood, dt = setup()

    t_vgp_trainer = tVGPTrainer(observations, likelihood, t, sde_p, data_sites_lr=0.9, all_sites_lr=0.1,
                                update_all_sites=True)
    t_vgp_elbo_vals, _, _ = t_vgp_trainer.run(update_prior=False)
    print(f"t-VGP model : {t_vgp_elbo_vals[-1]}")

    t_vgp_model = t_vgp_trainer.tvgp_model

    data_sites_nat1 = t_vgp_model.data_sites.nat1
    data_sites_nat2 = t_vgp_model.data_sites.nat2
    noise_var = likelihood.variance
    np.testing.assert_array_almost_equal(data_sites_nat1.numpy(), observations[1] / noise_var)
    np.testing.assert_array_almost_equal(tf.reduce_sum(data_sites_nat2),
                                         data_sites_nat2.shape[0] * (-1 / (2 * noise_var)))

    # Compare Q of the posterior SSM, should be equal/ close to true q.
    dist_q_Q = tf.square(t_vgp_model.dist_q.cholesky_process_covariances) / dt
    print(dist_q_Q)

    vgp_model = VariationalMarkovGP(observations, sde_p, t, likelihood, lr=0.5, initial_state_lr=0.05,
                                    convergence_tol=1e-4)

    # Initialize VGP model
    vgp_model.q_initial_cov = 1. + 0. * vgp_model.q_initial_cov
    vgp_model.q_initial_mean = observations[1][0] + 0. * vgp_model.q_initial_mean
    vgp_model.p_initial_mean = observations[1][0] + 0. * vgp_model.p_initial_mean
    vgp_model.p_initial_cov = 0.5 + 0. * vgp_model.p_initial_cov

    vgp_elbo_vals, _, _ = vgp_model.run(update_prior=False, update_initial_statistics=True)

    print(f"VGP model : {vgp_elbo_vals[-1]}")

    m, S = t_vgp_model.dist_q.marginals
    m = m.numpy().reshape(-1)
    S_std = np.sqrt(S.numpy()).reshape(-1)

    m_vgp, S_vgp = vgp_model.forward_pass
    m_vgp = m_vgp.numpy().reshape(-1)
    S_std_vgp = np.sqrt(S_vgp.numpy()).reshape(-1)

    plt.subplots(1, 1, figsize=(15, 5))
    plt.scatter(observations[0].numpy().reshape(-1), observations[1].numpy().reshape(-1))
    plt.fill_between(
        t,
        y1=(m.reshape(-1) - 2 * S_std.reshape(-1)).reshape(-1, ),
        y2=(m.reshape(-1) + 2 * S_std.reshape(-1)).reshape(-1, ),
        edgecolor="black",
        facecolor=(0, 0, 0, 0.),
        linestyle='dashed'
    )
    plt.fill_between(
        t,
        y1=(m_vgp.reshape(-1) - 2 * S_std_vgp.reshape(-1)).reshape(-1, ),
        y2=(m_vgp.reshape(-1) + 2 * S_std_vgp.reshape(-1)).reshape(-1, ),
        edgecolor="blue",
        facecolor=(0, 0, 0, 0.)
    )

    plt.plot(t.numpy().reshape(-1), m.reshape(-1), color="black", linestyle="dashed", label="t-VGP")
    plt.plot(t.numpy().reshape(-1), m_vgp.reshape(-1), color="blue", alpha=0.5, label="VGP")

    plt.legend()
    plt.savefig("test_posterior_inference.png")
    plt.show()
