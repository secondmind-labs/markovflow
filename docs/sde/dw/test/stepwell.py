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

from docs.sde.sde_exp_utils import euler_maruyama
from markovflow.sde.sde import StepWellSDE
from markovflow.models.vi_sde import VariationalMarkovGP

from docs.sde.t_vgp_trainer import tVGPTrainer


DTYPE = default_float()
wandb.init()

tf.random.set_seed(720)
np.random.seed(720)


def setup():
    t0 = 0
    t1 = 20
    dt = 0.001
    noise_var = 0.001
    n_observations = 30

    # Define q
    q = 1.5 * tf.ones((1, 1), dtype=DTYPE)

    # Generate observations.
    sde = StepWellSDE(q=q)

    x0_shape = (1, 1)
    x0 = 1. + tf.zeros(x0_shape, dtype=DTYPE)

    time_grid = tf.cast(np.linspace(t0, t1, int((t1 - t0) // dt) + 2), dtype=DTYPE)

    # Observation at every even place
    observation_idx = list(tf.cast(np.linspace(2, time_grid.shape[0] - 2, n_observations), dtype=tf.int32))
    obs_t = tf.gather(time_grid, observation_idx)

    latent_process = euler_maruyama(sde, x0, time_grid)
    latent_states = tf.gather(latent_process, observation_idx, axis=1)
    # Adding observation noise
    obs_val = latent_states + tf.random.normal(latent_states.shape, stddev=np.sqrt(noise_var), dtype=DTYPE)

    obs_val = tf.reshape(obs_val, (-1, 1))

    observations = (obs_t, obs_val)
    likelihood = Gaussian(variance=noise_var)

    plt.subplots(1, 1, figsize=(15, 5))
    plt.scatter(observations[0].numpy().reshape(-1), observations[1].numpy().reshape(-1))
    plt.show()

    return sde, observations, time_grid, likelihood, dt


if __name__ == '__main__':

    sde_p, observations, t, likelihood, dt = setup()

    # All sites
    t_vgp_trainer_all_sites = tVGPTrainer(observations, likelihood, t, sde_p, data_sites_lr=0.9, all_sites_lr=0.1,
                                          update_all_sites=True)
    t_vgp_elbo_vals_all_sites, _, _ = t_vgp_trainer_all_sites.run(update_prior=False)
    print(f"t-VGP model (All-sites) : {t_vgp_elbo_vals_all_sites[-1]}")

    t_vgp_model_all_sites = t_vgp_trainer_all_sites.tvgp_model

    data_sites_nat1 = t_vgp_model_all_sites.data_sites.nat1
    data_sites_nat2 = t_vgp_model_all_sites.data_sites.nat2
    noise_var = likelihood.variance
    np.testing.assert_array_almost_equal(data_sites_nat1.numpy(), observations[1] / noise_var)
    np.testing.assert_array_almost_equal(tf.reduce_sum(data_sites_nat2),
                                         data_sites_nat2.shape[0] * (-1 / (2 * noise_var)))

    vgp_model = VariationalMarkovGP(observations, sde_p, t, likelihood, lr=0.5, initial_state_lr=0.05,
                                    convergence_tol=1e-4)

    # Initialize VGP model
    vgp_model.q_initial_cov = 1. + 0. * vgp_model.q_initial_cov
    vgp_model.q_initial_mean = observations[1][0] + 0. * vgp_model.q_initial_mean
    vgp_model.p_initial_mean = observations[1][0] + 0. * vgp_model.p_initial_mean
    vgp_model.p_initial_cov = 0.5 + 0. * vgp_model.p_initial_cov

    vgp_elbo_vals, _, _ = vgp_model.run(update_prior=False, update_initial_statistics=True)

    print(f"VGP model : {vgp_elbo_vals[-1]}")

    m_vgp, S_vgp = vgp_model.forward_pass
    m_vgp = m_vgp.numpy().reshape(-1)
    S_std_vgp = np.sqrt(S_vgp.numpy()).reshape(-1)

    m_all_sites, S_all_sites = t_vgp_model_all_sites.dist_q.marginals
    m_all_sites = m_all_sites.numpy().reshape(-1)
    S_all_sites_std = np.sqrt(S_all_sites.numpy()).reshape(-1)

    # Plotting
    plt.subplots(1, 1, figsize=(15, 5))
    plt.scatter(observations[0].numpy().reshape(-1), observations[1].numpy().reshape(-1))
    plt.fill_between(
        t,
        y1=(m_all_sites.reshape(-1) - 2 * S_all_sites_std.reshape(-1)).reshape(-1, ),
        y2=(m_all_sites.reshape(-1) + 2 * S_all_sites_std.reshape(-1)).reshape(-1, ),
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

    plt.plot(t.numpy().reshape(-1), m_vgp.reshape(-1), color="blue", alpha=0.5, label="VGP")
    plt.plot(t.numpy().reshape(-1), m_all_sites.reshape(-1), color="black", linestyle="dashed", label="t-VGP (All sites)")

    plt.legend()
    plt.savefig("test_stepwell_posterior.png")
    plt.show()

    # Linearized prior
    lin_m, lin_S = t_vgp_model_all_sites.dist_p_ssm.marginals
    lin_m = lin_m.numpy().reshape(-1)
    lin_S_std = np.sqrt(lin_S.numpy()).reshape(-1)

    # Plotting
    plt.subplots(1, 1, figsize=(15, 5))
    plt.scatter(observations[0].numpy().reshape(-1), observations[1].numpy().reshape(-1))
    plt.fill_between(
        t,
        y1=(lin_m.reshape(-1) - 2 * lin_S_std.reshape(-1)).reshape(-1, ),
        y2=(lin_m.reshape(-1) + 2 * lin_S_std.reshape(-1)).reshape(-1, ),
        edgecolor="black",
        facecolor=(0, 0, 0, 0.),
        linestyle='dashed'
    )
    plt.hlines(2, 0, 10, linestyles="dashed")

    plt.plot(t.numpy().reshape(-1), lin_m.reshape(-1), color="black", linestyle="dashed")

    plt.legend()
    plt.savefig("test_stepwell_lin_prior.png")
    plt.show()

    b = t_vgp_model_all_sites.dist_p_ssm.state_offsets / dt
    A = (t_vgp_model_all_sites.dist_p_ssm.state_transitions - tf.eye(1, dtype=b.dtype)) / dt

    A = A.numpy().reshape(-1)
    b = b.numpy().reshape(-1)

    c = -b/A
    plt.plot(c)
    plt.savefig("c.png")
    plt.show()
