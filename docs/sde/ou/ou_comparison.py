"""OU SDE CVI vs SDE VI"""
import os
import argparse

import gpflow
import matplotlib.pyplot as plt
import numpy as np

# Don't use GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# Restrict TensorFlow to only use the first GPU
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     tf.config.set_visible_devices(gpus[1], 'GPU')

from gpflow import default_float
from gpflow.likelihoods import Gaussian
import wandb

from markovflow.sde.sde import OrnsteinUhlenbeckSDE, PriorOUSDE
from markovflow.kernels import OrnsteinUhlenbeck
from markovflow.models.cvi_sde import SDESSM
from markovflow.models.vi_sde import VariationalMarkovGP
from markovflow.sde.sde_utils import gaussian_log_predictive_density

from docs.sde.sde_exp_utils import predict_vgp, predict_ssm, plot_observations, plot_posterior, get_cvi_gpr_taylor, \
    predict_cvi_gpr_taylor, get_cvi_gpr, predict_cvi_gpr

DTYPE = default_float()
plt.rcParams["figure.figsize"] = [15, 5]

DATA_PATH = ""
DECAY = 1.
Q = 1.
NOISE_STDDEV = 0.1 * np.ones((1, 1))
X0 = 0.
T0 = 0.
T1 = 1.
DT = 0.
TIME_GRID = tf.zeros((1, 1))
OBSERVATION_DATA = ()
TEST_DATA = ()
LATENT_PROCESS = tf.zeros((1, 1))
OUTPUT_DIR = ""
LEARN_PRIOR_SDE = False
INITIAL_PRIOR_VALUE = 1.


def load_data(data_dir):
    """
    Get Data
    """
    global DATA_PATH, DECAY, Q, X0, NOISE_STDDEV, T0, T1, TIME_GRID, OBSERVATION_DATA, LATENT_PROCESS, TEST_DATA, DT
    DATA_PATH = data_dir
    data = np.load(os.path.join(data_dir, "data.npz"))
    DECAY = data["decay"]
    Q = data["q"]
    NOISE_STDDEV = data["noise_stddev"]
    X0 = data["x0"]
    OBSERVATION_DATA = (data["observation_grid"], tf.squeeze(data["observation_vals"], axis=0).numpy())
    LATENT_PROCESS = data["latent_process"]
    TIME_GRID = data["time_grid"]
    T0 = TIME_GRID[0]
    T1 = TIME_GRID[-1]
    TEST_DATA = (data["test_grid"], data["test_vals"])
    DT = TIME_GRID[1] - TIME_GRID[0]


def modify_time_grid(dt: float):
    """Modifying time grid."""
    global DT, TIME_GRID
    DT = dt
    TIME_GRID = tf.cast(np.arange(T0, T1 + DT, DT), dtype=DTYPE).numpy()


def plot_data():
    """Plot the data"""
    plt.clf()
    plot_observations(OBSERVATION_DATA[0], OBSERVATION_DATA[1])
    plt.plot(TIME_GRID, tf.reshape(LATENT_PROCESS, (-1)), label="Latent Process", alpha=0.2, color="gray")
    plt.plot(TEST_DATA[0].reshape(-1), TEST_DATA[1].reshape(-1), 'x', color="red", ms=8, mew=2,
             label="Test Observations (Y)")
    plt.xlabel("Time (t)")
    plt.ylabel("y(t)")
    plt.xlim([T0, T1])
    plt.title("Observations")
    plt.legend()
    plt.show()


def set_output_dir():
    """Create output directory"""
    global OUTPUT_DIR
    if LEARN_PRIOR_SDE:
        OUTPUT_DIR = os.path.join(DATA_PATH, "learning")
    else:
        OUTPUT_DIR = os.path.join(DATA_PATH, "inference")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def init_wandb(uname: str, log: bool = False, sites_lr: float = 0.5, ssm_prior_lr: float = 0.01,
               vgp_lr: float = 0.01, vgp_prior_lr: float = 0.01):
    """Initialize Wandb"""

    if not log:
        os.environ['WANDB_MODE'] = 'offline'

    config = {
        "seed": DATA_PATH.split("/")[-1],
        "learning": LEARN_PRIOR_SDE,
        "t0": T0,
        "t1": T1,
        "grid_dt": DT,
        "decay": DECAY,
        "q": Q,
        "noise_stddev": NOISE_STDDEV,
        "n_observations": OBSERVATION_DATA[0].shape[0],
        "sites_lr": sites_lr,
        "SSM_prior_lr": ssm_prior_lr,
        "vgp_lr": vgp_lr,
        "vgp_prior_lr": vgp_prior_lr
    }

    """Logging init"""
    wandb.init(project="VI-SDE", entity=uname, config=config)


def gpr_taylor():
    likelihood_gpr = Gaussian(NOISE_STDDEV**2)

    if LEARN_PRIOR_SDE:
        kernel = OrnsteinUhlenbeck(decay=INITIAL_PRIOR_VALUE, diffusion=Q)
        gpflow.set_trainable(kernel.diffusion, False)
    else:
        kernel = OrnsteinUhlenbeck(decay=DECAY, diffusion=Q)

    cvi_gpr_taylor_model, cvi_taylor_params, cvi_taylor_elbo_vals = get_cvi_gpr_taylor(OBSERVATION_DATA, kernel,
                                                                                       TIME_GRID, likelihood_gpr,
                                                                                       train=LEARN_PRIOR_SDE,
                                                                                       sites_lr=1.)
    print(f"CVI-GPR (Taylor) ELBO: {cvi_gpr_taylor_model.classic_elbo()}")

    return cvi_gpr_taylor_model


def cvi_gpr():
    likelihood_gpr = Gaussian(NOISE_STDDEV ** 2)

    if LEARN_PRIOR_SDE:
        kernel = OrnsteinUhlenbeck(decay=INITIAL_PRIOR_VALUE, diffusion=Q)
        gpflow.set_trainable(kernel.diffusion, False)
    else:
        kernel = OrnsteinUhlenbeck(decay=DECAY, diffusion=Q)

    cvi_gpr_model, cvi_params = get_cvi_gpr(OBSERVATION_DATA, kernel, likelihood_gpr, train=LEARN_PRIOR_SDE,
                                            sites_lr=0.9)
    return cvi_gpr_model, cvi_params


def perform_sde_ssm(sites_lr: float = 0.5, prior_lr: float = 0.01):
    if LEARN_PRIOR_SDE:
        prior_decay = INITIAL_PRIOR_VALUE
        true_q = Q * tf.ones((1, 1), dtype=DTYPE)
        # As prior OU SDE doesn't have a negative sign inside it.
        prior_sde_ssm = PriorOUSDE(initial_val=-1*prior_decay, q=true_q)
        # Steady covariance
        initial_cov = Q / (2 * -1 * prior_sde_ssm.decay)
    else:
        true_decay = DECAY * tf.ones((1, 1), dtype=DTYPE)
        true_q = Q * tf.ones((1, 1), dtype=DTYPE)
        prior_sde_ssm = OrnsteinUhlenbeckSDE(decay=true_decay, q=true_q)
        # Steady covariance
        initial_cov = Q / (2 * prior_sde_ssm.decay)

    # likelihood
    likelihood_ssm = Gaussian(NOISE_STDDEV**2)

    # model
    ssm_model = SDESSM(input_data=OBSERVATION_DATA, prior_sde=prior_sde_ssm, grid=TIME_GRID, likelihood=likelihood_ssm,
                       learning_rate=sites_lr, prior_params_lr=prior_lr, test_data=TEST_DATA)

    ssm_model.initial_mean = tf.zeros_like(ssm_model.initial_mean)
    ssm_model.initial_chol_cov = tf.linalg.cholesky(tf.reshape(initial_cov, ssm_model.initial_chol_cov.shape))
    ssm_model.fx_covs = ssm_model.initial_chol_cov.numpy().item()**2 + 0 * ssm_model.fx_covs
    ssm_model._linearize_prior()

    ssm_elbo, ssm_prior_prior_vals = ssm_model.run(update_prior=LEARN_PRIOR_SDE)

    return ssm_model, ssm_elbo, ssm_prior_prior_vals


def perform_vgp(vgp_lr: float = 0.01, prior_lr: float = 0.01):
    # Prior SDE
    if LEARN_PRIOR_SDE:
        vgp_prior_decay = INITIAL_PRIOR_VALUE
        true_q = Q * tf.ones((1, 1), dtype=DTYPE)
        # As prior OU SDE doesn't have a negative sign inside it.
        prior_sde_vgp = PriorOUSDE(initial_val=-1*vgp_prior_decay, q=true_q)
        # Steady covariance
        initial_cov = Q / (2 * -1 * prior_sde_vgp.decay)
    else:
        vgp_prior_decay = DECAY * tf.ones((1, 1), dtype=DTYPE)
        true_q = Q * tf.ones((1, 1), dtype=DTYPE)
        prior_sde_vgp = OrnsteinUhlenbeckSDE(decay=vgp_prior_decay, q=true_q)
        # Steady covariance
        initial_cov = Q / (2 * prior_sde_vgp.decay)

    # likelihood
    likelihood_vgp = Gaussian(NOISE_STDDEV**2)

    vgp_model = VariationalMarkovGP(input_data=OBSERVATION_DATA,
                                    prior_sde=prior_sde_vgp, grid=TIME_GRID, likelihood=likelihood_vgp,
                                    lr=vgp_lr, prior_params_lr=prior_lr, test_data=TEST_DATA)

    vgp_model.p_initial_cov = tf.reshape(initial_cov, vgp_model.p_initial_cov.shape)
    vgp_model.q_initial_cov = tf.identity(vgp_model.p_initial_cov)
    vgp_model.p_initial_mean = tf.zeros_like(vgp_model.p_initial_mean)
    vgp_model.q_initial_mean = tf.identity(vgp_model.p_initial_mean)

    vgp_model.A = vgp_prior_decay + 0. * vgp_model.A

    v_gp_elbo, v_gp_prior_vals = vgp_model.run(update_prior=LEARN_PRIOR_SDE)

    return vgp_model, v_gp_elbo, v_gp_prior_vals


def compare_plot_posterior(cvi_gpr_model, ssm_model, vgp_model):
    plt.clf()
    plot_observations(OBSERVATION_DATA[0], OBSERVATION_DATA[1])
    plt.plot(TEST_DATA[0].reshape(-1), TEST_DATA[1].reshape(-1), 'x', color="red", ms=8, mew=2,
             label="Test Observations (Y)")

    if LEARN_PRIOR_SDE:
        m_gpr, s_std_gpr = predict_cvi_gpr(cvi_gpr_model, TIME_GRID, NOISE_STDDEV)
    else:
        m_gpr, s_std_gpr = predict_cvi_gpr_taylor(cvi_gpr_model, NOISE_STDDEV)

    m_ssm, s_std_ssm = predict_ssm(ssm_model, NOISE_STDDEV)
    m_vgp, s_std_vgp = predict_vgp(vgp_model, NOISE_STDDEV)
    """
    Compare Posterior
    """
    # plt.vlines(time_grid.reshape(-1), -2, 2, alpha=0.2, color="black")
    plot_posterior(m_gpr, s_std_gpr, TIME_GRID, "GPR")
    plot_posterior(m_ssm, s_std_ssm, TIME_GRID, "SDE-SSM")
    plot_posterior(m_vgp, s_std_vgp, TIME_GRID, "VGP")
    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "posterior.svg"))
    wandb.log({"posterior": wandb.Image(plt)})

    plt.show()


def plot_elbo(ssm_elbo, v_gp_elbo):
    """ELBO comparison"""
    plt.clf()
    plt.plot(ssm_elbo, label="SDE-SSM")
    plt.plot(v_gp_elbo, label="VGP")
    plt.title("ELBO")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "elbo.svg"))

    wandb.log({"ELBO-Comparison": wandb.Image(plt)})

    plt.show()


def calculate_nlpd(ssm_model: SDESSM, vgp_model: VariationalMarkovGP):
    """Calculate NLPD on the test set for VGP and SDE-SSM"""

    m_ssm, s_std_ssm = predict_ssm(ssm_model, NOISE_STDDEV)
    m_vgp, s_std_vgp = predict_vgp(vgp_model, NOISE_STDDEV)

    """Calculate NLPD"""
    pred_idx = list((tf.where(TIME_GRID == TEST_DATA[0][..., None])[:, 1]).numpy())
    ssm_chol_covar = tf.reshape(tf.gather(s_std_ssm, pred_idx, axis=1) + NOISE_STDDEV, (-1, 1, 1))
    ssm_lpd = gaussian_log_predictive_density(mean=tf.gather(m_ssm, pred_idx, axis=0), chol_covariance=ssm_chol_covar,
                                              x=tf.reshape(TEST_DATA[1], (-1,)))
    ssm_nlpd = -1 * tf.reduce_mean(ssm_lpd).numpy().item()
    print(f"SDE-SSM NLPD: {ssm_nlpd}")

    vgp_chol_covar = tf.reshape(tf.gather(s_std_vgp, pred_idx) + NOISE_STDDEV, (-1, 1, 1))
    vgp_lpd = gaussian_log_predictive_density(mean=tf.gather(m_vgp, pred_idx, axis=0), chol_covariance=vgp_chol_covar,
                                              x=tf.reshape(TEST_DATA[1], (-1,)))
    vgp_nlpd = -1 * tf.reduce_mean(vgp_lpd).numpy().item()
    print(f"VGP NLPD: {vgp_nlpd}")


def save_data(ssm_elbo, vgp_elbo):
    """Save data into npz"""
    m_ssm, s_std_ssm = predict_ssm(ssm_model, NOISE_STDDEV)
    m_vgp, s_std_vgp = predict_vgp(vgp_model, NOISE_STDDEV)

    """Save SDE-SSM data"""
    np.savez(os.path.join(OUTPUT_DIR, "ssm_data_sites.npz"), nat1=ssm_model.data_sites.nat1.numpy(),
             nat2=ssm_model.data_sites.nat2.numpy(), log_norm=ssm_model.data_sites.log_norm.numpy())

    np.savez(os.path.join(OUTPUT_DIR, "ssm_inference.npz"), m=m_ssm, S=tf.square(s_std_ssm))
    np.savez(os.path.join(OUTPUT_DIR, "ssm_elbo.npz"), elbo=ssm_elbo)

    if LEARN_PRIOR_SDE:
        np.savez(os.path.join(OUTPUT_DIR, "ssm_learnt_sde.npz"), decay=ssm_prior_decay_values)

    "Save VGP data"
    np.savez(os.path.join(OUTPUT_DIR, "vgp_A_b.npz"), A=vgp_model.A.numpy(), b=vgp_model.b.numpy())
    np.savez(os.path.join(OUTPUT_DIR, "vgp_lagrange.npz"), psi_lagrange=vgp_model.psi_lagrange.numpy(),
             lambda_lagrange=vgp_model.lambda_lagrange.numpy())

    np.savez(os.path.join(OUTPUT_DIR, "vgp_inference.npz"), m=m_vgp, S=tf.square(s_std_vgp))
    np.savez(os.path.join(OUTPUT_DIR, "vgp_elbo.npz"), elbo=vgp_elbo)
    if LEARN_PRIOR_SDE:
        np.savez(os.path.join(OUTPUT_DIR, "vgp_learnt_sde.npz"), decay=v_gp_prior_decay_values)


def plot_prior_decay_learn_evolution():
    plt.clf()
    plt.hlines(-1 * DECAY, 0, max(len(v_gp_prior_decay_values), len(ssm_prior_decay_values)),
               label="True Value", color="black", linestyles="dashed")
    plt.plot(v_gp_prior_decay_values, label="VGP", color="green")
    plt.plot(ssm_prior_decay_values, label="SDE-SSM", color="blue")
    plt.title("Prior Learning (decay)")
    plt.legend()
    plt.ylabel("decay")

    wandb.log({"prior-learning": wandb.Image(plt)})

    plt.savefig(os.path.join(OUTPUT_DIR, "prior_learning_decay.svg"))
    plt.show()


def plot_elbo_bound():
    """Plot ELBO bound to see the learning objective bound"""
    decay_value_range = np.linspace(0.01, DECAY + 2.5, 10)
    # gpr_taylor_elbo_vals = []
    ssm_elbo_vals = []
    vgp_elbo_vals = []
    true_q = Q * tf.ones((1, 1), dtype=DTYPE)

    for decay_val in decay_value_range:
        # kernel = OrnsteinUhlenbeck(decay=decay_val, diffusion=q)
        # cvi_gpr_taylor_model.orig_kernel = kernel
        # gpr_taylor_elbo_vals.append(cvi_gpr_taylor_model.classic_elbo().numpy().item())

        ssm_model.prior_sde = OrnsteinUhlenbeckSDE(decay=decay_val, q=true_q)
        ssm_model._linearize_prior()  # To linearize the new prior
        ssm_elbo_vals.append(ssm_model.classic_elbo())

        vgp_model.prior_sde = OrnsteinUhlenbeckSDE(decay=decay_val, q=true_q)
        vgp_elbo_vals.append(vgp_model.elbo())

    plt.clf()
    plt.subplots(1, 1, figsize=(5, 5))
    plt.plot(decay_value_range, ssm_elbo_vals, label="SDE-SSM")
    plt.plot(decay_value_range, vgp_elbo_vals, label="VGP")
    # plt.plot(decay_value_range, gpr_taylor_elbo_vals, label="CVI-GPR (Taylor) ELBO", alpha=0.2, linestyle="dashed",
    #          color="black")
    plt.vlines(DECAY, np.min(vgp_elbo_vals), np.max(vgp_elbo_vals))
    plt.legend()
    wandb.log({"elbo_bound": wandb.Image(plt)})
    plt.savefig(os.path.join(OUTPUT_DIR, "elbo_bound.svg"))
    plt.show()


def vgp_from_sdessm(sde_ssm_model: SDESSM):
    """Load the posterior A and b value from SDE-SSM trained model and print ELBO and NLPD"""
    A, b, post_ssm_params = sde_ssm_model.get_posterior_drift_params()

    if LEARN_PRIOR_SDE:
        vgp_prior_decay = INITIAL_PRIOR_VALUE
        true_q = Q * tf.ones((1, 1), dtype=DTYPE)
        # As prior OU SDE doesn't have a negative sign inside it.
        prior_sde = PriorOUSDE(initial_val=-1*vgp_prior_decay, q=true_q)
    else:
        vgp_prior_decay = DECAY * tf.ones((1, 1), dtype=DTYPE)
        true_q = Q * tf.ones((1, 1), dtype=DTYPE)
        prior_sde = OrnsteinUhlenbeckSDE(decay=vgp_prior_decay, q=true_q)

    likelihood = Gaussian(NOISE_STDDEV**2)
    vgp_model = VariationalMarkovGP(input_data=OBSERVATION_DATA, prior_sde=prior_sde, grid=TIME_GRID,
                                    likelihood=likelihood)

    # Steady covariance
    initial_cov = Q / (2 * prior_sde.decay)
    vgp_model.p_initial_cov = tf.reshape(initial_cov, vgp_model.p_initial_cov.shape)
    vgp_model.p_initial_mean = tf.zeros_like(vgp_model.p_initial_mean)

    vgp_model.q_initial_mean = tf.reshape(post_ssm_params[4], vgp_model.q_initial_mean.shape)
    vgp_model.q_initial_cov = tf.reshape(tf.math.square(post_ssm_params[2]), shape=vgp_model.q_initial_cov.shape)

    # -1 because of how VGP is parameterized
    vgp_model.A = -1 * tf.concat([A, -1 * tf.ones((1, 1, 1), dtype=A.dtype)], axis=0)
    vgp_model.b = tf.concat([b, tf.zeros((1, 1), dtype=b.dtype)], axis=0)

    print(f"VGP (SDE-SSM params) ELBO: {vgp_model.elbo()}")

    pred_idx = list((tf.where(TIME_GRID == TEST_DATA[0][..., None])[:, 1]).numpy())
    m_vgp, s_std_vgp = predict_vgp(vgp_model, NOISE_STDDEV)
    vgp_chol_covar = tf.reshape(tf.gather(s_std_vgp, pred_idx) + NOISE_STDDEV, (-1, 1, 1))
    vgp_lpd = gaussian_log_predictive_density(mean=tf.gather(m_vgp, pred_idx, axis=0), chol_covariance=vgp_chol_covar,
                                              x=tf.reshape(TEST_DATA[1], (-1,)))
    vgp_nlpd = -1 * tf.reduce_mean(vgp_lpd).numpy().item()
    print(f"VGP (SDE-SSM params) NLPD: {vgp_nlpd}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run VGP and SDE-SSM for OU process')

    parser.add_argument('-dir', '--data_dir', type=str, help='Data directory of the OU data.', required=True)
    parser.add_argument('-wandb_username', type=str, help='Wandb username to be used for logging', default="")
    parser.add_argument('-l', '--learn_prior_sde', type=bool, default=False, help='Train Prior SDE or not.')
    parser.add_argument('-d', '--prior_decay', type=float, default=1.,
                        help='Prior decay value to be used when learning the prior SDE.')
    parser.add_argument('-log', type=bool, default=False, help='Whether to log in wandb or not')
    parser.add_argument('-dt', type=float, default=0., help='Modify dt for time-grid.')
    parser.add_argument('-sites_lr', type=float, default=0.5, help='Learning rate for sites.')
    parser.add_argument('-prior_ssm_lr', type=float, default=0.01, help='Learning rate for prior learning in SSM.')
    parser.add_argument('-prior_vgp_lr', type=float, default=0.01, help='Learning rate for prior learning in VGP.')
    parser.add_argument('-vgp_lr', type=float, default=0.01, help='Learning rate for VGP parameters.')

    print(f"True decay value of the OU SDE is {DECAY}")
    print(f"Noise std-dev is {NOISE_STDDEV}")

    args = parser.parse_args()
    LEARN_PRIOR_SDE = args.learn_prior_sde

    load_data(args.data_dir)

    plot_data()

    if args.dt != 0:
        modify_time_grid(args.dt)

    set_output_dir()

    init_wandb(args.wandb_username, args.log, args.sites_lr, args.prior_ssm_lr, args.vgp_lr, args.prior_vgp_lr)

    INITIAL_PRIOR_VALUE = args.prior_decay

    if LEARN_PRIOR_SDE:
        gpr_model, cvi_params = cvi_gpr()
    else:
        gpr_model = gpr_taylor()

    ssm_model, ssm_elbo_vals, ssm_prior_prior_vals = perform_sde_ssm(args.sites_lr, args.prior_ssm_lr)
    if LEARN_PRIOR_SDE:
        ssm_prior_decay_values = ssm_prior_prior_vals[0]

    vgp_model, vgp_elbo_vals, vgp_prior_prior_vals = perform_vgp(args.vgp_lr, args.prior_vgp_lr)
    if LEARN_PRIOR_SDE:
        v_gp_prior_decay_values = vgp_prior_prior_vals[0]

    compare_plot_posterior(gpr_model, ssm_model, vgp_model)

    plot_elbo(ssm_elbo_vals, vgp_elbo_vals)

    calculate_nlpd(ssm_model, vgp_model)

    save_data(ssm_elbo_vals, vgp_elbo_vals)

    if LEARN_PRIOR_SDE:
        plot_prior_decay_learn_evolution()
    else:
        plot_elbo_bound()

        vgp_from_sdessm(ssm_model)
