"""OU SDE-SSM"""
import os
import argparse

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
from markovflow.models.cvi_sde import SDESSM
from markovflow.sde.sde_utils import gaussian_log_predictive_density

from docs.sde.sde_exp_utils import predict_ssm, plot_observations, plot_posterior

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
    if data["test_grid"].shape[0] > 0:
        TEST_DATA = (data["test_grid"], data["test_vals"])
    else:
        TEST_DATA = None

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
    if TEST_DATA is not None:
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
        OUTPUT_DIR = os.path.join(DATA_PATH, "learning_ssm")
    else:
        OUTPUT_DIR = os.path.join(DATA_PATH, "inference_ssm")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def init_wandb(uname: str, log: bool = False, sites_lr: float = 0.5, ssm_prior_lr: float = 0.01):
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
    }

    """Logging init"""
    wandb.init(project="VI-SDE", entity=uname, config=config)


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


def ssm_plot_posterior(ssm_model):
    plt.clf()
    plot_observations(OBSERVATION_DATA[0], OBSERVATION_DATA[1])
    if TEST_DATA is not None:
        plt.plot(TEST_DATA[0].reshape(-1), TEST_DATA[1].reshape(-1), 'x', color="red", ms=8, mew=2,
                 label="Test Observations (Y)")

    m_ssm, s_std_ssm = predict_ssm(ssm_model, NOISE_STDDEV)
    plot_posterior(m_ssm, s_std_ssm, TIME_GRID, "SDE-SSM")
    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "posterior.svg"))
    wandb.log({"posterior": wandb.Image(plt)})

    plt.show()


def plot_elbo(ssm_elbo):
    """ELBO comparison"""
    plt.clf()
    plt.plot(ssm_elbo, label="SDE-SSM")
    plt.title("ELBO")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "elbo.svg"))

    wandb.log({"ELBO-Comparison": wandb.Image(plt)})

    plt.show()


def calculate_nlpd(ssm_model: SDESSM):
    """Calculate NLPD on the test set"""

    if TEST_DATA is None:
        print("Test data is not available so slipping calculating NLPD!")
        return

    m_ssm, s_std_ssm = predict_ssm(ssm_model, NOISE_STDDEV)

    pred_idx = list((tf.where(TIME_GRID == TEST_DATA[0][..., None])[:, 1]).numpy())
    ssm_chol_covar = tf.reshape(tf.gather(s_std_ssm, pred_idx, axis=1), (-1, 1, 1))
    ssm_lpd = gaussian_log_predictive_density(mean=tf.gather(m_ssm, pred_idx, axis=0), chol_covariance=ssm_chol_covar,
                                              x=tf.reshape(TEST_DATA[1], (-1,)))
    ssm_nlpd = -1 * tf.reduce_mean(ssm_lpd).numpy().item()
    print(f"SDE-SSM NLPD: {ssm_nlpd}")


def save_data(ssm_elbo):
    """Save data into npz"""
    m_ssm, s_std_ssm = predict_ssm(ssm_model, NOISE_STDDEV)

    """Save SDE-SSM data"""
    np.savez(os.path.join(OUTPUT_DIR, "ssm_data_sites.npz"), nat1=ssm_model.data_sites.nat1.numpy(),
             nat2=ssm_model.data_sites.nat2.numpy(), log_norm=ssm_model.data_sites.log_norm.numpy())

    np.savez(os.path.join(OUTPUT_DIR, "ssm_inference.npz"), m=m_ssm, S=tf.square(s_std_ssm))
    np.savez(os.path.join(OUTPUT_DIR, "ssm_elbo.npz"), elbo=ssm_elbo)

    if LEARN_PRIOR_SDE:
        np.savez(os.path.join(OUTPUT_DIR, "ssm_learnt_sde.npz"), decay=ssm_prior_decay_values)


def plot_prior_decay_learn_evolution():
    plt.clf()
    plt.hlines(-1 * DECAY, 0, max(len(ssm_prior_decay_values), len(ssm_prior_decay_values)),
               label="True Value", color="black", linestyles="dashed")
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
    ssm_elbo_vals = []
    true_q = Q * tf.ones((1, 1), dtype=DTYPE)

    for decay_val in decay_value_range:

        ssm_model.prior_sde = OrnsteinUhlenbeckSDE(decay=decay_val, q=true_q)
        ssm_model._linearize_prior()  # To linearize the new prior
        ssm_elbo_vals.append(ssm_model.classic_elbo())

    plt.clf()
    plt.subplots(1, 1, figsize=(5, 5))
    plt.plot(decay_value_range, ssm_elbo_vals, label="SDE-SSM")

    plt.legend()
    wandb.log({"elbo_bound": wandb.Image(plt)})
    plt.savefig(os.path.join(OUTPUT_DIR, "elbo_bound.svg"))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SDE-SSM for OU process')

    parser.add_argument('-dir', '--data_dir', type=str, help='Data directory of the OU data.', required=True)
    parser.add_argument('-wandb_username', type=str, help='Wandb username to be used for logging', default="")
    parser.add_argument('-l', '--learn_prior_sde', type=bool, default=False, help='Train Prior SDE or not.')
    parser.add_argument('-d', '--prior_decay', type=float, default=1.,
                        help='Prior decay value to be used when learning the prior SDE.')
    parser.add_argument('-log', type=bool, default=False, help='Whether to log in wandb or not')
    parser.add_argument('-dt', type=float, default=0., help='Modify dt for time-grid.')
    parser.add_argument('-sites_lr', type=float, default=0.5, help='Learning rate for sites.')
    parser.add_argument('-prior_ssm_lr', type=float, default=0.01, help='Learning rate for prior learning in SSM.')

    args = parser.parse_args()
    LEARN_PRIOR_SDE = args.learn_prior_sde

    load_data(args.data_dir)

    plot_data()

    if args.dt != 0:
        modify_time_grid(args.dt)

    assert TIME_GRID[-1] == T1

    set_output_dir()

    init_wandb(args.wandb_username, args.log, args.sites_lr, args.prior_ssm_lr)

    INITIAL_PRIOR_VALUE = args.prior_decay

    ssm_model, ssm_elbo_vals, ssm_prior_prior_vals = perform_sde_ssm(args.sites_lr, args.prior_ssm_lr)
    if LEARN_PRIOR_SDE:
        ssm_prior_decay_values = ssm_prior_prior_vals[0]

    ssm_plot_posterior(ssm_model)

    plot_elbo(ssm_elbo_vals)

    calculate_nlpd(ssm_model)

    save_data(ssm_elbo_vals)

    if LEARN_PRIOR_SDE:
        plot_prior_decay_learn_evolution()
    else:
        plot_elbo_bound()
