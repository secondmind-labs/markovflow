"""OU SDE CVI vs SDE VI"""
import os
import argparse
import random

import gpflow
import matplotlib.pyplot as plt
import numpy as np

# Don't use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# Restrict TensorFlow to only use the first GPU
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     tf.config.set_visible_devices(gpus[1], 'GPU')

from gpflow import default_float
from gpflow.likelihoods import Gaussian
import wandb

from markovflow.sde.sde import PriorOUSDE
from markovflow.kernels import OrnsteinUhlenbeck
from markovflow.models.cvi_sde import SDESSM
from markovflow.models.vi_sde import VariationalMarkovGP
from markovflow.sde.sde_utils import gaussian_log_predictive_density

from docs.sde.sde_exp_utils import predict_vgp, predict_ssm, plot_observations, plot_posterior, get_cvi_gpr_taylor, \
    predict_cvi_gpr_taylor, get_cvi_gpr, predict_cvi_gpr
from docs.sde.t_vgp_trainer import tVGPTrainer

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
INITIAL_PRIOR_VALUE = -1.
UPDATE_ALL_SITES = False


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
    TIME_GRID = tf.cast(np.linspace(T0, T1, int((T1 - T0) // DT) + 2), dtype=DTYPE).numpy()


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


def set_output_dir(dir_name):
    """Create output directory"""
    global OUTPUT_DIR
    if dir_name is not None:
        OUTPUT_DIR = os.path.join(DATA_PATH, dir_name)
    elif LEARN_PRIOR_SDE:
        OUTPUT_DIR = os.path.join(DATA_PATH, "learning")
    else:
        OUTPUT_DIR = os.path.join(DATA_PATH, "inference")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def init_wandb(uname: str, log: bool = False, data_sites_lr: float = 0.5, ssm_prior_lr: float = 0.01,
               vgp_lr: float = 0.01, vgp_prior_lr: float = 0.01, x0_lr: float = 0.01, all_sites_lr: float = 0.1):
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
        "data_sites_lr": data_sites_lr,
        "all_sites_lr": all_sites_lr,
        "SSM_prior_lr": ssm_prior_lr,
        "vgp_lr": vgp_lr,
        "vgp_prior_lr": vgp_prior_lr,
        "vgp_x0_lr": x0_lr
    }

    """Logging init"""
    exp_name = "OU-" + config["seed"] + "-" + str(LEARN_PRIOR_SDE) + "-dt-" + str(DT) + "-" + str(random.randint(1,
                                                                                                                 10000))
    wandb.init(project="VI-SDE", entity=uname, config=config, name=exp_name, group="OU")


def gpr_taylor():
    likelihood_gpr = Gaussian(NOISE_STDDEV ** 2)

    kernel = OrnsteinUhlenbeck(decay=-1 * INITIAL_PRIOR_VALUE, diffusion=Q)
    gpflow.set_trainable(kernel.diffusion, False)

    cvi_gpr_taylor_model, cvi_taylor_params, cvi_taylor_elbo_vals = get_cvi_gpr_taylor(OBSERVATION_DATA, kernel,
                                                                                       TIME_GRID, likelihood_gpr,
                                                                                       train=LEARN_PRIOR_SDE,
                                                                                       sites_lr=1.)
    print(f"CVI-GPR (Taylor) ELBO: {cvi_gpr_taylor_model.classic_elbo()}")

    return cvi_gpr_taylor_model


def cvi_gpr():
    likelihood_gpr = Gaussian(NOISE_STDDEV ** 2)

    kernel = OrnsteinUhlenbeck(decay=-1 * INITIAL_PRIOR_VALUE, diffusion=Q)
    gpflow.set_trainable(kernel.diffusion, False)

    cvi_gpr_model, cvi_params = get_cvi_gpr(OBSERVATION_DATA, kernel, likelihood_gpr, train=LEARN_PRIOR_SDE,
                                            sites_lr=0.9)
    print(f"CVI-GPR ELBO: {cvi_gpr_model.classic_elbo()}")
    return cvi_gpr_model, cvi_params


def perform_sde_ssm(data_sites_lr: float = 0.5, all_sites_lr: float = 0.1, prior_lr: float = 0.01):
    true_q = Q * tf.ones((1, 1), dtype=DTYPE)
    # As prior OU SDE doesn't have a negative sign inside it.
    prior_sde_ssm = PriorOUSDE(initial_val=INITIAL_PRIOR_VALUE, q=true_q)

    # likelihood
    likelihood_ssm = Gaussian(NOISE_STDDEV ** 2)

    t_vgp_trainer = tVGPTrainer(observation_data=OBSERVATION_DATA, likelihood=likelihood_ssm, time_grid=TIME_GRID,
                                prior_sde=prior_sde_ssm, data_sites_lr=data_sites_lr,
                                all_sites_lr=all_sites_lr, prior_sde_lr=prior_lr, test_data=TEST_DATA,
                                update_all_sites=UPDATE_ALL_SITES)

    ssm_elbo, ssm_prior_vals, ssm_m_step_data = t_vgp_trainer.run(update_prior=LEARN_PRIOR_SDE)

    return t_vgp_trainer.tvgp_model, ssm_elbo, ssm_prior_vals, ssm_m_step_data


def perform_vgp(vgp_lr: float = 0.01, prior_lr: float = 0.01, x0_lr: float = 0.01):
    # Prior SDE
    true_q = Q * tf.ones((1, 1), dtype=DTYPE)
    # As prior OU SDE doesn't have a negative sign inside it.
    prior_sde_vgp = PriorOUSDE(initial_val=INITIAL_PRIOR_VALUE, q=true_q)
    initial_cov = Q / (2 * -1 * INITIAL_PRIOR_VALUE)  # Steady covariance

    # likelihood
    likelihood_vgp = Gaussian(NOISE_STDDEV ** 2)

    vgp_model = VariationalMarkovGP(input_data=OBSERVATION_DATA,
                                    prior_sde=prior_sde_vgp, grid=TIME_GRID, likelihood=likelihood_vgp,
                                    lr=vgp_lr, prior_params_lr=prior_lr, test_data=TEST_DATA, initial_state_lr=x0_lr)

    vgp_model.p_initial_cov = tf.reshape(initial_cov, vgp_model.p_initial_cov.shape)
    vgp_model.q_initial_cov = tf.identity(vgp_model.p_initial_cov)
    vgp_model.p_initial_mean = tf.zeros_like(vgp_model.p_initial_mean)
    vgp_model.q_initial_mean = tf.identity(vgp_model.p_initial_mean)

    vgp_model.A = -1 * INITIAL_PRIOR_VALUE + 0. * vgp_model.A

    v_gp_elbo, v_gp_prior_vals, v_gp_m_step_data = vgp_model.run(update_prior=LEARN_PRIOR_SDE)

    return vgp_model, v_gp_elbo, v_gp_prior_vals, v_gp_m_step_data


def compare_plot_posterior(cvi_gpr_model, ssm_model, vgp_model):
    plt.clf()
    plot_observations(OBSERVATION_DATA[0], OBSERVATION_DATA[1])
    if TEST_DATA is not None:
        plt.plot(TEST_DATA[0].reshape(-1), TEST_DATA[1].reshape(-1), 'x', color="red", ms=8, mew=2,
                 label="Test Observations (Y)")

    if LEARN_PRIOR_SDE:
        m_gpr, s_std_gpr = predict_cvi_gpr(cvi_gpr_model, TIME_GRID, NOISE_STDDEV)
    else:
        m_gpr, s_std_gpr = predict_cvi_gpr_taylor(cvi_gpr_model, NOISE_STDDEV)

    if ssm_model is not None:
        m_ssm, s_std_ssm = predict_ssm(ssm_model, NOISE_STDDEV)
        plot_posterior(m_ssm, s_std_ssm, TIME_GRID, "SDE-SSM")

    if vgp_model is not None:
        m_vgp, s_std_vgp = predict_vgp(vgp_model, NOISE_STDDEV)
        plot_posterior(m_vgp, s_std_vgp, TIME_GRID, "VGP")

    # plt.vlines(time_grid.reshape(-1), -2, 2, alpha=0.2, color="black")
    plot_posterior(m_gpr, s_std_gpr, TIME_GRID, "GPR")

    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "posterior.svg"))
    wandb.log({"posterior": wandb.Image(plt)})

    plt.show()


def plot_elbo(ssm_elbo, v_gp_elbo):
    """ELBO comparison"""
    plt.clf()
    if len(ssm_elbo) > 0:
        plt.plot(ssm_elbo, label="SDE-SSM")
    if len(v_gp_elbo) > 0:
        plt.plot(v_gp_elbo, label="VGP")
    plt.title("ELBO")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "elbo.svg"))

    wandb.log({"ELBO-Comparison": wandb.Image(plt)})

    plt.show()


def calculate_nlpd(ssm_model: SDESSM, vgp_model: VariationalMarkovGP):
    """Calculate NLPD on the test set for VGP and SDE-SSM"""

    if TEST_DATA is None:
        print("Test data is not available so slipping calculating NLPD!")
        return

    pred_idx = list((tf.where(TIME_GRID == TEST_DATA[0][..., None])[:, 1]).numpy())

    if ssm_model is not None:
        m_ssm, s_std_ssm = predict_ssm(ssm_model, NOISE_STDDEV)
        ssm_chol_covar = tf.reshape(tf.gather(s_std_ssm, pred_idx, axis=1), (-1, 1, 1))
        ssm_lpd = gaussian_log_predictive_density(mean=tf.gather(m_ssm, pred_idx, axis=0),
                                                  chol_covariance=ssm_chol_covar,
                                                  x=tf.reshape(TEST_DATA[1], (-1,)))
        ssm_nlpd = -1 * tf.reduce_mean(ssm_lpd).numpy().item()
        print(f"SDE-SSM NLPD: {ssm_nlpd}")

    if vgp_model is not None:
        m_vgp, s_std_vgp = predict_vgp(vgp_model, NOISE_STDDEV)
        vgp_chol_covar = tf.reshape(tf.gather(s_std_vgp, pred_idx), (-1, 1, 1))
        vgp_lpd = gaussian_log_predictive_density(mean=tf.gather(m_vgp, pred_idx, axis=0),
                                                  chol_covariance=vgp_chol_covar,
                                                  x=tf.reshape(TEST_DATA[1], (-1,)))
        vgp_nlpd = -1 * tf.reduce_mean(vgp_lpd).numpy().item()
        print(f"VGP NLPD: {vgp_nlpd}")


def save_data(ssm_elbo, vgp_elbo, vgp_m_step_data, ssm_m_step_data):
    if ssm_model is not None:
        m_ssm, s_std_ssm = predict_ssm(ssm_model, NOISE_STDDEV)
        np.savez(os.path.join(OUTPUT_DIR, "ssm_data_sites.npz"), nat1=ssm_model.data_sites.nat1.numpy(),
                 nat2=ssm_model.data_sites.nat2.numpy(), log_norm=ssm_model.data_sites.log_norm.numpy())

        np.savez(os.path.join(OUTPUT_DIR, "ssm_inference.npz"), m=m_ssm, S=tf.square(s_std_ssm))
        np.savez(os.path.join(OUTPUT_DIR, "ssm_elbo.npz"), elbo=ssm_elbo)

        if LEARN_PRIOR_SDE:
            np.savez(os.path.join(OUTPUT_DIR, "ssm_learnt_sde.npz"), decay=ssm_prior_decay_values)
            np.savez(os.path.join(OUTPUT_DIR, "ssm_m_step.npz"), vals=ssm_m_step_data[0])

    if vgp_model is not None:
        m_vgp, s_std_vgp = predict_vgp(vgp_model, NOISE_STDDEV)
        np.savez(os.path.join(OUTPUT_DIR, "vgp_A_b.npz"), A=vgp_model.A.numpy(), b=vgp_model.b.numpy())
        np.savez(os.path.join(OUTPUT_DIR, "vgp_lagrange.npz"), psi_lagrange=vgp_model.psi_lagrange.numpy(),
                 lambda_lagrange=vgp_model.lambda_lagrange.numpy())

        np.savez(os.path.join(OUTPUT_DIR, "vgp_inference.npz"), m=m_vgp, S=tf.square(s_std_vgp))
        np.savez(os.path.join(OUTPUT_DIR, "vgp_elbo.npz"), elbo=vgp_elbo)
        if LEARN_PRIOR_SDE:
            np.savez(os.path.join(OUTPUT_DIR, "vgp_learnt_sde.npz"), decay=v_gp_prior_decay_values)
            np.savez(os.path.join(OUTPUT_DIR, "vgp_m_step.npz"), vals=vgp_m_step_data[0])


def plot_prior_decay_learn_evolution():
    plt.clf()
    plt.hlines(-1 * DECAY, 0, max(len(v_gp_prior_decay_values), len(ssm_prior_decay_values)),
               label="True Value", color="black", linestyles="dashed")
    plt.hlines(-1 * cvi_params[0][-1], 0, max(len(v_gp_prior_decay_values), len(ssm_prior_decay_values)),
               label="GPR Value", color="red", linestyles="dashed")
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
    decay_value_range = np.linspace(0.01, DECAY + 2.5, 40)
    gpr_taylor_elbo_vals = []
    ssm_elbo_vals = []
    vgp_elbo_vals = []
    true_q = Q * tf.ones((1, 1), dtype=DTYPE)

    for decay_val in decay_value_range:
        kernel = OrnsteinUhlenbeck(decay=decay_val, diffusion=Q)
        gpr_model.orig_kernel = kernel
        gpr_taylor_elbo_vals.append(gpr_model.classic_elbo().numpy().item())

        if ssm_model is not None:
            ssm_model.prior_sde = PriorOUSDE(-1 * decay_val, q=true_q)
            # Steady covariance
            ssm_model.initial_chol_cov = tf.linalg.cholesky((Q / (2 * decay_val)) + 0. * ssm_model.initial_chol_cov)
            ssm_model._linearize_prior()  # To linearize the new prior
            ssm_elbo_vals.append(ssm_model.classic_elbo())

        if vgp_model is not None:
            vgp_model.prior_sde = PriorOUSDE(-1 * decay_val, q=true_q)
            # Steady covariance
            vgp_model.p_initial_cov = (Q / (2 * decay_val)) + 0. * vgp_model.p_initial_cov
            vgp_elbo_vals.append(vgp_model.elbo())

    plt.clf()
    plt.subplots(1, 1, figsize=(5, 5))
    plt.plot(-1 * decay_value_range, ssm_elbo_vals, label="SDE-SSM")
    plt.plot(-1 * decay_value_range, vgp_elbo_vals, label="VGP")
    plt.plot(-1 * decay_value_range, gpr_taylor_elbo_vals, label="CVI-GPR (Taylor) ELBO", alpha=0.2, linestyle="dashed",
             color="black")
    plt.vlines(-1 * DECAY, np.min(vgp_elbo_vals) - 0.5, np.max(ssm_elbo_vals) + 0.5, linestyle="dashed", alpha=0.2,
               color="red", label="Simulating Decay")
    plt.vlines(INITIAL_PRIOR_VALUE, np.min(vgp_elbo_vals) - 0.5, np.max(ssm_elbo_vals) + 0.5, linestyle="dashed",
               alpha=0.2, color="green", label="Initial Decay")
    plt.legend()
    plt.xlim(-1 * np.max(decay_value_range), -1 * np.min(decay_value_range))

    wandb.log({"elbo_bound": wandb.Image(plt)})
    plt.savefig(os.path.join(OUTPUT_DIR, "elbo_bound.svg"))
    plt.show()

    np.savez(os.path.join(OUTPUT_DIR, "elbo_bound.npz"), ssm_elbo=ssm_elbo_vals, vgp_elbo=vgp_elbo_vals,
             gpr_taylor_elbo=gpr_taylor_elbo_vals, decay_values=decay_value_range)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run VGP and SDE-SSM for OU process')

    parser.add_argument('-dir', '--data_dir', type=str, help='Data directory of the OU data.', required=True)
    parser.add_argument('-wandb_username', type=str, help='Wandb username to be used for logging', default="")
    parser.add_argument('-l', '--learn_prior_sde', choices=('True', 'False'), default='False', help='Train Prior SDE or not.')
    parser.add_argument('-d', '--prior_decay', type=float, default=-1.,
                        help='Prior decay value to be used when learning the prior SDE.')
    parser.add_argument('-log', choices=('True', 'False'), default='False', help='Whether to log in wandb or not')
    parser.add_argument('-dt', type=float, default=0., help='Modify dt for time-grid.')
    parser.add_argument('-data_sites_lr', type=float, default=0.5,
                        help='Learning rate for data-sites. Set to 0 if not training.')
    parser.add_argument('-all_sites_lr', type=float, default=0.1, help='Learning rate for all-sites.')
    parser.add_argument('-prior_ssm_lr', type=float, default=0.01, help='Learning rate for prior learning in SSM.')
    parser.add_argument('-prior_vgp_lr', type=float, default=0.01, help='Learning rate for prior learning in VGP.')
    parser.add_argument('-vgp_lr', type=float, default=0.01,
                        help='Learning rate for VGP parameters. Set to 0 if not training.')
    parser.add_argument('-vgp_x0_lr', type=float, default=0.01, help='Learning rate for VGP initial state.')
    parser.add_argument('-all_sites', choices=('True', 'False'), default='False',
                        help='Update all sites using cross-term or only data-sites')
    parser.add_argument('-o', type=str, default=None, help='Output directory name')

    print(f"True decay value of the OU SDE is {DECAY}")
    print(f"Noise std-dev is {NOISE_STDDEV}")

    args = parser.parse_args()
    LEARN_PRIOR_SDE = args.learn_prior_sde == 'True'

    UPDATE_ALL_SITES = args.all_sites == 'True'

    load_data(args.data_dir)

    plot_data()

    if args.dt != 0:
        modify_time_grid(args.dt)

    assert TIME_GRID[-1] == T1
    assert TIME_GRID[1] - TIME_GRID[0] == DT

    set_output_dir(args.o)

    log = args.log == 'True'
    init_wandb(args.wandb_username, args.log, args.data_sites_lr, args.prior_ssm_lr, args.vgp_lr, args.prior_vgp_lr,
               args.vgp_x0_lr, args.all_sites_lr)

    INITIAL_PRIOR_VALUE = args.prior_decay

    if LEARN_PRIOR_SDE:
        gpr_model, cvi_params = cvi_gpr()
        print(f"Learnt CVI param : {cvi_params}")
    else:
        gpr_model = gpr_taylor()

    ssm_m_step_data = None
    if args.data_sites_lr > 0.:
        ssm_model, ssm_elbo_vals, ssm_prior_prior_vals, ssm_m_step_data = perform_sde_ssm(args.data_sites_lr,
                                                                                          args.all_sites_lr,
                                                                                          args.prior_ssm_lr)
        if LEARN_PRIOR_SDE:
            ssm_prior_decay_values = ssm_prior_prior_vals[0]
    else:
        ssm_model = None
        ssm_elbo_vals = []
        ssm_prior_decay_values = []

    vgp_m_step_data = None
    if args.vgp_lr > 0.:
        vgp_model, vgp_elbo_vals, vgp_prior_prior_vals, vgp_m_step_data = perform_vgp(args.vgp_lr, args.prior_vgp_lr,
                                                                                      args.vgp_x0_lr)
        if LEARN_PRIOR_SDE:
            v_gp_prior_decay_values = vgp_prior_prior_vals[0]
    else:
        vgp_model = None
        vgp_elbo_vals = []
        v_gp_prior_decay_values = []

    compare_plot_posterior(gpr_model, ssm_model, vgp_model)

    plot_elbo(ssm_elbo_vals, vgp_elbo_vals)

    calculate_nlpd(ssm_model, vgp_model)

    save_data(ssm_elbo_vals, vgp_elbo_vals, vgp_m_step_data, ssm_m_step_data)

    if ssm_model is not None and vgp_model is not None:
        if LEARN_PRIOR_SDE:
            plot_prior_decay_learn_evolution()
        else:
            plot_elbo_bound()
