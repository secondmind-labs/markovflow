"""Learning bound for SDE parameters"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from gpflow import default_float
from gpflow.likelihoods import Gaussian
from gpflow.base import Parameter
import wandb
import argparse

from markovflow.sde.sde import PriorDoubleWellSDE
from markovflow.models.cvi_sde import SDESSM


DTYPE = default_float()
MODEL_DIR = ""
Q = 1.
NOISE_STDDEV = 0.1 * np.ones((1, 1))
TIME_GRID = tf.zeros((1, 1))
OBSERVATION_DATA = ()
SDESSM_MODEL = None
ELBO_VALS = []
A_VALUE_RANGE = []
C_VALUE_RANGE = []


def load_data(data_dir):
    """
    Get Data
    """
    global Q, NOISE_STDDEV, TIME_GRID, OBSERVATION_DATA

    data = np.load(os.path.join(data_dir, "data.npz"))
    Q = data["q"]
    NOISE_STDDEV = data["noise_stddev"]
    OBSERVATION_DATA = (data["observation_grid"], tf.squeeze(data["observation_vals"], axis=0).numpy())
    TIME_GRID = data["time_grid"]


def load_model():
    global SDESSM_MODEL
    """Load SDE-SSM model with trained variables"""
    true_q = Q * tf.ones((1, 1), dtype=DTYPE)
    likelihood = Gaussian(NOISE_STDDEV ** 2)

    prior_sde = PriorDoubleWellSDE(q=true_q)

    # model
    SDESSM_MODEL = SDESSM(input_data=OBSERVATION_DATA, prior_sde=prior_sde, grid=TIME_GRID, likelihood=likelihood)

    # Load trained model variables
    data_sites = np.load(os.path.join(MODEL_DIR, "ssm_data_sites.npz"))
    SDESSM_MODEL.data_sites.nat1 = Parameter(data_sites["nat1"])
    SDESSM_MODEL.data_sites.nat2 = Parameter(data_sites["nat2"])
    SDESSM_MODEL.data_sites.log_norm = Parameter(data_sites["log_norm"])

    sites = np.load(os.path.join(MODEL_DIR, "ssm_sites.npz"))
    SDESSM_MODEL.sites_nat1 = sites["nat1"]
    SDESSM_MODEL.sites_nat2 = sites["nat2"]

    lin_path = np.load(os.path.join(MODEL_DIR, "ssm_linearization_path.npz"))
    SDESSM_MODEL.linearization_pnts = (lin_path["fx_mus"].reshape(SDESSM_MODEL.linearization_pnts[0].shape),
                                       lin_path["fx_covs"].reshape(SDESSM_MODEL.linearization_pnts[1].shape))

    ssm_learning_path = os.path.join(MODEL_DIR, "ssm_learnt_sde.npz")
    ssm_learning = np.load(ssm_learning_path)
    SDESSM_MODEL.prior_sde.a = ssm_learning["a"][-1] * tf.ones_like(SDESSM_MODEL.prior_sde.a)
    SDESSM_MODEL.prior_sde.c = ssm_learning["c"][-1] * tf.ones_like(SDESSM_MODEL.prior_sde.c)

    q_path = np.load(os.path.join(MODEL_DIR, "ssm_inference.npz"))
    SDESSM_MODEL.fx_mus = q_path["m"].reshape(SDESSM_MODEL.fx_mus.shape) * tf.ones_like(SDESSM_MODEL.fx_mus)
    # cov without the noise variance
    fx_covs = q_path["S"] - NOISE_STDDEV ** 2
    SDESSM_MODEL.fx_covs = fx_covs.reshape(SDESSM_MODEL.fx_covs.shape)

    SDESSM_MODEL._linearize_prior()


def compare_elbo():
    """Compare ELBO between loaded moel and trained model"""
    loaded_model_elbo = SDESSM_MODEL.classic_elbo()
    print(f"ELBO (Loaded model): {loaded_model_elbo}")

    trained_model_elbo = np.load(os.path.join(MODEL_DIR, "ssm_elbo.npz"))["elbo"][-1]
    print(f"ELBO (Trained model) : {trained_model_elbo}")

    np.testing.assert_array_almost_equal(trained_model_elbo, loaded_model_elbo)


def calculate_elbo_bound(n=30):
    """ELBO BOUND"""
    global ELBO_VALS, A_VALUE_RANGE, C_VALUE_RANGE

    A_VALUE_RANGE = np.linspace(0.2, 6, n).reshape((-1, 1))
    C_VALUE_RANGE = np.linspace(0.2, 2., n).reshape((1, -1))

    A_VALUE_RANGE = np.repeat(A_VALUE_RANGE, n, axis=1)
    C_VALUE_RANGE = np.repeat(C_VALUE_RANGE, n, axis=0)

    true_q = Q * tf.ones((1, 1), dtype=DTYPE)

    for a, c in zip(A_VALUE_RANGE.reshape(-1), C_VALUE_RANGE.reshape(-1)):
        print(f"Calculating ELBO bound for a={a}, c={c}")
        SDESSM_MODEL.prior_sde = PriorDoubleWellSDE(q=true_q, initial_a_val=a, initial_c_val=c)
        SDESSM_MODEL._linearize_prior()  # To linearize the new prior
        ELBO_VALS.append(SDESSM_MODEL.classic_elbo().numpy().item())

    ELBO_VALS = np.array(ELBO_VALS).reshape((n, n)).T


def normalize_elbo_vals():
    global ELBO_VALS
    mean_val = np.mean(ELBO_VALS)
    std_val = np.std(ELBO_VALS)

    ELBO_VALS = (ELBO_VALS - mean_val) / std_val


def plot_elbo_bound():
    clipped_elbo_vals = np.clip(ELBO_VALS, -1, np.max(ELBO_VALS))
    levels = np.linspace(-1, 1, 25)

    plt.clf()
    fig = plt.figure(1, figsize=(6, 5))
    contour1 = plt.contourf(A_VALUE_RANGE[:, 0], C_VALUE_RANGE[0], clipped_elbo_vals, levels=levels, cmap="gray")
    fig.colorbar(contour1)
    plt.savefig(os.path.join(MODEL_DIR, "ssm_learning_bound.svg"))
    plt.show()

    np.savez(os.path.join(MODEL_DIR, "ssm_learning_bound.npz"), elbo=clipped_elbo_vals, a=A_VALUE_RANGE,
             c=C_VALUE_RANGE)


def plot_learning_plot():
    ssm_learning_path = os.path.join(MODEL_DIR, "ssm_learnt_sde.npz")
    ssm_learning = np.load(ssm_learning_path)
    ssm_learnt_a = ssm_learning["a"]
    ssm_learnt_c = ssm_learning["c"]

    clipped_elbo_vals = np.clip(ELBO_VALS, -1, np.max(ELBO_VALS))
    levels = np.linspace(-1, 1, 25)

    plt.clf()
    fig = plt.figure(1, figsize=(6, 5))
    contour1 = plt.contourf(A_VALUE_RANGE[:, 0], C_VALUE_RANGE[0], clipped_elbo_vals, levels=levels, cmap="gray")
    fig.colorbar(contour1)
    plt.plot(ssm_learnt_a, ssm_learnt_c, "x")

    plt.savefig(os.path.join(MODEL_DIR, "ssm_learning_bound_learning.svg"))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SDE-SSM ELBO bound for DW process')

    parser.add_argument('-dir', '--data_dir', type=str, help='Directory of the saved SDE-SSM model', required=True)

    args = parser.parse_args()

    # setting wandb loggin to off
    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project="VI-SDE")

    MODEL_DIR = args.data_dir

    load_data("/".join(MODEL_DIR.split("/")[:-1]))

    load_model()

    compare_elbo()

    calculate_elbo_bound()

    normalize_elbo_vals()

    plot_elbo_bound()

    plot_learning_plot()