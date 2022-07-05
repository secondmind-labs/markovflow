"""Learning bound for SDE parameters"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from gpflow import default_float
from gpflow.likelihoods import Gaussian
import wandb
import argparse

from markovflow.sde.sde import PriorDoubleWellSDE
from markovflow.models.vi_sde import VariationalMarkovGP


DTYPE = default_float()
MODEL_DIR = ""
Q = 1.
NOISE_STDDEV = 0.1 * np.ones((1, 1))
TIME_GRID = tf.zeros((1, 1))
OBSERVATION_DATA = ()
VGP_MODEL = None
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
    global VGP_MODEL
    """Load VGP model with trained variables"""
    true_q = Q * tf.ones((1, 1), dtype=DTYPE)
    likelihood = Gaussian(NOISE_STDDEV ** 2)

    prior_sde = PriorDoubleWellSDE(q=true_q)

    VGP_MODEL = VariationalMarkovGP(input_data=OBSERVATION_DATA,
                                    prior_sde=prior_sde, grid=TIME_GRID, likelihood=likelihood)

    # Load trained model variables
    A_b_data = np.load(os.path.join(MODEL_DIR, "vgp_A_b.npz"))
    lagrange_data = np.load(os.path.join(MODEL_DIR, "vgp_lagrange.npz"))
    sde_params = np.load(os.path.join(MODEL_DIR, "vgp_learnt_sde.npz"))
    vgp_inference = np.load(os.path.join(MODEL_DIR, "vgp_inference.npz"))

    VGP_MODEL.A = A_b_data["A"]
    VGP_MODEL.b = A_b_data["b"]
    VGP_MODEL.lambda_lagrange = lagrange_data["lambda_lagrange"]
    VGP_MODEL.psi_lagrange = lagrange_data["psi_lagrange"]
    VGP_MODEL.prior_sde.a = sde_params["a"][-1] * tf.ones_like(VGP_MODEL.prior_sde.a)
    VGP_MODEL.prior_sde.c = sde_params["c"][-1] * tf.ones_like(VGP_MODEL.prior_sde.c)
    VGP_MODEL.q_initial_mean = vgp_inference["m"][0] * tf.ones_like(VGP_MODEL.q_initial_mean)

    # cov without the noise variance
    fx_cov_0 = vgp_inference["S"][0] - NOISE_STDDEV**2
    VGP_MODEL.q_initial_cov = fx_cov_0 * tf.ones_like(VGP_MODEL.q_initial_cov)


def compare_elbo():
    """Compare ELBO between loaded moel and trained model"""
    loaded_model_elbo = VGP_MODEL.elbo()
    print(f"ELBO (Loaded model): {loaded_model_elbo}")

    trained_model_elbo = np.load(os.path.join(MODEL_DIR, "vgp_elbo.npz"))["elbo"][-1]
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
        VGP_MODEL.prior_sde = PriorDoubleWellSDE(q=true_q, initial_a_val=a, initial_c_val=c)
        ELBO_VALS.append(VGP_MODEL.elbo())

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
    plt.savefig(os.path.join(MODEL_DIR, "vgp_learning_bound.svg"))
    plt.show()

    np.savez(os.path.join(MODEL_DIR, "vgp_learning_bound.npz"), elbo=clipped_elbo_vals, a=A_VALUE_RANGE,
             c=C_VALUE_RANGE)


def plot_learning_plot():
    ssm_learning_path = os.path.join(MODEL_DIR, "vgp_learnt_sde.npz")
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

    plt.savefig(os.path.join(MODEL_DIR, "vgp_learning_bound_learning.svg"))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGP ELBO bound for DW process')

    parser.add_argument('-dir', '--data_dir', type=str, help='Directory of the saved VGP model', required=True)
    parser.add_argument('-dt', type=float, default=0., help='Modify dt for time-grid.')

    args = parser.parse_args()

    # setting wandb loggin to off
    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project="VI-SDE")

    MODEL_DIR = args.data_dir

    load_data("/".join(MODEL_DIR.split("/")[:-1]))

    if args.dt > 0:
        dt = args.dt
        t0 = TIME_GRID[0]
        t1 = TIME_GRID[-1]
        TIME_GRID = tf.cast(np.arange(t0, t1 + dt, dt), dtype=DTYPE).numpy()

    load_model()

    compare_elbo()

    calculate_elbo_bound()

    normalize_elbo_vals()

    plot_elbo_bound()

    plot_learning_plot()
