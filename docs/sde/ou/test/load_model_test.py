import os
import argparse

import tensorflow as tf
import wandb
import numpy as np
from gpflow import default_float
from gpflow.likelihoods import Gaussian
from gpflow.base import Parameter

from markovflow.sde.sde import PriorOUSDE
from markovflow.models.cvi_sde import SDESSM
from markovflow.models.vi_sde import VariationalMarkovGP

DTYPE = default_float()

MODEL_DIR = ""
Q = 1.
NOISE_STDDEV = 0.1 * np.ones((1, 1))
TIME_GRID = tf.zeros((1, 1))
OBSERVATION_DATA = ()
SDESSM_MODEL = None
VGP_MODEL = None
ELBO_VALS = []


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
    global SDESSM_MODEL, VGP_MODEL

    true_q = Q * tf.ones((1, 1), dtype=DTYPE)
    likelihood = Gaussian(NOISE_STDDEV ** 2)

    prior_sde = PriorOUSDE(q=true_q)

    # model
    SDESSM_MODEL = SDESSM(input_data=OBSERVATION_DATA, prior_sde=prior_sde, grid=TIME_GRID, likelihood=likelihood)

    # Load trained model variables
    data_sites = np.load(os.path.join(MODEL_DIR, "ssm_data_sites.npz"))
    SDESSM_MODEL.data_sites.nat1 = Parameter(data_sites["nat1"])
    SDESSM_MODEL.data_sites.nat2 = Parameter(data_sites["nat2"])
    SDESSM_MODEL.data_sites.log_norm = Parameter(data_sites["log_norm"])

    # sites = np.load(os.path.join(MODEL_DIR, "ssm_sites.npz"))
    # SDESSM_MODEL.sites_nat1 = sites["nat1"]
    # SDESSM_MODEL.sites_nat2 = sites["nat2"]

    q_path = np.load(os.path.join(MODEL_DIR, "ssm_inference.npz"))
    SDESSM_MODEL.fx_mus = q_path["m"].reshape(SDESSM_MODEL.fx_mus.shape) * tf.ones_like(SDESSM_MODEL.fx_mus)
    # cov without the noise variance
    fx_covs = q_path["S"] - NOISE_STDDEV**2
    SDESSM_MODEL.fx_covs = fx_covs.reshape(SDESSM_MODEL.fx_covs.shape)

    ssm_learning_path = os.path.join(MODEL_DIR, "ssm_learnt_sde.npz")
    ssm_learning = np.load(ssm_learning_path)
    SDESSM_MODEL.prior_sde.decay = ssm_learning["decay"][-1] * tf.ones_like(SDESSM_MODEL.prior_sde.decay)

    SDESSM_MODEL.initial_chol_cov = tf.linalg.cholesky(tf.reshape(Q / (2 * -1 * SDESSM_MODEL.prior_sde.decay),
                                                                  SDESSM_MODEL.initial_chol_cov.shape))

    SDESSM_MODEL._linearize_prior()

    # VGP
    likelihood = Gaussian(NOISE_STDDEV ** 2)

    prior_sde = PriorOUSDE(q=true_q)
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
    VGP_MODEL.q_initial_mean = vgp_inference["m"][0] * tf.ones_like(VGP_MODEL.q_initial_mean)
    VGP_MODEL.prior_sde.decay = sde_params["decay"][-1] * tf.ones_like(VGP_MODEL.prior_sde.decay)

    # cov without the noise variance
    fx_cov_0 = vgp_inference["S"][0] - NOISE_STDDEV ** 2
    VGP_MODEL.q_initial_cov = fx_cov_0 * tf.ones_like(VGP_MODEL.q_initial_cov)

    VGP_MODEL.p_initial_cov = tf.reshape(Q / (2 * -1 * VGP_MODEL.prior_sde.decay),
                                         VGP_MODEL.p_initial_cov.shape)


def compare_elbo():
    """Compare ELBO between loaded moel and trained model"""
    loaded_model_elbo = SDESSM_MODEL.classic_elbo()
    print(f"SSM ELBO (Loaded model): {loaded_model_elbo}")

    trained_model_elbo = np.load(os.path.join(MODEL_DIR, "ssm_elbo.npz"))["elbo"][-1]
    print(f"SSM ELBO (Trained model) : {trained_model_elbo}")

    np.testing.assert_array_almost_equal(trained_model_elbo, loaded_model_elbo)

    loaded_model_elbo = VGP_MODEL.elbo()
    print(f"VGP ELBO (Loaded model): {loaded_model_elbo}")

    trained_model_elbo = np.load(os.path.join(MODEL_DIR, "vgp_elbo.npz"))["elbo"][-1]
    print(f"VGP ELBO (Trained model) : {trained_model_elbo}")

    np.testing.assert_array_almost_equal(trained_model_elbo, loaded_model_elbo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test loading for trained model for OU process')

    parser.add_argument('-dir', '--data_dir', type=str, help='Directory of the saved models', required=True)
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
