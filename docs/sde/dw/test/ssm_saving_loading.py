import argparse
import os
import tempfile

import numpy as np
import wandb
import tensorflow as tf
from gpflow import default_float
from gpflow.likelihoods import Gaussian
from gpflow.base import Parameter

from markovflow.sde.sde import PriorDoubleWellSDE
from markovflow.models.cvi_sde import SDESSM

DTYPE = default_float()

DIR = ""
Q = 1.
NOISE_STDDEV = 0.1 * np.ones((1, 1))
TIME_GRID = tf.zeros((1, 1))
OBSERVATION_DATA = ()
SSMMODEL_TRAINED = None
SSMMODEL_LOADED = None
SSMMODEL_LOCAL_LOADED = None
TMP_DIR = tempfile.gettempdir()


def load_data():
    global Q, NOISE_STDDEV, TIME_GRID, OBSERVATION_DATA
    data_path = os.path.join(DIR, "data.npz")
    data = np.load(data_path)
    Q = data["q"]
    NOISE_STDDEV = data["noise_stddev"]
    TIME_GRID = data["time_grid"]
    OBSERVATION_DATA = (data["observation_grid"], tf.squeeze(data["observation_vals"], axis=0).numpy())


def train_ssm_model():
    global SSMMODEL_TRAINED
    true_q = Q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_ssm = PriorDoubleWellSDE(q=true_q)
    likelihood_ssm = Gaussian(NOISE_STDDEV**2)

    SSMMODEL_TRAINED = SDESSM(input_data=OBSERVATION_DATA, prior_sde=prior_sde_ssm, grid=TIME_GRID,
                              likelihood=likelihood_ssm, learning_rate=1., prior_params_lr=0.01)
    SSMMODEL_TRAINED.update_sites()
    SSMMODEL_TRAINED.update_prior_sde()
    SSMMODEL_TRAINED._linearize_prior()


def save_ssm_model():
    np.savez(os.path.join(TMP_DIR, "ssm_data_sites.npz"), nat1=SSMMODEL_TRAINED.data_sites.nat1.numpy(),
             nat2=SSMMODEL_TRAINED.data_sites.nat2.numpy(), log_norm=SSMMODEL_TRAINED.data_sites.log_norm.numpy())

    np.savez(os.path.join(TMP_DIR, "ssm_sites.npz"), nat1=SSMMODEL_TRAINED.sites_nat1.numpy(),
             nat2=SSMMODEL_TRAINED.sites_nat2.numpy())

    m_ssm, S_ssm = SSMMODEL_TRAINED.dist_q.marginals
    m_ssm = m_ssm.numpy().reshape(-1)
    s_std_ssm = np.sqrt(S_ssm.numpy().reshape(-1) + NOISE_STDDEV**2)

    np.savez(os.path.join(TMP_DIR, "ssm_inference.npz"), m=m_ssm, S=tf.square(s_std_ssm))

    np.savez(os.path.join(TMP_DIR, "ssm_learnt_sde.npz"), a=SSMMODEL_TRAINED.prior_sde.a,
             c=SSMMODEL_TRAINED.prior_sde.c, fx_mus=SSMMODEL_TRAINED.fx_mus, fx_covs=SSMMODEL_TRAINED.fx_covs)


def load_saved_model():
    global SSMMODEL_LOCAL_LOADED
    true_q = Q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_ssm = PriorDoubleWellSDE(q=true_q)
    likelihood_ssm = Gaussian(NOISE_STDDEV ** 2)

    SSMMODEL_LOCAL_LOADED = SDESSM(input_data=OBSERVATION_DATA, prior_sde=prior_sde_ssm, grid=TIME_GRID,
                                   likelihood=likelihood_ssm, learning_rate=1., prior_params_lr=0.01)
    # Load trained model variables
    data_sites = np.load(os.path.join(TMP_DIR, "ssm_data_sites.npz"))
    SSMMODEL_LOCAL_LOADED.data_sites.nat1 = Parameter(data_sites["nat1"])
    SSMMODEL_LOCAL_LOADED.data_sites.nat2 = Parameter(data_sites["nat2"])
    SSMMODEL_LOCAL_LOADED.data_sites.log_norm = Parameter(data_sites["log_norm"])

    sites = np.load(os.path.join(TMP_DIR, "ssm_sites.npz"))
    SSMMODEL_LOCAL_LOADED.sites_nat1 = sites["nat1"]
    SSMMODEL_LOCAL_LOADED.sites_nat2 = sites["nat2"]

    ssm_learning_path = os.path.join(TMP_DIR, "ssm_learnt_sde.npz")
    ssm_learning = np.load(ssm_learning_path)
    SSMMODEL_LOCAL_LOADED.prior_sde.a = ssm_learning["a"] * tf.ones_like(SSMMODEL_LOCAL_LOADED.prior_sde.a)
    SSMMODEL_LOCAL_LOADED.prior_sde.c = ssm_learning["c"] * tf.ones_like(SSMMODEL_LOCAL_LOADED.prior_sde.c)

    SSMMODEL_LOCAL_LOADED.fx_mus = ssm_learning["fx_mus"].reshape(SSMMODEL_LOCAL_LOADED.fx_mus.shape) * tf.ones_like(SSMMODEL_LOCAL_LOADED.fx_mus)
    SSMMODEL_LOCAL_LOADED.fx_covs = ssm_learning["fx_covs"].reshape(SSMMODEL_LOCAL_LOADED.fx_covs.shape)

    SSMMODEL_LOCAL_LOADED._linearize_prior()


def load_model():
    global SSMMODEL_LOADED

    true_q = Q * tf.ones((1, 1), dtype=DTYPE)
    prior_sde_ssm = PriorDoubleWellSDE(q=true_q)
    likelihood_ssm = Gaussian(NOISE_STDDEV ** 2)

    SSMMODEL_LOADED = SDESSM(input_data=OBSERVATION_DATA, prior_sde=prior_sde_ssm, grid=TIME_GRID,
                             likelihood=likelihood_ssm)

    # Load trained model variables
    SSMMODEL_LOADED.data_sites.nat1 = SSMMODEL_TRAINED.data_sites.nat1
    SSMMODEL_LOADED.data_sites.nat2 = SSMMODEL_TRAINED.data_sites.nat2
    SSMMODEL_LOADED.data_sites.log_norm = SSMMODEL_TRAINED.data_sites.log_norm

    SSMMODEL_LOADED.sites_nat1 = SSMMODEL_TRAINED.sites_nat1
    SSMMODEL_LOADED.sites_nat2 = SSMMODEL_TRAINED.sites_nat2

    SSMMODEL_LOADED.fx_mus = tf.identity(SSMMODEL_TRAINED.fx_mus)
    SSMMODEL_LOADED.fx_covs = tf.identity(SSMMODEL_TRAINED.fx_covs)

    SSMMODEL_LOADED.prior_sde.a = tf.identity(SSMMODEL_TRAINED.prior_sde.a)
    SSMMODEL_LOADED.prior_sde.c = tf.identity(SSMMODEL_TRAINED.prior_sde.c)

    SSMMODEL_LOADED._linearize_prior()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test loading for trained model for OU process')

    parser.add_argument('-dir', '--data_dir', type=str, help='Directory of the saved models', required=True)
    args = parser.parse_args()

    os.environ['WANDB_MODE'] = 'offline'
    wandb.init()

    DIR = args.data_dir

    load_data()
    train_ssm_model()
    save_ssm_model()
    load_model()
    load_saved_model()

    print(f"Trained model ELBO : {SSMMODEL_TRAINED.classic_elbo()}")
    print(f"Loaded model ELBO : {SSMMODEL_LOADED.classic_elbo()}")
    print(f"Local Loaded model ELBO : {SSMMODEL_LOCAL_LOADED.classic_elbo()}")
