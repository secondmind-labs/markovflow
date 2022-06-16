"""GPR for the DoubleWell process"""

import os

import tensorflow as tf
import gpflow
import numpy as np
import matplotlib.pyplot as plt

from markovflow.sde.sde import DoubleWellSDE, PriorDoubleWellSDE
from docs.sde.sde_exp_utils import plot_observations


if __name__ == '__main__':
    DTYPE = gpflow.config.default_float()
    plt.rcParams["figure.figsize"] = [15, 5]

    """
    Parameters
    """
    data_dir = "data/91"

    """
    Generate observations for a linear SDE
    """
    data_path = os.path.join(data_dir, "data.npz")
    data = np.load(data_path)
    q = data["q"]
    noise_stddev = data["noise_stddev"]
    x0 = data["x0"]
    observation_vals = data["observation_vals"]
    observation_grid = data["observation_grid"]
    latent_process = data["latent_process"]
    time_grid = data["time_grid"]
    t0 = time_grid[0]
    t1 = time_grid[-1]

    dt = time_grid[1] - time_grid[0]

    true_dw_sde = DoubleWellSDE(q=q * tf.ones((1, 1), dtype=DTYPE))

    plt.clf()
    plot_observations(observation_grid, observation_vals)
    plt.plot(time_grid, tf.reshape(latent_process, (-1)), label="Latent Process", alpha=0.2, color="gray")
    plt.xlabel("Time (t)")
    plt.ylabel("y(t)")
    plt.ylim([-2, 2])
    plt.xlim([t0, t1])
    plt.title("Observations")
    plt.legend()
    plt.show()

    """
    Create data for GPR
    """
    y = tf.squeeze(observation_vals, axis=0)
    obs_dt = observation_grid[1] - observation_grid[0]  # As grid is homogeneous
    input_data = (y[:-1], (y[1:] - y[:-1])/obs_dt)

    """
    GPR model
    """
    # kernel = gpflow.kernels.linears.Linear(10) + gpflow.kernels.RBF(lengthscales=1, variance=20)
    kernel = gpflow.kernels.linears.Linear() + gpflow.kernels.RBF()
    model = gpflow.models.GPR(data=input_data, kernel=kernel, noise_variance=q*obs_dt)

    opt = gpflow.optimizers.Scipy()

    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=1000))

    """
    Prior SDE
    """
    prior_sde = PriorDoubleWellSDE(q=tf.ones((1, 1), dtype=DTYPE) * q)

    """
    SDE-SSM
    """
    ssm_path = os.path.join("data/91/learning_orig_dt/ssm_learnt_sde.npz")

    ssm_data = np.load(ssm_path)
    ssm_a = ssm_data["a"][-1]
    ssm_c = ssm_data["c"][-1]
    ssm_sde = PriorDoubleWellSDE(q=tf.ones((1, 1), dtype=DTYPE) * q, initial_a_val=ssm_a, initial_c_val=ssm_c)

    """
    VGP
    """
    vgp_path = os.path.join("data/91/learning_orig_dt/vgp_learnt_sde.npz")

    vgp_data = np.load(vgp_path)
    vgp_a = vgp_data["a"][-1]
    vgp_c = vgp_data["c"][-1]
    vgp_sde = PriorDoubleWellSDE(q=tf.ones((1, 1), dtype=DTYPE) * q, initial_a_val=vgp_a, initial_c_val=vgp_c)

    """
    Plotting
    """
    xx = np.linspace(-2, 2, 100).reshape(100, 1)  # test points must be of shape (N, D)

    true_drift = true_dw_sde.drift(xx, None)
    ssm_drift = ssm_sde.drift(xx, None)
    vgp_drift = vgp_sde.drift(xx, None)
    prior_drift = prior_sde.drift(xx, None)

    mean, var = model.predict_f(xx)

    plt.figure(figsize=(15, 10))
    # plt.plot(input_data[0], input_data[1], "kx", alpha=0.5)
    plt.plot(xx, mean, label="GPR", linestyle="dashed", alpha=0.5)
    plt.plot(xx, true_drift, label="True", linestyle="dashed", color="black", alpha=0.5)
    plt.plot(xx, ssm_drift, label="SDE-SSM")
    plt.plot(xx, vgp_drift, label="VGP")
    plt.plot(xx, prior_drift, label="Prior drift")
    # plt.fill_between(
    #     xx[:, 0],
    #     mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    #     mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    #     color="C0",
    #     alpha=0.2,
    # )
    plt.xlim(-2, 2)
    plt.ylim(-4, 4)

    plt.legend()
    plt.show()
