"""Plot data from npz files"""

import os

import tensorflow as tf
from gpflow.config import default_float
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from markovflow.sde.sde import PriorDoubleWellSDE, DoubleWellSDE
from docs.sde.sde_exp_utils import plot_posterior, plot_observations

DTYPE = default_float()


def plot_drift(data_dir: str, q: tf.Tensor, latent_process: tf.Tensor, time_grid: tf.Tensor,
               observation_grid: tf.Tensor, dt: float):
    ssm_path = os.path.join(data_dir, "ssm_learnt_sde.npz")
    vgp_path = os.path.join(data_dir, "vgp_learnt_sde.npz")

    ssm_data = np.load(ssm_path)
    ssm_a = ssm_data["a"][-1]
    ssm_c = ssm_data["c"][-1]
    ssm_sde = PriorDoubleWellSDE(q=q, initial_a_val=ssm_a, initial_c_val=ssm_c)

    vgp_data = np.load(vgp_path)
    vgp_a = vgp_data["a"][-1]
    vgp_c = vgp_data["c"][-1]
    vgp_sde = PriorDoubleWellSDE(q=q, initial_a_val=vgp_a, initial_c_val=vgp_c)

    true_dw_sde = DoubleWellSDE(q)

    x = np.linspace(-2, 2, 40).reshape((-1, 1))

    true_drift = true_dw_sde.drift(x, None)
    sde_ssm_learnt_drift = ssm_sde.drift(x, None)
    vgp_learnt_drift = vgp_sde.drift(x, None)

    plt.subplots(1, 1, figsize=(5, 5))

    d = (latent_process[1:] - latent_process[:-1]) / dt

    indices = tf.where(tf.equal(time_grid, observation_grid))[:, 1][..., None]
    latent_obs = tf.gather(latent_process, indices)
    latent_obs_d = tf.gather(d, indices)

    plt.clf()

    plt.scatter(latent_obs, latent_obs_d, alpha=0.2, c="black")
    plt.plot(x, sde_ssm_learnt_drift, label="SDE-SSM", color="blue")
    plt.plot(x, vgp_learnt_drift, label="VGP", color="green")
    plt.plot(x, true_drift, label="True drift", color="black")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])

    plt.title("Drift")
    plt.legend()
    # plt.savefig(os.path.join(plot_save_dir, "drift.svg"))
    plt.show()

    print(f"SSM learnt drift : f(x) = {ssm_a} * x * ({ssm_c} - x^2)")
    print(f"VGP learnt drift : f(x) = {vgp_a} * x * ({vgp_c} - x^2)")


def plot_posteriors(data_dir: str, time_grid: np.ndarray, observation_grid: np.ndarray, observation_vals: np.ndarray):
    ssm_path = os.path.join(data_dir, "ssm_inference.npz")
    vgp_path = os.path.join(data_dir, "vgp_inference.npz")

    ssm_data = np.load(ssm_path)
    ssm_m = ssm_data["m"]
    ssm_S = ssm_data["S"]

    vgp_data = np.load(vgp_path)
    vgp_m = vgp_data["m"]
    vgp_S = vgp_data["S"]

    dt = 0.001
    time_grid = tf.cast(np.arange(time_grid[0], time_grid[-1] + dt, dt), dtype=DTYPE).numpy()

    plt.rcParams["figure.figsize"] = [15, 5]
    plt.clf()
    plot_observations(observation_grid, observation_vals)
    plot_posterior(ssm_m, np.sqrt(ssm_S), time_grid, "SDE-SSM")
    plot_posterior(vgp_m, np.sqrt(vgp_S), time_grid, "VGP")
    plt.legend()

    # plt.savefig(os.path.join(data_dir, "posterior.svg"))

    plt.show()


def plot_elbo_bound(data_dir: str):
    ssm_path = os.path.join(data_dir, "ssm_learning_bound.npz")
    ssm_data = np.load(ssm_path)
    a = ssm_data["a"]
    c = ssm_data["c"]
    elbo = ssm_data["elbo"]

    plt.clf()
    plt.subplots(1, 1, figsize=(5, 5))

    c = plt.pcolormesh(a[:, 0], c[0], elbo, vmin=np.min(elbo), vmax=np.max(elbo), shading='auto')
    plt.colorbar(c)
    plt.savefig(os.path.join(data_dir, "ssm_learning_bound.svg"))
    plt.show()


def plot_elbo_bound_3D(data_dir: str):

    ssm_path = os.path.join(data_dir, "ssm_learning_bound.npz")
    ssm_data = np.load(ssm_path)
    a = ssm_data["a"]
    c = ssm_data["c"]
    elbo = ssm_data["elbo"]

    plt.clf()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(a, c, elbo, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(np.min(elbo), np.max(elbo))
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(os.path.join(data_dir, "ssm_learning_bound.svg"))
    plt.show()

    vgp_path = os.path.join(data_dir, "vgp_learning_bound.npz")
    vgp_data = np.load(vgp_path)
    a = vgp_data["a"]
    c = vgp_data["c"]
    elbo = vgp_data["elbo"]

    plt.clf()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(a, c, elbo, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(np.min(elbo), np.max(elbo))
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(os.path.join(data_dir, "ssm_learning_bound.svg"))
    plt.show()


def plot_elbo_bound_learning(data_dir: str):
    ssm_path = os.path.join(data_dir, "ssm_learning_bound.npz")
    ssm_data = np.load(ssm_path)
    a = ssm_data["a"]
    c = ssm_data["c"]
    elbo = ssm_data["elbo"]

    ssm_learning_path = os.path.join(data_dir, "ssm_learnt_sde.npz")
    ssm_learning = np.load(ssm_learning_path)
    ssm_learnt_a = ssm_learning["a"]
    ssm_learnt_c = ssm_learning["c"]

    plt.clf()
    plt.subplots(1, 1, figsize=(5, 5))

    c = plt.pcolormesh(a[:, 0], c[0], elbo, vmin=np.min(elbo), vmax=np.max(elbo), shading='auto')
    plt.colorbar(c)
    plt.plot(ssm_learnt_a, ssm_learnt_c, "x")
    plt.savefig(os.path.join(data_dir, "ssm_learning_bound.svg"))
    plt.show()

    vgp_path = os.path.join(data_dir, "vgp_learning_bound.npz")
    vgp_data = np.load(vgp_path)
    a = vgp_data["a"]
    c = vgp_data["c"]
    elbo = vgp_data["elbo"]
    vgp_learning_path = os.path.join(data_dir, "vgp_learnt_sde.npz")
    vgp_learning = np.load(vgp_learning_path)
    vgp_learnt_a = vgp_learning["a"]
    vgp_learnt_c = vgp_learning["c"]

    plt.clf()
    plt.subplots(1, 1, figsize=(5, 5))

    c = plt.pcolormesh(a[:, 0], c[0], elbo, vmin=np.min(elbo), vmax=np.max(elbo), shading='auto')
    plt.colorbar(c)
    plt.plot(vgp_learnt_a, vgp_learnt_c, "x")
    plt.savefig(os.path.join(data_dir, "vgp_learning_bound.svg"))
    plt.show()


if __name__ == '__main__':
    data_dir = "data/128"

    data_path = os.path.join(data_dir, "data.npz")
    data = np.load(data_path)

    q = data["q"]
    q = q * tf.ones((1, 1), dtype=DTYPE)

    latent_process = data["latent_process"].reshape((-1, 1))
    observation_grid = tf.convert_to_tensor(data["observation_grid"].reshape((-1, 1)))
    observation_vals = data["observation_vals"].reshape((-1, 1))

    time_grid = data["time_grid"]
    dt = time_grid[1] - time_grid[0]

    learning_dir = os.path.join(data_dir, "learning")

    # plot_elbo_bound_3D(learning_dir)
    plot_elbo_bound_learning(learning_dir)
    # plot_elbo_bound(learning_dir)
    # plot_posteriors(learning_dir, time_grid, observation_grid.numpy(), observation_vals)

    # plot_drift(learning_dir, q, latent_process, time_grid, observation_grid, dt)
