# +
import numpy as np
import tensorflow as tf
from gpflow import set_trainable
from gpflow.ci_utils import ci_niter
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from matplotlib import pyplot as plt
from markovflow.kernels import Matern32
from markovflow.models import SparseSpatioTemporalVariational
from markovflow.ssm_natgrad import SSMNaturalGradient

np.random.seed(10)
# -

# Declaring the model 

# +
M_time = 7
M_space = 4

kernel_space = RBF(variance=1.0, lengthscales=0.2)
kernel_time = Matern32(variance=1.0, lengthscale=0.2)
likelihood = Gaussian(variance=0.1)

inducing_space = np.linspace(0.1, 0.9, M_space).reshape(-1, 1)
inducing_time = np.linspace(0, 1, M_time).reshape(-1, )

model = SparseSpatioTemporalVariational(
    inducing_time=tf.identity(inducing_time),
    inducing_space=tf.identity(inducing_space),
    kernel_space=kernel_space,
    kernel_time=kernel_time,
    likelihood=likelihood,
)
# -

# Creating data
num_data = 200
time_points = np.random.rand(num_data, 1)
space_points = np.random.rand(num_data, 1)
X = np.concatenate([space_points, time_points], -1)
f = lambda v: np.cos(5.0 * (v[..., 1:] + v[..., :1]))
F = f(X)
Y = F + np.random.randn(num_data, 1)
data = (X, Y)

# Creating a plotting grid and plotting function

# +
x_grid, t_grid = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
X_grid = np.concatenate([x_grid.reshape(-1, 1), t_grid.reshape(-1, 1)], axis=-1)


def plot_model(model):
    mu_f, var_f = model.space_time_predict_f(X_grid)
    fig, axarr = plt.subplots(2, 1)
    axarr[0].scatter(x=space_points, y=time_points, c=Y)
    axarr[1].scatter(x=X_grid[..., :1], y=X_grid[..., 1:], c=mu_f.numpy())

    for ax in axarr:
        ax.hlines(model.inducing_space, xmin=time_points.min(), xmax=time_points.max(), colors="r")
        ax.vlines(
            model.inducing_space, ymin=space_points.min(), ymax=space_points.max(), colors="k"
        )

    plt.savefig("spatio_temporal.pdf", dpi=300)
    plt.show()


# -

# Training

# +
# Start at a small learning rate
adam_learning_rate = 0.0001
natgrad_learning_rate = 0.5

adam_opt = tf.optimizers.Adam(learning_rate=adam_learning_rate)
natgrad_opt = SSMNaturalGradient(gamma=natgrad_learning_rate, momentum=False)

set_trainable(model.ssm_q, False)
adam_var_list = model.trainable_variables  # trainable_variables
set_trainable(model.ssm_q, True)


# +
# tf.function
def loss(input_data):
    return -model.elbo(input_data)


# tf.function
def opt_step(input_data):
    natgrad_opt.minimize(lambda: loss(input_data), model.ssm_q)
    adam_opt.minimize(lambda: loss(input_data), adam_var_list)


# +
max_iter = ci_niter(500)

for i in range(max_iter):
    opt_step(data)
    if i % 20 == 0:
        plot_model(model)
        print("Iteration:", i, ", Loss:", model.loss(data).numpy())
