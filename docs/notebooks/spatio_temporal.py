# +
import numpy as np
import tensorflow as tf
from gpflow import set_trainable
from gpflow.ci_utils import ci_niter
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from matplotlib import pyplot as plt
from markovflow.kernels import Matern32
from markovflow.models import SpatioTemporalSparseCVI
from markovflow.ssm_natgrad import SSMNaturalGradient

np.random.seed(10)
# -

# Declaring the model

# +
M_time = 7
M_space = 7
T = 5.0

kernel_space = RBF(variance=1.0, lengthscales=0.5)
kernel_time = Matern32(variance=1.0, lengthscale=T / 2.0)
likelihood = Gaussian(variance=0.1)

inducing_space = np.linspace(0.1, 0.9, M_space).reshape(-1, 1)
inducing_time = np.linspace(0, T, M_time).reshape(-1,)

model = SpatioTemporalSparseCVI(
    inducing_time=tf.identity(inducing_time),
    inducing_space=tf.identity(inducing_space),
    kernel_space=kernel_space,
    kernel_time=kernel_time,
    likelihood=likelihood,
)
# -

# Creating data
num_data = 500
std_noise = 0.5
time_points = np.random.uniform(0, T, num_data).reshape(-1, 1)
space_points = np.random.rand(num_data, 1)
X = np.concatenate([space_points, time_points], -1)
f = lambda v: np.cos(5.0 * (v[..., 1:] + v[..., :1]))
F = f(X)
Y = F + np.random.randn(num_data, 1) * std_noise
data = (X, Y)

# Creating a plotting grid and plotting function

# +
x_grid, t_grid = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, T, 50))
X_grid = np.concatenate([x_grid.reshape(-1, 1), t_grid.reshape(-1, 1)], axis=-1)


def plot_model(model):
    mu_f, var_f = model.space_time_predict_f(X_grid)
    fig, axarr = plt.subplots(2, 1)
    axarr[0].scatter(x=time_points, y=space_points, c=Y)
    axarr[1].scatter(x=X_grid[..., 1:], y=X_grid[..., :1], c=mu_f.numpy())

    axarr[1].hlines(
        model._inducing_space, xmin=time_points.min(), xmax=time_points.max(), colors="r"
    )
    axarr[1].vlines(
        model._inducing_time, ymin=space_points.min(), ymax=space_points.max(), colors="k"
    )

    plt.savefig("spatio_temporal.png", dpi=300)
    plt.show()


# -

# Training

# +
# Start at a small learning rate
adam_learning_rate = 0.05
natgrad_learning_rate = 0.5

adam_opt = tf.optimizers.Adam(learning_rate=adam_learning_rate)
natgrad_opt = SSMNaturalGradient(gamma=natgrad_learning_rate, momentum=False)

set_trainable(model.nat2, False)
set_trainable(model.nat1, False)

adam_var_list = model.kernel.trainable_variables  # trainable_variables
print(adam_var_list)
set_trainable(model.nat2, True)
set_trainable(model.nat1, True)

# +
@tf.function
def loss(input_data):
    return model.loss(input_data)


@tf.function
def opt_step(input_data):
    model.update_sites(input_data)
    adam_opt.minimize(lambda: loss(input_data), adam_var_list)


# +
max_iter = ci_niter(500)

for i in range(max_iter):
    opt_step(data)
    if i % 20 == 0:
        plot_model(model)
        print(model.kernel.kernel_time.lengthscale)
        print("Iteration:", i, ", Loss:", model.loss(data).numpy())
