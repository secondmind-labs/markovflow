.. Copyright 2021 The Markovflow Contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

..
   Note: Items in the toctree form the top-level navigation.

.. toctree::
   :hidden:

   Markovflow <self>
   Tutorials <tutorials>
   API Reference <autoapi/markovflow/index>


Welcome to Markovflow
=====================

Markovflow is a research toolbox dedicated to Markovian Gaussian processes.

Markovflow uses the mathematical building blocks from `GPflow <http://www.gpflow.org/>`_ :cite:p:`gpflow2020` and marries these with the powerful linear algebra routine for banded matrices provided by `banded_matrices <https://github.com/Secondmind-labs/banded_matrices>`_.
This combination leads to a framework that can be used for:

- researching Markovian Gaussian process models (e.g., :cite:p:`durrande2019banded, adam2020doubly, wilkinson2021sparse`), and
- building, training, evaluating and deploying Gaussian processes in a modern way, making use of the tools developed by the deep learning community.


Getting started
---------------

We have provided multiple `Tutorials <tutorials>` showing the basic functionality of the toolbox, and have a comprehensive `API Reference <autoapi/markovflow/index>`.

As a quick teaser, here's a snippet that demonstrates how to perform classic GP regression for a simple one-dimensional dataset:


.. code-block:: python

    # Create a GPR model
    kernel = Matern32(lengthscale=8.0, variance=1.0)
    observation_covariance = tf.constant([[0.0001]], dtype=FLOAT_TYPE)
    input_data = (tf.constant(time_points), tf.constant(norm_observations))
    gpr = GaussianProcessRegression(input_data=input_data, kernel=kernel,
                                chol_obs_covariance=tf.linalg.cholesky(observation_covariance))

    # Compile and fit
    opt = tf.optimizers.Adam()
    for _ in range(100):
        opt.minimize(gpr.loss, gpr.trainable_variables)
    print(gpr.log_likelihood())


Installation
------------

Latest release from PyPI
^^^^^^^^^^^^^^^^^^^^^^^^

To install Markovflow using the latest release from PyPI, run

.. code::

   $ pip install markovflow

The library supports Python 3.7 onwards, and uses `semantic versioning <https://semver.org/>`_.

Latest development release from GitHub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a check-out of the develop branch of the `Markovflow GitHub repository <https://github.com/secondmind-labs/markovflow>`_, run

.. code::

   $ pip install -e .


Join the community
------------------

Markovflow is an open source project. We welcome contributions. To submit a pull request, file a bug report, or make a feature request, see the `contribution guidelines <https://github.com/secondmind-labs/markovflow/blob/develop/CONTRIBUTING.md>`_.

We have a public `Slack workspace <https://join.slack.com/t/secondmind-labs/shared_invite/zt-ph07nuie-gMlkle__tjvXBay4FNSLkw>`_. Please use this invite link if you'd like to join, whether to ask short informal questions or to be involved in the discussion and future development of Markovflow.


Bibliography
------------

.. bibliography::
   :all: