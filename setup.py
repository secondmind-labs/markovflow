# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['markovflow',
 'markovflow.kernels',
 'markovflow.likelihoods',
 'markovflow.models']

package_data = \
{'': ['*']}

install_requires = \
['banded_matrices>=0.32.1,<0.33.0',
 'google-auth==1.16.0',
 'gpflow>=2.0.5,<3.0.0',
 'importlib_metadata>=1.6,<2.0',
 'numpy>=1.18.0,<2.0.0',
 'scipy==1.4.1',
 'setuptools>=41.0.0,<42.0.0',
 'tensorflow-probability==0.10.1',
 'tensorflow==2.2.1']

setup_kwargs = {
    'name': 'markovflow',
    'version': '0.33.0',
    'description': 'A Tensorflow based library for Time Series Modelling with Gaussian Processes',
    'long_description': None,
    'author': 'Markovflow Team',
    'author_email': 'markovflow@prowler.io',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7,<4.0',
}


setup(**setup_kwargs)
