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
['banded-matrices==0.0.6',
 'google-auth==1.16.0',
 'gpflow>=2.0.5,<3.0.0',
 'importlib_metadata>=1.6,<2.0',
 'numpy>=1.18.0,<2.0.0',
 'scipy==1.4.1',
 'setuptools>=41.0.0,<42.0.0',
 'tensorflow-probability==0.11.0',
 'tensorflow==2.2.1']

with open("VERSION") as file:
    version = file.read().strip()

with open("README.md") as file:
    long_description = file.read()

setup_kwargs = {
    'name': 'markovflow',
    "version": version,
    "long_description": None,
    "long_description_content_type": None,
    'description': 'A Tensorflow based library for Time Series Modelling with Gaussian Processes',
    'author': 'Markovflow Team',
    'author_email': 'markovflow@secondmind.ai',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7,<4.0',
}


setup(**setup_kwargs)
