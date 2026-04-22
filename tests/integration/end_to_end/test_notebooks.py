#
# Copyright (c) 2021 The Markovflow Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import glob
import os
import sys
import traceback
from contextlib import contextmanager
from typing import List

import jupytext
import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

# To blacklist a notebook, add its full base name (including .ipynb extension,
# but without any directory component). If there are several notebooks in
# different directories with the same base name, they will all get blacklisted
# (change the blacklisting check to something else in that case, if need be!)
BLACKLISTED_NOTEBOOKS: List[str] = []

NOTEBOOK_TEST_ENVIRONMENT = {"CI": "true", "MPLBACKEND": "Agg"}

NOTEBOOK_TEST_SETUP = """
import tensorflow as tf


def _markovflow_notebook_minimize(optimizer, loss_fn, var_list=None, **kwargs):
    if var_list is None:
        var_list = kwargs.pop("var_list", None)
    if var_list is None and kwargs:
        var_list = kwargs.pop("variables", None)
    if var_list is None:
        raise TypeError("var_list or variables must be provided")

    with tf.GradientTape() as tape:
        loss = loss_fn() if callable(loss_fn) else loss_fn
    gradients = tape.gradient(loss, var_list)
    optimizer.apply_gradients(
        (gradient, variable)
        for gradient, variable in zip(gradients, var_list)
        if gradient is not None
    )
    return loss


if not hasattr(tf.optimizers.Optimizer, "minimize"):
    tf.optimizers.Optimizer.minimize = _markovflow_notebook_minimize
"""


@contextmanager
def _notebook_test_environment():
    old_values = {key: os.environ.get(key) for key in NOTEBOOK_TEST_ENVIRONMENT}
    os.environ.update(NOTEBOOK_TEST_ENVIRONMENT)
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _nbpath():
    this_dir = os.path.dirname(__file__)
    return os.path.join(this_dir, "../../../docs/notebooks/")


def test_notebook_dir_exists():
    assert os.path.isdir(_nbpath())


def get_notebooks():
    """
    Returns all notebooks in `_nbpath` that are not blacklisted.
    """

    def notebook_blacklisted(nb):
        blacklisted_notebooks_basename = map(os.path.basename, BLACKLISTED_NOTEBOOKS)
        return os.path.basename(nb) in blacklisted_notebooks_basename

    # recursively traverse the notebook directory in search for ipython notebooks
    all_notebooks = glob.iglob(os.path.join(_nbpath(), "**", "*.py"), recursive=True)
    notebooks_to_test = [nb for nb in all_notebooks if not notebook_blacklisted(nb)]
    return notebooks_to_test


def _preproc():
    pythonkernel = "python" + str(sys.version_info[0])
    return ExecutePreprocessor(timeout=300, kernel_name=pythonkernel, interrupt_on_timeout=True)


def _exec_notebook(notebook_filename):
    with open(notebook_filename) as notebook_file:
        nb = jupytext.read(notebook_file, as_version=nbformat.current_nbformat)
        nb.cells.insert(0, nbformat.v4.new_code_cell(NOTEBOOK_TEST_SETUP))
        try:
            meta_data = {"path": os.path.dirname(notebook_filename)}
            with _notebook_test_environment():
                _preproc().preprocess(nb, {"metadata": meta_data})
        except CellExecutionError as cell_error:
            traceback.print_exc(file=sys.stdout)
            msg = "Error executing the notebook {0}. See above for error.\nCell error: {1}"
            pytest.fail(msg.format(notebook_filename, str(cell_error)))


@pytest.mark.notebooks
@pytest.mark.parametrize("notebook_file", get_notebooks())
def test_notebook(notebook_file):
    _exec_notebook(notebook_file)


def test_has_notebooks():
    assert len(get_notebooks()) >= 2, "there are probably some notebooks that were not discovered"
