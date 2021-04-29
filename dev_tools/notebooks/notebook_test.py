# Copyright 2020 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ========================== CONTINUOUS NOTEBOOK TESTS ============================================
#
# These tests are run for all of our notebooks against the current branch. It is assumed that
# notebooks will not install cirq in case cirq is on the path. The simple `import cirq` path is the
# main focus and it is executed in a shared virtual environment for the notebooks. Thus, these
# tests ensure that notebooks are still working with the latest version of cirq.

# In order to make these test run fast, we leverage papermills ability to modify parameters.
# In particular one can create a single cell (not multiple) which has the tag "parameters"
# This cell will be run with the variables defined in this cell replaced by the values in
# a yaml file of the same name as the notebook.

import os

import pytest

from dev_tools import shell_tools
from dev_tools.notebooks import filter_notebooks, list_all_notebooks

SKIP_NOTEBOOKS = [
    # skipping vendor notebooks as we don't have auth sorted out
    "**/aqt/*.ipynb",
    "**/ionq/*.ipynb",
    "**/google/*.ipynb",
    "**/pasqal/*.ipynb",
    # skipping fidelity estimation due to
    # https://github.com/quantumlib/Cirq/issues/3502
    "examples/*fidelity*",
    # chemistry.ipynb requires openfermion, that installs cirq 0.9.1, which interferes
    # with testing cirq itself...
    'docs/tutorials/educators/chemistry.ipynb',
]


@pytest.mark.slow
@pytest.mark.parametrize("notebook_path", filter_notebooks(list_all_notebooks(), SKIP_NOTEBOOKS))
def test_notebooks_against_released_cirq(notebook_path):
    notebook_file = os.path.basename(notebook_path)
    notebook_rel_dir = os.path.dirname(os.path.relpath(notebook_path, "."))
    notebook_test_yaml_file = os.path.splitext(notebook_path)[0] + '.yaml'
    yaml_flag = f'-f {notebook_test_yaml_file}' if os.path.exists(notebook_test_yaml_file) else ''
    out_path = f"out/{notebook_rel_dir}/{notebook_file[:-6]}.out.ipynb"
    cmd = f"""mkdir -p out/{notebook_rel_dir}
papermill {notebook_path} {out_path} {yaml_flag}"""
    _, stderr, status = shell_tools.run_shell(
        cmd=cmd,
        log_run_to_stderr=False,
        raise_on_fail=False,
        out=shell_tools.TeeCapture(),
        err=shell_tools.TeeCapture(),
    )

    if status != 0:
        print(stderr)
        pytest.fail(
            f"Notebook failure: {notebook_file}, please see {out_path} for the output "
            f"notebook (in Github Actions, you can download it from the workflow artifact"
            f" 'notebook-outputs')"
        )
