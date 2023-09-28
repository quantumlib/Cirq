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

import importlib.metadata
import os
import tempfile

import pytest

from dev_tools import shell_tools
from dev_tools.notebooks import filter_notebooks, list_all_notebooks, rewrite_notebook

SKIP_NOTEBOOKS = [
    # skipping vendor notebooks as we don't have auth sorted out
    '**/aqt/*.ipynb',
    '**/azure-quantum/*.ipynb',
    '**/ionq/*.ipynb',
    '**/pasqal/*.ipynb',
    '**/rigetti/*.ipynb',
    # skipp cirq-ft notebooks since they are included in individual tests
    'cirq-ft/**',
    # skipping fidelity estimation due to
    # https://github.com/quantumlib/Cirq/issues/3502
    'examples/*fidelity*',
    # skipping quantum utility simulation (too large)
    'examples/advanced/*quantum_utility*',
    # tutorials that use QCS and arent skipped due to one or more cleared output cells
    'docs/tutorials/google/identifying_hardware_changes.ipynb',
    'docs/tutorials/google/echoes.ipynb',
    'docs/noise/qcvv/xeb_calibration_example.ipynb',
    'docs/noise/calibration_api.ipynb',
    'docs/noise/floquet_calibration_example.ipynb',
    # temporary: need to fix QVM metrics and device spec
    'docs/tutorials/google/spin_echoes.ipynb',
    'docs/tutorials/google/visualizing_calibration_metrics.ipynb',
    # shouldn't have outputs generated for style reasons
    'docs/simulate/qvm_builder_code.ipynb',
]


@pytest.fixture
def require_packages_not_changed():
    """Verify notebook test does not change packages in the Python test environment.

    Raise AssertionError if the pre-existing set of Python packages changes in any way.
    """
    packages_before = set((d.name, d.version) for d in importlib.metadata.distributions())
    yield
    packages_after = set((d.name, d.version) for d in importlib.metadata.distributions())
    assert packages_after == packages_before


@pytest.fixture
def env_with_temporary_pip_target():
    """Setup system environment that tells pip to install packages to a temporary directory."""
    with tempfile.TemporaryDirectory(suffix='-notebook-site-packages') as tmpdirname:
        # Note: We need to append tmpdirname to the PYTHONPATH, because PYTHONPATH may
        # already point to the development sources of Cirq (as happens with check/pytest).
        # Should some notebook pip-install a stable version of Cirq to tmpdirname,
        # it would appear in PYTHONPATH after the development Cirq.
        pythonpath = (
            f'{os.environ["PYTHONPATH"]}{os.pathsep}{tmpdirname}'
            if 'PYTHONPATH' in os.environ
            else tmpdirname
        )
        env = {**os.environ, 'PYTHONPATH': pythonpath, 'PIP_TARGET': tmpdirname}
        yield env


@pytest.mark.slow
@pytest.mark.parametrize("notebook_path", filter_notebooks(list_all_notebooks(), SKIP_NOTEBOOKS))
def test_notebooks_against_cirq_head(
    notebook_path, require_packages_not_changed, env_with_temporary_pip_target
):
    """Test that jupyter notebooks execute.

    In order to speed up the execution of these tests an auxiliary file may be supplied which
    performs substitutions on the notebook to make it faster.

    Specifically for a notebook file notebook.ipynb, one can supply a file notebook.tst which
    contains the substitutes.  The substitutions are provide in the form `pattern->replacement`
    where the pattern is what is matched and replaced. While the pattern is compiled as a
    regular expression, it is considered best practice to not use complicated regular expressions.
    Lines in this file that do not have `->` are ignored.
    """
    notebook_file = os.path.basename(notebook_path)
    notebook_rel_dir = os.path.dirname(os.path.relpath(notebook_path, "."))
    out_path = f"out/{notebook_rel_dir}/{notebook_file[:-6]}.out.ipynb"
    rewritten_notebook_path = rewrite_notebook(notebook_path)
    papermill_flags = "--no-request-save-on-cell-execute --autosave-cell-every 0"
    cmd = f"""mkdir -p out/{notebook_rel_dir}
papermill {rewritten_notebook_path} {out_path} {papermill_flags}"""

    result = shell_tools.run(
        cmd,
        log_run_to_stderr=False,
        shell=True,
        check=False,
        capture_output=True,
        env=env_with_temporary_pip_target,
    )

    if result.returncode != 0:
        print(result.stderr)
        pytest.fail(
            f"Notebook failure: {notebook_file}, please see {out_path} for the output "
            f"notebook (in Github Actions, you can download it from the workflow artifact"
            f" 'notebook-outputs')"
        )
    os.remove(rewritten_notebook_path)
