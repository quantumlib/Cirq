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

# ========================== ISOLATED NOTEBOOK TESTS ============================================
#
# In these tests are only changed notebooks are tested. It is assumed that notebooks install cirq
# conditionally if they can't import cirq. This installation path is the main focus and it is
# exercised in an isolated virtual environment for each notebook. This is also the path that is
# tested in the devsite workflows, these tests meant to provide earlier feedback.
#
# In case the dev environment changes or this particular file changes, all notebooks are executed!
# This can take a long time and even lead to timeout on Github Actions, hence partitioning of the
# tests is possible, via setting the NOTEBOOK_PARTITIONS env var to e.g. 5, and then passing to
# pytest the `-k partition-0` or `-k partition-1`, etc. argument to limit to the given partition.
import os
import re
import subprocess
import sys
import warnings
from typing import Set, List

import pytest

from dev_tools import shell_tools
from dev_tools.notebooks import list_all_notebooks, filter_notebooks, rewrite_notebook

# these notebooks rely on features that are not released yet
# after every release we should raise a PR and empty out this list
# note that these notebooks are still tested in dev_tools/notebook_test.py
# Please, always indicate in comments the feature used for easier bookkeeping.

NOTEBOOKS_DEPENDING_ON_UNRELEASED_FEATURES: List[str] = [
    # Hardcoded qubit placement
    'docs/google/qubit-placement.ipynb',
    # get_qcs_objects_for_notebook
    'docs/noise/calibration_api.ipynb',
    'docs/tutorials/google/colab.ipynb',
    'docs/tutorials/google/identifying_hardware_changes.ipynb',
    'docs/tutorials/google/echoes.ipynb',
    'docs/noise/floquet_calibration_example.ipynb',
    'docs/tutorials/google/spin_echoes.ipynb',
    'docs/tutorials/google/start.ipynb',
    'docs/tutorials/google/visualizing_calibration_metrics.ipynb',
    'docs/noise/qcvv/xeb_calibration_example.ipynb',
    'docs/named_topologies.ipynb',
    'docs/start/intro.ipynb',
    # Circuit routing
    'docs/transform/routing_transformer.ipynb',
]

# By default all notebooks should be tested, however, this list contains exceptions to the rule
# please always add a reason for skipping.
SKIP_NOTEBOOKS = [
    # skipping vendor notebooks as we don't have auth sorted out
    "**/aqt/*.ipynb",
    "**/azure-quantum/*.ipynb",
    "**/google/*.ipynb",
    "**/ionq/*.ipynb",
    "**/pasqal/*.ipynb",
    # Rigetti uses local simulation with docker, so should work
    # if you run into issues locally, run
    # `docker compose -f cirq-rigetti/docker-compose.test.yaml up`
    "**/rigetti/*.ipynb",
    # skipping fidelity estimation due to
    # https://github.com/quantumlib/Cirq/issues/3502
    "examples/*fidelity*",
    # Also skipping stabilizer code testing.
    "examples/*stabilizer_code*",
    # An intentionally empty/template code notebook.
    "docs/simulate/qvm_builder_code.ipynb",
    *NOTEBOOKS_DEPENDING_ON_UNRELEASED_FEATURES,
]

# The Rigetti integration requires Python >= 3.7.
if sys.version_info < (3, 7):
    SKIP_NOTEBOOKS.append("**/rigetti/*.ipynb")

# As these notebooks run in an isolated env, we want to minimize dependencies that are
# installed. We assume colab packages (feel free to add dependencies here that appear in colab, as
# needed by the notebooks) exist. These packages are installed into a base environment as a starting
# point, that is then cloned to a separate folder for each test.
PACKAGES = [
    # for running the notebooks
    "papermill",
    "jupyter",
    # assumed to be part of colab
    "seaborn~=0.11.1",
    # https://github.com/nteract/papermill/issues/519
    'ipykernel==5.3.4',
    # https://github.com/ipython/ipython/issues/12941
    'ipython==7.22',
    # to ensure networkx works nicely
    # https://github.com/networkx/networkx/issues/4718 pinned networkx 2.5.1 to 4.4.2
    # however, jupyter brings in 5.0.6
    'decorator<5',
]


# TODO(3577): extract these out to common utilities when we rewrite bash scripts in python
def _find_base_revision():
    for rev in ['upstream/master', 'origin/master', 'master']:
        try:
            result = subprocess.run(
                f'git cat-file -t {rev}'.split(), stdout=subprocess.PIPE, universal_newlines=True
            )
            if result.stdout == "commit\n":
                return rev
        except subprocess.CalledProcessError as e:
            print(e)
    raise ValueError("Can't find a base revision to compare the files with.")


def _list_changed_notebooks() -> Set[str]:
    try:
        rev = _find_base_revision()
        output = subprocess.check_output(f'git diff --diff-filter=d --name-only {rev}'.split())
        lines = output.decode('utf-8').splitlines()
        # run all tests if this file or any of the dev tool dependencies change
        if any(
            l
            for l in lines
            if l.endswith("isolated_notebook_test.py") or l.startswith("dev_tools/requirements")
        ):
            return list_all_notebooks()
        return set(l for l in lines if l.endswith(".ipynb"))
    except ValueError as e:
        # It would be nicer if we could somehow automatically skip the execution of this completely,
        # however, in order to be able to rely on parallel pytest (xdist) we need parametrization to
        # work, thus this will be executed during the collection phase even when the notebook tests
        # are not included (the "-m slow" flag is not passed to pytest). So, in order to not break
        # the complete test collection phase in other tests where there is no git history
        # (fetch-depth:1), we'll just swallow the error here with a warning.
        warnings.warn(
            f"No changed notebooks are tested "
            f"(this is expected in non-notebook tests in CI): {e}"
        )
        return set()


def _partitioned_test_cases(notebooks):
    n_partitions = int(os.environ.get("NOTEBOOK_PARTITIONS", "1"))
    return [(f"partition-{i%n_partitions}", notebook) for i, notebook in enumerate(notebooks)]


@pytest.mark.slow
@pytest.mark.parametrize(
    "partition, notebook_path",
    _partitioned_test_cases(filter_notebooks(_list_changed_notebooks(), SKIP_NOTEBOOKS)),
)
def test_notebooks_against_released_cirq(partition, notebook_path, cloned_env):
    """Tests the notebooks in isolated virtual environments.

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
    notebook_env = cloned_env("isolated_notebook_tests", *PACKAGES)

    notebook_file = os.path.basename(notebook_path)

    rewritten_notebook_descriptor, rewritten_notebook_path = rewrite_notebook(notebook_path)

    cmd = f"""
mkdir -p out/{notebook_rel_dir}
cd {notebook_env}
. ./bin/activate
pip list
papermill {rewritten_notebook_path} {os.getcwd()}/{out_path}"""
    result = shell_tools.run(
        cmd,
        log_run_to_stderr=False,
        shell=True,
        check=False,
        capture_output=True,
        # important to get rid of PYTHONPATH specifically, which contains
        # the Cirq repo path due to check/pytest
        env={},
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        pytest.fail(
            f"Notebook failure: {notebook_file}, please see {out_path} for the output "
            f"notebook (in Github Actions, you can download it from the workflow artifact"
            f" 'notebook-outputs'). \n"
            f"If this is a new failure in this notebook due to a new change, "
            f"that is only available in master for now, consider adding `pip install --pre cirq` "
            f"instead of `pip install cirq` to this notebook, and exclude it from "
            f"dev_tools/notebooks/isolated_notebook_test.py."
        )

    if rewritten_notebook_descriptor:
        os.close(rewritten_notebook_descriptor)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS_DEPENDING_ON_UNRELEASED_FEATURES)
def test_ensure_unreleased_notebooks_install_cirq_pre(notebook_path):
    # utf-8 is important for Windows testing, otherwise characters like ┌──┐ fail on cp1252
    with open(notebook_path, encoding="utf-8") as notebook:
        content = notebook.read()
        mandatory_matches = [
            r"!pip install --quiet cirq(-google)? --pre",
            r"Note: this notebook relies on unreleased Cirq features\. "
            r"If you want to try these features, make sure you install cirq(-google)? via "
            r"`pip install cirq(-google)? --pre`\.",
        ]

        for m in mandatory_matches:
            assert re.search(m, content), (
                f"{notebook_path} is marked as NOTEBOOKS_DEPENDING_ON_UNRELEASED_FEATURES, "
                f"however it contains no line matching:\n{m}"
            )
