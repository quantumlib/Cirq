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
import functools
import glob
import os
import sys

import pytest
from filelock import FileLock

from dev_tools import shell_tools
from dev_tools.env_tools import create_virtual_env

SKIP_NOTEBOOKS = [
    # skipping vendor notebooks as we don't have auth sorted out
    "**/google/*.ipynb",
    "**/pasqal/*.ipynb",
    "**/aqt/*.ipynb",
    # skipping quantum volume notebooks as they have issues
    # see https://github.com/quantumlib/Cirq/issues/3501
    "examples/advanced/*.ipynb",
    # skipping fidelity estimation due to
    # https://github.com/quantumlib/Cirq/issues/3502
    "examples/*fidelity*",
]


def _tested_notebooks():
    all_notebooks = set(glob.glob("**/*.ipynb", recursive=True))
    skipped_notebooks = functools.reduce(
        lambda a, b: a.union(b), list(set(glob.glob(g, recursive=True)) for g in SKIP_NOTEBOOKS)
    )

    # sorted is important otherwise pytest-xdist will complain that
    # the workers have differnent parametrization:
    # https://github.com/pytest-dev/pytest-xdist/issues/432
    return sorted(os.path.abspath(n) for n in all_notebooks.difference(skipped_notebooks))


TESTED_NOTEBOOKS = _tested_notebooks()

PACKAGES = [
    # for running the notebooks
    "papermill",
    "jupyter",
    "virtualenv-clone",
    # assumed to be part of colab
    "seaborn",
]


@pytest.mark.slow
@pytest.fixture(scope="session")
def base_env(tmp_path_factory, worker_id):
    # get the temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent.parent
    proto_dir = root_tmp_dir / "proto_dir"
    with FileLock(str(proto_dir) + ".lock"):
        if proto_dir.is_dir():
            print(f"{worker_id} returning as {proto_dir} is a dir!")
            print(
                f"If all the notebooks are failing, the test framework might "
                f"have left this directory around. Try 'rm -rf {proto_dir}'"
            )
        else:
            print(f"{worker_id} creating stuff...")
            create_base_env(proto_dir)

    return root_tmp_dir, proto_dir


def create_base_env(proto_dir):
    create_virtual_env(str(proto_dir), [], sys.executable, True)
    pip_path = str(proto_dir / "bin" / "pip")
    shell_tools.run_cmd(pip_path, "install", *PACKAGES)


@pytest.mark.slow
@pytest.mark.parametrize("notebook_path", TESTED_NOTEBOOKS)
def test_notebooks(notebook_path, base_env):
    """Ensures testing the notebooks in isolated virtual environments."""
    tmpdir, proto_dir = base_env

    notebook_file = os.path.basename(notebook_path)
    dir_name = notebook_file.rstrip(".ipynb")

    notebook_env = os.path.join(tmpdir, f"{dir_name}")
    cmd = f"""
{proto_dir}/bin/virtualenv-clone {proto_dir} {notebook_env}
cd {notebook_env}
. ./bin/activate
papermill {notebook_path}"""
    stdout, stderr, status = shell_tools.run_shell(
        cmd=cmd,
        log_run_to_stderr=False,
        raise_on_fail=False,
        out=shell_tools.TeeCapture(),
        err=shell_tools.TeeCapture(),
    )

    if status != 0:
        print(stdout)
        print(stderr)
        pytest.fail(f"Notebook failure: {notebook_file}")
