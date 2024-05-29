# Copyright 2021 The Cirq Developers
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

import os
import shutil
import subprocess
from unittest import mock

import pytest

from dev_tools import shell_tools
from dev_tools.modules import list_modules
from dev_tools.test_utils import only_on_posix

PACKAGES = ["-r", "dev_tools/requirements/isolated-base.env.txt"]


@only_on_posix
@pytest.mark.slow
# ensure that no cirq packages are on the PYTHONPATH, this is important, otherwise
# the "isolation" fails and for example cirq-core would be on the PATH
@mock.patch.dict(os.environ, {"PYTHONPATH": ""})
@pytest.mark.parametrize('module', list_modules(), ids=[m.name for m in list_modules()])
def test_isolated_packages(cloned_env, module, tmp_path):
    env = cloned_env("isolated_packages", *PACKAGES)

    if str(module.root) != "cirq-core":
        assert f'cirq-core=={module.version}' in module.install_requires

    # TODO: Remove after upgrading package builds from setup.py to PEP-517
    # Create per-worker copy of cirq-core sources so that parallel builds
    # of cirq-core wheel do not conflict.
    opt_cirq_core = (
        [str(shutil.copytree("./cirq-core", tmp_path / "cirq-core"))]
        if str(module.root) != "cirq-core"
        else []
    )
    result = shell_tools.run(
        [f"{env}/bin/pip", "install", f"./{module.root}", *opt_cirq_core],
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, f"Failed to install {module.name}:\n{result.stderr}"

    result = shell_tools.run(
        f"{env}/bin/pytest ./{module.root} --ignore ./cirq-core/cirq/contrib".split(),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, f"Failed isolated tests for {module.name}:\n{result.stdout}"
    shutil.rmtree(env)
