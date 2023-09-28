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

PACKAGES = ["-r", "dev_tools/requirements/isolated-base.env.txt"]


@pytest.mark.slow
# ensure that no cirq packages are on the PYTHONPATH, this is important, otherwise
# the "isolation" fails and for example cirq-core would be on the PATH
@mock.patch.dict(os.environ, {"PYTHONPATH": ""})
@pytest.mark.parametrize('module', list_modules(), ids=[m.name for m in list_modules()])
def test_isolated_packages(cloned_env, module):
    env = cloned_env("isolated_packages", *PACKAGES)

    if str(module.root) != "cirq-core":
        assert f'cirq-core=={module.version}' in module.install_requires

    result = shell_tools.run(
        f"{env}/bin/pip install ./{module.root} ./cirq-core".split(),
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
