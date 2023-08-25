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

"""Tests the cloned_env fixture in conftest.py"""
import json
import os
import shutil
import subprocess
from unittest import mock

import pytest

from dev_tools import shell_tools
from dev_tools.test_utils import only_on_posix


# due to shell_tools dependencies windows builds break on this
# see https://github.com/quantumlib/Cirq/issues/4394
@only_on_posix
# ensure that no cirq packages are on the PYTHONPATH, this is important, otherwise
# the "isolation" fails and all the cirq modules would be in the list
@mock.patch.dict(os.environ, {"PYTHONPATH": ""})
@pytest.mark.parametrize('param', ['a', 'b', 'c'])
def test_isolated_env_cloning(cloned_env, param):
    env = cloned_env("test_isolated", "flynt==0.64")
    assert (env / "bin" / "pip").is_file()

    result = shell_tools.run(f"{env}/bin/pip list --format=json".split(), stdout=subprocess.PIPE)
    packages = json.loads(result.stdout)
    assert {"name": "flynt", "version": "0.64"} in packages
    assert {"astor", "flynt", "pip", "setuptools", "wheel"} == set(p['name'] for p in packages)
    shutil.rmtree(env)
