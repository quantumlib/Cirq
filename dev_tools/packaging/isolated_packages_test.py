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
import subprocess

import pytest

from dev_tools.modules import list_modules

# point, that is then cloned to a separate folder for each test.
PACKAGES = [
    "-r",
    "dev_tools/requirements/deps/pytest.txt",
    "-r",
    # one of the _compat_test.py tests uses flynt for testing metadata
    "dev_tools/requirements/deps/flynt.txt",
]


@pytest.mark.slow
@pytest.mark.parametrize('module', list_modules(), ids=[m.name for m in list_modules()])
def test_isolated_packages(cloned_env, module):
    env = cloned_env("isolated_packages", *PACKAGES)

    # ensure that no cirq packages are on the PYTHONPATH, this is important, otherwise
    # the "isolation" fails and for example cirq-core would be on the PATH
    if "PYTHONPATH" in os.environ:
        del os.environ["PYTHONPATH"]  # coverage: ignore

    if str(module.root) != "cirq-core":
        result = subprocess.run(
            f"{env}/bin/pip install ./{module.root}", capture_output=True, shell=True
        )
        assert result.returncode != 0, (
            f"{module.name} should have failed to " f"install without cirq-core!"
        )

    result = subprocess.run(
        f"{env}/bin/pip install ./{module.root} ./cirq-core", capture_output=True, shell=True
    )
    assert (
        result.returncode == 0
    ), f"Failed to install {module.name}:\n{str(result.stderr, encoding='UTF-8')}"

    result = subprocess.run(
        f"{env}/bin/pytest ./{module.root} --ignore ./cirq-core/cirq/contrib",
        capture_output=True,
        shell=True,
    )
    assert (
        result.returncode == 0
    ), f"Failed isolated tests for {module.name}:\n{str(result.stdout, encoding='UTF-8')}"
