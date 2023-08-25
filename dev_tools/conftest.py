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
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Tuple

import pytest
from filelock import FileLock

from dev_tools import shell_tools
from dev_tools.env_tools import create_virtual_env


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark tests as slow")
    config.addinivalue_line("markers", "weekly: mark tests as run only by weekly automation")


def pytest_collection_modifyitems(config, items):
    markexpr = config.option.markexpr
    if markexpr:
        return  # let pytest handle this

    skip_slow_marker = pytest.mark.skip(reason='slow marker not selected')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow_marker)

    skip_weekly_marker = pytest.mark.skip(reason='only run by weekly automation')
    for item in items:
        if 'weekly' in item.keywords:
            item.add_marker(skip_weekly_marker)


@pytest.fixture(scope="session")
def cloned_env(testrun_uid, worker_id):
    """Fixture to allow tests to run in a clean virtual env.

    It de-duplicates installation of base packages. Assuming `virtualenv-clone` exists on the PATH,
    it creates first a prototype environment and then clones for each new request the same env.
    This fixture is safe to use with parallel execution, i.e. pytest-xdist. The workers synchronize
    via a file lock, the first worker will (re)create the prototype environment, the others will
    reuse it via cloning.

    A group of tests that share the same base environment is identified by a name, `env_dir`,
    which will become the directory within the temporary directory to hold the virtualenv.

    Usage:

    >>> def test_something_in_clean_env(cloned_env):
            # base_env will point to a pathlib.Path containing the virtual env which will
            # have quimb, jinja and whatever reqs.txt contained.
            base_env = cloned_env("some_tests", "quimb", "jinja", "-r", "reqs.txt")

            # To install new packages (that are potentially different for each test instance)
            # just run pip install from the virtual env
            subprocess.run(f"{base_env}/bin/pip install something".split(" "))
            ...

    Returns:
        a function to create the cloned base environment with signature
        `def base_env_creator(env_dir: str, *pip_install_args: str) -> Path`.
        Use `env_dir` to specify the directory name per shared base packages.
        Use `pip_install_args` varargs to pass arguments to `pip install`, these
        can be requirements files, e.g. `'-r','dev_tools/.../something.txt'` or
        actual packages as well, e.g.`'quimb'`.
    """
    base_dir = None

    def base_env_creator(env_dir_name: str, *pip_install_args: str) -> Path:
        """The function to create a cloned base environment."""
        # get/create a temp directory shared by all workers
        base_temp_path = Path(tempfile.gettempdir()) / "cirq-pytest"
        os.makedirs(name=base_temp_path, exist_ok=True)
        nonlocal base_dir
        base_dir = base_temp_path / env_dir_name
        with FileLock(str(base_dir) + ".lock"):
            if _check_for_reuse_or_recreate(base_dir):
                print(f"Pytest worker [{worker_id}] is reusing {base_dir} for '{env_dir_name}'.")
            else:
                print(f"Pytest worker [{worker_id}] is creating {base_dir} for '{env_dir_name}'.")
                _create_base_env(base_dir, pip_install_args)

        clone_dir = base_temp_path / str(uuid.uuid4())
        shell_tools.run(["virtualenv-clone", str(base_dir), str(clone_dir)])
        return clone_dir

    def _check_for_reuse_or_recreate(env_dir: Path):
        reuse = False
        if env_dir.is_dir() and (env_dir / "testrun.uid").is_file():
            uid = open(env_dir / "testrun.uid").readlines()[0]
            # if the dir is from this test session, let's reuse it
            if uid == testrun_uid:
                reuse = True
            else:
                # if we have a dir from a previous test session, recreate it
                shutil.rmtree(env_dir)
        return reuse

    def _create_base_env(base_dir: Path, pip_install_args: Tuple[str, ...]):
        try:
            create_virtual_env(str(base_dir), [], sys.executable, True)
            with open(base_dir / "testrun.uid", mode="w") as f:
                f.write(testrun_uid)
            if pip_install_args:
                shell_tools.run([f"{base_dir}/bin/pip", "install", *pip_install_args])
        except BaseException as ex:
            # cleanup on failure
            if base_dir.is_dir():
                print(f"Removing {base_dir}, due to error: {ex}")
                shutil.rmtree(base_dir)
            raise

    return base_env_creator
