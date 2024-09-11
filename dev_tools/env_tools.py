# Copyright 2018 The Cirq Developers
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
import sys
from typing import Optional, Iterable, Callable, cast

from dev_tools import shell_tools, git_env_tools
from dev_tools.github_repository import GithubRepository
from dev_tools.prepared_env import PreparedEnv


def get_unhidden_ungenerated_python_files(directory: str) -> Iterable[str]:
    """Iterates through relevant python files within the given directory.

    Args:
        directory: The top-level directory to explore.

    Yields:
        File paths.
    """
    for dirpath, dirnames, filenames in os.walk(directory, topdown=True):
        if os.path.split(dirpath)[-1].startswith('.'):
            dirnames.clear()
            continue

        for filename in filenames:
            if filename.endswith('.py') and not filename.endswith('_pb2.py'):
                yield os.path.join(dirpath, filename)


def create_virtual_env(
    venv_path: str, requirements_paths: Iterable[str], python_path: str, verbose: bool
) -> None:
    """Creates a new virtual environment and then installs dependencies.

    Args:
        venv_path: Where to put the virtual environment's state.
        requirements_paths: Location of requirements files to -r install.
        python_path: The python binary to use.
        verbose: When set, more progress output is produced.
    """
    optional_quiet = [] if verbose else ['--quiet']
    shell_tools.run(
        ['virtualenv', *optional_quiet, '-p', python_path, venv_path], stdout=sys.stderr
    )
    pip_path = os.path.join(venv_path, 'bin', 'pip')
    for req_path in requirements_paths:
        shell_tools.run([pip_path, 'install', *optional_quiet, '-r', req_path], stdout=sys.stderr)


def prepare_temporary_test_environment(
    destination_directory: str,
    repository: GithubRepository,
    pull_request_number: Optional[int],
    verbose: bool,
    env_name: str = '.test_virtualenv',
    python_path: str = sys.executable,
    commit_ids_known_callback: Optional[Callable[[PreparedEnv], None]] = None,
) -> PreparedEnv:
    """Prepares a temporary test environment at the (existing empty) directory.

    Args:
        destination_directory: The location to put files. The caller is
            responsible for deleting the directory, whether or not this method
             succeeds or fails.
        repository: The github repository to download content from, if a pull
            request number is given.
        pull_request_number: If set, test content is fetched from github.
            Otherwise copies of local files are used.
        verbose: When set, more progress output is produced.
        env_name: The name to use for the virtual environment.
        python_path: Location of the python binary to use within the
            virtual environment.
        commit_ids_known_callback: A function to call when the actual commit id
            being tested is known, before the virtual environment is ready.

    Returns:
        Commit ids corresponding to content to test/compare.
    """
    # Fetch content.
    if pull_request_number is not None:
        env = git_env_tools.fetch_github_pull_request(
            destination_directory=destination_directory,
            repository=repository,
            pull_request_number=pull_request_number,
            verbose=verbose,
        )
    else:
        env = git_env_tools.fetch_local_files(
            destination_directory=destination_directory, verbose=verbose
        )

    if commit_ids_known_callback is not None:
        commit_ids_known_callback(env)

    # Create virtual environment.
    base_path = cast(str, env.destination_directory)
    env_path = os.path.join(base_path, env_name)
    req_path = os.path.join(base_path, 'requirements.txt')
    dev_req_path = os.path.join(base_path, 'dev_tools', 'requirements', 'deps', 'dev-tools.txt')
    contrib_req_path = os.path.join(base_path, 'cirq', 'contrib', 'requirements.txt')
    rev_paths = [req_path, dev_req_path, contrib_req_path]
    create_virtual_env(
        venv_path=env_path, python_path=python_path, requirements_paths=rev_paths, verbose=verbose
    )

    return PreparedEnv(
        github_repo=env.repository,
        actual_commit_id=env.actual_commit_id,
        compare_commit_id=env.compare_commit_id,
        destination_directory=env.destination_directory,
        virtual_env_path=env_path,
    )
