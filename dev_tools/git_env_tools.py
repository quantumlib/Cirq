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
import shutil
import subprocess
import sys

from dev_tools import shell_tools, github_repository, prepared_env


def get_repo_root() -> str:
    """Get the root of the git repository the cwd is within."""
    return shell_tools.output_of('git', 'rev-parse', '--show-toplevel')


def _git_fetch_for_comparison(remote: str,
                              actual_branch: str,
                              compare_branch: str) -> prepared_env.PreparedEnv:
    """Fetches two branches including their common ancestor.

    Limits the depth of the fetch to avoid unnecessary work. Scales up the
    depth exponentially and tries again when the initial guess is not deep
    enough.

    Args:
        remote: The location of the remote repository, in a format that the
            git command will understand.
        actual_branch: A remote branch or ref to fetch,
        compare_branch: Another remote branch or ref to fetch,

    Returns:
        A ComparableCommits containing the commit id of the actual branch and
        a the id of a commit to compare against (e.g. for when doing incremental
        checks).
    """
    actual_id = ''
    base_id = ''
    for depth in [10, 100, 1000, None]:
        depth_str = '' if depth is None else '--depth={}'.format(depth)

        shell_tools.run_cmd('git', 'fetch', remote, actual_branch, depth_str)
        actual_id = shell_tools.output_of('git', 'rev-parse', 'FETCH_HEAD')

        shell_tools.run_cmd('git', 'fetch', remote, compare_branch, depth_str)
        base_id = shell_tools.output_of('git', 'rev-parse', 'FETCH_HEAD')

        try:
            base_id = shell_tools.output_of(
                'git',
                'merge-base',
                actual_id,
                base_id)
            break
        except subprocess.CalledProcessError:
            # No common ancestor. We need to dig deeper.
            pass

    return prepared_env.PreparedEnv(None, actual_id, base_id, None, None)


def fetch_github_pull_request(destination_directory: str,
                              repository: github_repository.GithubRepository,
                              pull_request_number: int
                              ) -> prepared_env.PreparedEnv:
    """Uses content from github to create a dir for testing and comparisons.

    Args:
        destination_directory: The location to fetch the contents into.
        repository: The github repository that the commit lives under.
        pull_request_number: The id of the pull request to clone. If None, then
            the master branch is cloned instead.

    Returns:
        Commit ids corresponding to content to test/compare.
    """

    branch = 'pull/{}/head'.format(pull_request_number)
    os.chdir(destination_directory)
    print('chdir', destination_directory, file=sys.stderr)

    shell_tools.run_cmd('git', 'init', out=sys.stderr)
    result = _git_fetch_for_comparison(remote=repository.as_remote(),
                                       actual_branch=branch,
                                       compare_branch='master')
    shell_tools.run_cmd(
        'git', 'branch', 'compare_commit', result.compare_commit_id)
    shell_tools.run_cmd(
        'git', 'checkout', '-b', 'actual_commit', result.actual_commit_id)
    return prepared_env.PreparedEnv(
        repository=repository,
        actual_commit_id=result.actual_commit_id,
        compare_commit_id=result.compare_commit_id,
        destination_directory=destination_directory,
        virtual_env_path=None)


def fetch_local_files(destination_directory: str) -> prepared_env.PreparedEnv:
    """Uses local files to create a directory for testing and comparisons.

    Args:
        destination_directory: The directory where the copied files should go.

    Returns:
        Commit ids corresponding to content to test/compare.
    """
    shutil.rmtree(destination_directory)
    shutil.copytree(get_repo_root(), destination_directory)
    os.chdir(destination_directory)
    print('chdir', destination_directory, file=sys.stderr)

    shell_tools.run_cmd(
        'git',
        'commit',
        '-a',
        '-m', 'working changes',
        '--allow-empty',
        out=sys.stderr)
    shell_tools.run_shell(r'find | grep \.pyc$ | xargs rm -f')
    shell_tools.run_shell('find | grep __pycache__ | xargs rmdir')
    commit_id = shell_tools.output_of('git', 'rev-parse', 'HEAD')
    compare_id = shell_tools.output_of(
        'git', 'merge-base', commit_id, 'master')
    return prepared_env.PreparedEnv(
        repository=None,
        actual_commit_id=commit_id,
        compare_commit_id=compare_id,
        destination_directory=destination_directory,
        virtual_env_path=None)
