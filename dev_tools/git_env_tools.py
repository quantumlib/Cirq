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
                              compare_branch: str,
                              verbose: bool) -> prepared_env.PreparedEnv:
    """Fetches two branches including their common ancestor.

    Limits the depth of the fetch to avoid unnecessary work. Scales up the
    depth exponentially and tries again when the initial guess is not deep
    enough.

    Args:
        remote: The location of the remote repository, in a format that the
            git command will understand.
        actual_branch: A remote branch or ref to fetch,
        compare_branch: Another remote branch or ref to fetch,
        verbose: When set, more progress output is produced.

    Returns:
        A ComparableCommits containing the commit id of the actual branch and
        a the id of a commit to compare against (e.g. for when doing incremental
        checks).
    """
    actual_id = ''
    base_id = ''
    for depth in [10, 100, 1000, None]:
        depth_str = '' if depth is None else '--depth={}'.format(depth)

        shell_tools.run_cmd(
            'git',
            'fetch',
            None if verbose else '--quiet',
            remote,
            actual_branch,
            depth_str,
            log_run_to_stderr=verbose)
        actual_id = shell_tools.output_of('git', 'rev-parse', 'FETCH_HEAD')

        shell_tools.run_cmd('git',
                            'fetch',
                            None if verbose else '--quiet',
                            remote,
                            compare_branch,
                            depth_str,
                            log_run_to_stderr=verbose)
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
                              pull_request_number: int,
                              verbose: bool
                              ) -> prepared_env.PreparedEnv:
    """Uses content from github to create a dir for testing and comparisons.

    Args:
        destination_directory: The location to fetch the contents into.
        repository: The github repository that the commit lives under.
        pull_request_number: The id of the pull request to clone. If None, then
            the master branch is cloned instead.
        verbose: When set, more progress output is produced.

    Returns:
        Commit ids corresponding to content to test/compare.
    """

    branch = 'pull/{}/head'.format(pull_request_number)
    os.chdir(destination_directory)
    print('chdir', destination_directory, file=sys.stderr)

    shell_tools.run_cmd(
        'git',
        'init',
        None if verbose else '--quiet',
        out=sys.stderr)
    result = _git_fetch_for_comparison(remote=repository.as_remote(),
                                       actual_branch=branch,
                                       compare_branch='master',
                                       verbose=verbose)
    shell_tools.run_cmd(
        'git',
        'branch',
        None if verbose else '--quiet',
        'compare_commit',
        result.compare_commit_id,
        log_run_to_stderr=verbose)
    shell_tools.run_cmd(
        'git',
        'checkout',
        None if verbose else '--quiet',
        '-b',
        'actual_commit',
        result.actual_commit_id,
        log_run_to_stderr=verbose)
    return prepared_env.PreparedEnv(
        github_repo=repository,
        actual_commit_id=result.actual_commit_id,
        compare_commit_id=result.compare_commit_id,
        destination_directory=destination_directory,
        virtual_env_path=None)


def fetch_local_files(destination_directory: str,
                      verbose: bool) -> prepared_env.PreparedEnv:
    """Uses local files to create a directory for testing and comparisons.

    Args:
        destination_directory: The directory where the copied files should go.
        verbose: When set, more progress output is produced.

    Returns:
        Commit ids corresponding to content to test/compare.
    """
    staging_dir = destination_directory + '-staging'
    try:
        shutil.copytree(get_repo_root(), staging_dir)
        os.chdir(staging_dir)
        if verbose:
            print('chdir', staging_dir, file=sys.stderr)

        shell_tools.run_cmd(
            'git',
            'add',
            '--all',
            out=sys.stderr,
            log_run_to_stderr=verbose)

        shell_tools.run_cmd(
            'git',
            'commit',
            '-m', 'working changes',
            '--allow-empty',
            '--no-gpg-sign',
            None if verbose else '--quiet',
            out=sys.stderr,
            log_run_to_stderr=verbose)

        cur_commit = shell_tools.output_of('git', 'rev-parse', 'HEAD')

        os.chdir(destination_directory)
        if verbose:
            print('chdir', destination_directory, file=sys.stderr)
        shell_tools.run_cmd('git',
                            'init',
                            None if verbose else '--quiet',
                            out=sys.stderr,
                            log_run_to_stderr=verbose)
        result = _git_fetch_for_comparison(staging_dir,
                                           cur_commit,
                                           'master',
                                           verbose=verbose)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    shell_tools.run_cmd(
        'git',
        'branch',
        None if verbose else '--quiet',
        'compare_commit',
        result.compare_commit_id,
        log_run_to_stderr=verbose)
    shell_tools.run_cmd(
        'git',
        'checkout',
        None if verbose else '--quiet',
        '-b',
        'actual_commit',
        result.actual_commit_id,
        log_run_to_stderr=verbose)
    return prepared_env.PreparedEnv(
        github_repo=None,
        actual_commit_id=result.actual_commit_id,
        compare_commit_id=result.compare_commit_id,
        destination_directory=destination_directory,
        virtual_env_path=None)
