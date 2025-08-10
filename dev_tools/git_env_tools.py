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

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from dev_tools import github_repository, prepared_env, shell_tools


def get_repo_root() -> str:
    """Get the root of the git repository the cwd is within."""
    return shell_tools.output_of(['git', 'rev-parse', '--show-toplevel'])


def _git_fetch_for_comparison(
    remote: str, actual_branch: str, compare_branch: str, verbose: bool
) -> prepared_env.PreparedEnv:
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
    optional_quiet = [] if verbose else ['--quiet']
    for depth in [10, 100, 1000, None]:
        optional_depth = [] if depth is None else [f'--depth={depth}']

        shell_tools.run(
            ['git', 'fetch', *optional_quiet, remote, actual_branch, *optional_depth],
            log_run_to_stderr=verbose,
        )
        actual_id = shell_tools.output_of(['git', 'rev-parse', 'FETCH_HEAD'])

        shell_tools.run(
            ['git', 'fetch', *optional_quiet, remote, compare_branch, *optional_depth],
            log_run_to_stderr=verbose,
        )
        base_id = shell_tools.output_of(['git', 'rev-parse', 'FETCH_HEAD'])

        try:
            base_id = shell_tools.output_of(['git', 'merge-base', actual_id, base_id])
            break
        except subprocess.CalledProcessError:
            # No common ancestor. We need to dig deeper.
            pass

    return prepared_env.PreparedEnv(None, actual_id, base_id, None, None)


def fetch_github_pull_request(
    destination_directory: str,
    repository: github_repository.GithubRepository,
    pull_request_number: int,
    verbose: bool,
) -> prepared_env.PreparedEnv:
    """Uses content from github to create a dir for testing and comparisons.

    Args:
        destination_directory: The location to fetch the contents into.
        repository: The github repository that the commit lives under.
        pull_request_number: The id of the pull request to clone. If None, then
            the main branch is cloned instead.
        verbose: When set, more progress output is produced.

    Returns:
        Commit ids corresponding to content to test/compare.
    """

    branch = f'pull/{pull_request_number}/head'
    os.chdir(destination_directory)
    print('chdir', destination_directory, file=sys.stderr)

    optional_quiet = [] if verbose else ['--quiet']
    shell_tools.run(['git', 'init', *optional_quiet], stdout=sys.stderr)
    result = _git_fetch_for_comparison(
        remote=repository.as_remote(), actual_branch=branch, compare_branch='main', verbose=verbose
    )
    optional_actual_commit_id = [] if result.actual_commit_id is None else [result.actual_commit_id]
    shell_tools.run(
        ['git', 'branch', *optional_quiet, 'compare_commit', result.compare_commit_id],
        log_run_to_stderr=verbose,
    )
    shell_tools.run(
        ['git', 'checkout', *optional_quiet, '-b', 'actual_commit', *optional_actual_commit_id],
        log_run_to_stderr=verbose,
    )
    return prepared_env.PreparedEnv(
        github_repo=repository,
        actual_commit_id=result.actual_commit_id,
        compare_commit_id=result.compare_commit_id,
        destination_directory=destination_directory,
        virtual_env_path=None,
    )


def fetch_local_files(destination_directory: str, verbose: bool) -> prepared_env.PreparedEnv:
    """Uses local files to create a directory for testing and comparisons.

    Args:
        destination_directory: The directory where the copied files should go.
        verbose: When set, more progress output is produced.

    Returns:
        Commit ids corresponding to content to test/compare.
    """
    staging_dir = destination_directory + '-staging'
    optional_quiet = [] if verbose else ['--quiet']
    try:
        shutil.copytree(get_repo_root(), staging_dir)
        os.chdir(staging_dir)
        if verbose:
            print('chdir', staging_dir, file=sys.stderr)

        shell_tools.run(['git', 'add', '--all'], stdout=sys.stderr, log_run_to_stderr=verbose)

        shell_tools.run(
            [
                'git',
                'commit',
                '-m',
                'working changes',
                '--allow-empty',
                '--no-gpg-sign',
                *optional_quiet,
            ],
            stdout=sys.stderr,
            log_run_to_stderr=verbose,
        )

        cur_commit = shell_tools.output_of(['git', 'rev-parse', 'HEAD'])

        os.chdir(destination_directory)
        if verbose:
            print('chdir', destination_directory, file=sys.stderr)
        shell_tools.run(
            ['git', 'init', *optional_quiet], stdout=sys.stderr, log_run_to_stderr=verbose
        )
        result = _git_fetch_for_comparison(staging_dir, cur_commit, 'main', verbose=verbose)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    optional_actual_commit_id = [] if result.actual_commit_id is None else [result.actual_commit_id]
    shell_tools.run(
        ['git', 'branch', *optional_quiet, 'compare_commit', result.compare_commit_id],
        log_run_to_stderr=verbose,
    )
    shell_tools.run(
        ['git', 'checkout', *optional_quiet, '-b', 'actual_commit', *optional_actual_commit_id],
        log_run_to_stderr=verbose,
    )
    return prepared_env.PreparedEnv(
        github_repo=None,
        actual_commit_id=result.actual_commit_id,
        compare_commit_id=result.compare_commit_id,
        destination_directory=destination_directory,
        virtual_env_path=None,
    )
