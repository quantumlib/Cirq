# Copyright 2018 Google LLC
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

import asyncio
import subprocess
import sys
from typing import List, Optional, TypeVar, Tuple, Iterable, Callable

import os
import shutil

import requests

BOLD = 1
DIM = 2
RED = 31
GREEN = 32
YELLOW = 33
TResult = TypeVar('TResult')


class GithubRepository:
    def __init__(self,
                 organization: str,
                 name: str,
                 access_token: Optional[str]) -> None:
        self.organization = organization
        self.name = name
        self.access_token = access_token

    def as_remote(self) -> str:
        return 'git@github.com:{}/{}.git'.format(self.organization,
                                                 self.name)


class PreparedEnv:
    def __init__(self,
                 repository: Optional[GithubRepository],
                 actual_commit_id: str,
                 compare_commit_id: str,
                 destination_directory: Optional[str],
                 virtual_env_path: Optional[str]) -> None:
        self.repository = repository
        self.actual_commit_id = actual_commit_id
        self.compare_commit_id = compare_commit_id
        if self.compare_commit_id == self.actual_commit_id:
            self.compare_commit_id += '~1'

        self.destination_directory = destination_directory
        self.virtual_env_path = virtual_env_path

    def report_status(self,
                      state: str,
                      description: str,
                      context: str,
                      target_url: Optional[str] = None):
        """Sets a commit status indicator on github, if running from a PR.

        If not running from a PR (i.e. repository is None), then just prints
        to stderr.

        Args:
            state: The state of the status indicator.
                Must be 'error', 'failure', 'pending', or 'success'.
            description: A summary of why the state is what it is,
                e.g. '5 lint errors' or 'tests passed!'.
            context: The name of the status indicator, e.g. 'pytest' or 'lint'.
            target_url: Optional location where additional details about the
                status can be found, e.g. an online test results page.

        Raises:
            ValueError: Not one of the allowed states.
            IOError: The HTTP post request failed, or the response didn't have
                a 201 code indicating success in the expected way.
        """

        print(repr(('report_status',
                    context,
                    state,
                    description,
                    target_url)), file=sys.stderr)

        if (self.repository is not None and
                self.repository.access_token is not None):
            github_set_status_indicator(
                repository_organization=self.repository.organization,
                repository_name=self.repository.name,
                repository_access_token=self.repository.access_token,
                commit_id=self.actual_commit_id,
                state=state,
                description=description,
                context=context,
                target_url=target_url)
        else:
            print('(no access token; skipped reporting status to github)',
                  file=sys.stderr)


def highlight(text: str, color_code: int, bold: bool=False) -> str:
    """Wraps the given string with terminal color codes.

    Args:
        text: The content to highlight.
        color_code: The color to highlight with, e.g. 'shelltools.RED'.
        bold: Whether to bold the content in addition to coloring.

    Returns:
        The highlighted string.
    """
    return '{}\033[{}m{}\033[0m'.format(
        '\033[1m' if bold else '',
        color_code,
        text,)


def sub_run(*cmd: str,
            capture_stdout: bool = False,
            capture_stderr: bool = False,
            redirect_stdout_into_stderr: bool = False,
            raise_error_if_process_fails: bool = True,
            silent: bool = False,
            **kwargs
            ) -> Tuple[Optional[str], Optional[str], int]:
    """Run a shell command and return the output as a string."""
    future = _async_run(
        *cmd,
        capture_stdout=capture_stdout,
        capture_stderr=capture_stderr,
        redirect_stdout_into_stderr=redirect_stdout_into_stderr,
        raise_error_if_process_fails=raise_error_if_process_fails,
        silent=silent,
        **kwargs)
    return asyncio.get_event_loop().run_until_complete(future)


def output_of(*cmd: str,
              silent: bool = False,
              **kwargs) -> str:
    """Run a shell command and return the output as a string."""
    result = sub_run(*cmd,
                     silent=silent,
                     capture_stdout=True,
                     redirect_stdout_into_stderr=True,
                     **kwargs)[0]

    # Strip final newline.
    if result.endswith('\n'):
        result = result[:-1]

    return result


async def _async_forward(input_pipe,
                         output_pipe,
                         capture: bool,
                         silent: bool) -> Optional[str]:
    chunks = [] if capture else None
    async for chunk in input_pipe:
        if not isinstance(chunk, str):
            chunk = chunk.decode()
        if not silent:
            print(chunk, file=output_pipe, end='')
        if capture:
            chunks.append(chunk)

    return ''.join(chunks) if capture else None


async def _async_run(*cmd: str,
                     capture_stdout: bool = False,
                     capture_stderr: bool = False,
                     redirect_stdout_into_stderr: bool = False,
                     raise_error_if_process_fails: bool = True,
                     silent: bool = False,
                     shell: bool = False,
                     **kwargs
                     ) -> Tuple[Optional[str], Optional[str], int]:
    if not silent:
        print('run', cmd, file=sys.stderr)
    if shell:
        process = await asyncio.create_subprocess_shell(
            ' '.join(cmd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs)
    else:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs)

    future_stdout_lines = _async_forward(
        process.stdout,
        sys.stderr if redirect_stdout_into_stderr else sys.stdout,
        capture=capture_stdout,
        silent=silent)
    future_stderr_lines = _async_forward(
        process.stderr,
        sys.stderr,
        capture=capture_stderr,
        silent=silent)

    stderr_lines, stdout_lines = await asyncio.gather(
        future_stderr_lines,
        future_stdout_lines
    )
    await process.wait()

    if raise_error_if_process_fails and process.returncode:
        raise subprocess.CalledProcessError(process.returncode, cmd)

    return stdout_lines, stderr_lines, process.returncode


def py_files(files: List[str]) -> List[str]:
    """Filter a list of filenames to include non-autogenerated python files."""
    return [f
            for f in files
            if f.endswith('.py') and not f.endswith('_pb2.py')]


def get_repo_root() -> str:
    """Get the root of the current git repository."""
    return output_of('git', 'rev-parse', '--show-toplevel')


def get_changed_files(env: PreparedEnv) -> List[str]:
    """Get the files changed on one git branch, since diverging from another.

    Args:
        env: The environment to run in.

    Returns:
        List[str]: File paths of changed files, relative to the git repo root.
    """
    out = output_of(
        'git',
        'diff',
        '--name-only',
        env.compare_commit_id,
        env.actual_commit_id,
        silent=True,
        cwd=env.destination_directory)
    return [e for e in out.split('\n') if e.strip()]


def get_unhidden_ungenerated_python_files(directory) -> Iterable[str]:
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
            path = os.path.join(dirpath, filename)
            if path.endswith('.py') and not path.endswith('_pb2.py'):
                yield path


def github_set_status_indicator(repository_organization: str,
                                repository_name: str,
                                repository_access_token: str,
                                commit_id: str,
                                state: str,
                                description: str,
                                context: str,
                                target_url: Optional[str] = None) -> None:
    """Sets a commit status indicator on github.

    Args:
        repository_organization: The github organization or account that the
            repository that the commit lives under.
        repository_name: The name of the github repository (not including
            the account name) that the commit lives under.
        repository_access_token: An access token generated by a github account
            with permission to update commit statuses on the target repository.
        commit_id: A hash that identities the commit to set a status on.
        state: The state of the status indicator. Must be 'error', 'failure',
            'pending', or 'success'.
        description: A summary of why the state is what it is,
            e.g. '5 lint errors' or 'tests passed!'.
        context: The name of the status indicator, e.g. 'pytest' or 'coverage'.
        target_url: Optional location where additional details about the status
            can be found, e.g. an online test results page.

    Raises:
        ValueError: Not one of the allowed states.
        IOError: The HTTP post request failed, or the response didn't have a
            201 code indicating success in the expected way.
    """
    if state not in ['error', 'failure', 'pending', 'success']:
        raise ValueError('Unrecognized state: {!r}'.format(state))

    payload = {
        'state': state,
        'description': description,
        'context': context,
    }
    if target_url is not None:
        payload['target_url'] = target_url

    url = ("https://api.github.com/repos/{}/{}/statuses/{}?access_token={}"
           .format(repository_organization,
                   repository_name,
                   commit_id,
                   repository_access_token))

    response = requests.post(url, json=payload)

    if response.status_code != 201:
        raise IOError('Request failed. Code: {}. Content: {}.'.format(
            response.status_code, response.content))


def git_fetch_for_comparison(remote: str,
                             actual_branch: str,
                             compare_branch: str) -> PreparedEnv:
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
    actual_id = None
    base_id = None
    for depth in [10, 100, 1000, None]:
        depth_str = '' if depth is None else '--depth={}'.format(depth)

        sub_run('git', 'fetch', remote, actual_branch, depth_str)
        actual_id = output_of('git', 'rev-parse', 'FETCH_HEAD')

        sub_run('git', 'fetch', remote, compare_branch, depth_str)
        base_id = output_of('git', 'rev-parse', 'FETCH_HEAD')

        try:
            common_ancestor_id = output_of(
                'git',
                'merge-base',
                actual_id,
                base_id)
            return PreparedEnv(None, actual_id, common_ancestor_id, None, None)
        except subprocess.CalledProcessError:
            # No common ancestor. We need to dig deeper.
            pass

    return PreparedEnv(None, actual_id, base_id, None, None)


def fetch_github_pull_request(destination_directory: str,
                              repository: GithubRepository,
                              pull_request_number: int) -> PreparedEnv:
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

    sub_run('git', 'init', redirect_stdout_into_stderr=True)
    result = git_fetch_for_comparison(remote=repository.as_remote(),
                                      actual_branch=branch,
                                      compare_branch='master')
    sub_run('git', 'branch', 'compare_commit', result.compare_commit_id)
    sub_run('git', 'checkout', '-b', 'actual_commit', result.actual_commit_id)
    return PreparedEnv(repository=repository,
                       actual_commit_id=result.actual_commit_id,
                       compare_commit_id=result.compare_commit_id,
                       destination_directory=destination_directory,
                       virtual_env_path=None)


def fetch_local_files(destination_directory: str) -> PreparedEnv:
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

    sub_run('git',
            'commit',
            '-a',
            '-m', 'working changes',
            '--allow-empty',
            redirect_stdout_into_stderr=True)
    sub_run('find',
            '|', 'grep', '\\.pyc$',
            '|', 'xargs', 'rm', '-f',
            shell=True)
    sub_run('find',
            '|', 'grep', '__pycache__',
            '|', 'xargs', 'rmdir',
            shell=True)
    commit_id = output_of('git', 'rev-parse', 'HEAD')
    compare_id = output_of('git', 'merge-base', commit_id, 'master')
    return PreparedEnv(repository=None,
                       actual_commit_id=commit_id,
                       compare_commit_id=compare_id,
                       destination_directory=destination_directory,
                       virtual_env_path=None)


def create_virtual_env(python_path, env_path):
    sub_run('virtualenv', '-p', python_path, env_path,
            redirect_stdout_into_stderr=True)
    pip_path = os.path.join(env_path, 'bin', 'pip')
    sub_run(pip_path, 'install', '-r', 'requirements.txt',
            redirect_stdout_into_stderr=True)


def prepare_temporary_test_environment(
        destination_directory: str,
        repository: GithubRepository,
        pull_request_number: Optional[int],
        env_name: str = '.test_virtualenv',
        python_path: str = '/usr/bin/python3.5',
        commit_ids_known_callback: Callable[[PreparedEnv], None] = None
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
        env_name: The name to use for the virtual environment.
        python_path: Location of the python binary to use within the
            virtual environment.
        commit_ids_known_callback: A method to call when the actual commit id
            being tested is known, before the virtual environment is ready.

    Returns:
        Commit ids corresponding to content to test/compare.
    """
    # Fetch content.
    if pull_request_number is not None:
        env = fetch_github_pull_request(
            destination_directory=destination_directory,
            repository=repository,
            pull_request_number=pull_request_number)
    else:
        env = fetch_local_files(
            destination_directory=destination_directory)

    if commit_ids_known_callback is not None:
        commit_ids_known_callback(env)

    # Create virtual environment.
    env_path = os.path.join(env.destination_directory, env_name)
    create_virtual_env(env_path=env_path,
                       python_path=python_path)

    return PreparedEnv(repository=env.repository,
                       actual_commit_id=env.actual_commit_id,
                       compare_commit_id=env.compare_commit_id,
                       destination_directory=env.destination_directory,
                       virtual_env_path=env_path)


def derive_temporary_python2_environment(
        destination_directory: str,
        python3_environment: PreparedEnv,
        env_name: str = '.test_virtualenv_py2',
        python_path: str = "/usr/bin/python2.7") -> PreparedEnv:

    shutil.rmtree(destination_directory)
    os.chdir(python3_environment.destination_directory)
    conversion_script_path = os.path.join(
        python3_environment.destination_directory,
        'python2.7-generate.sh')
    sub_run('bash',
            conversion_script_path,
            destination_directory,
            redirect_stdout_into_stderr=True)
    os.chdir(destination_directory)

    # Create virtual environment.
    env_path = os.path.join(destination_directory, env_name)
    create_virtual_env(env_path=env_path, python_path=python_path)

    return PreparedEnv(repository=python3_environment.repository,
                       actual_commit_id=python3_environment.actual_commit_id,
                       compare_commit_id=python3_environment.compare_commit_id,
                       destination_directory=destination_directory,
                       virtual_env_path=env_path)
