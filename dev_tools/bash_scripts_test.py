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

import sys

import os

import cirq
from dev_tools import shell_tools


def only_in_python3_on_posix(func):
    if os.name != 'posix':
        return None
    if sys.version_info.major < 3:
        return None
    return func


def run(*, script_file: str, arg: str ='', setup: str = ''
        ) -> shell_tools.CommandOutput:
    """Invokes the given script within a temporary test environment."""

    with open(script_file) as f:
        script_lines = f.readlines()

    intercepted = [
        'python',
        'python2',
        'python3',
        'pylint',
        'pytest',
        'mypy',
    ]
    assert script_lines[0] == '#!/usr/bin/env bash\n'
    for e in intercepted:
        script_lines.insert(1, e + '() {\n  echo INTERCEPTED ' + e + ' $@\n}\n')

    with cirq.testing.TempDirectoryPath() as dir_path:
        with open(os.path.join(dir_path, 'test-script'), 'w') as f:
            f.writelines(script_lines)

        cmd = r"""
dir=$(git rev-parse --show-toplevel)
cd {}
git init --quiet
git commit -m init --allow-empty --quiet --no-gpg-sign
{}
chmod +x ./test-script
./test-script {}
""".format(dir_path, setup, arg)
        return shell_tools.run_shell(
            cmd=cmd,
            log_run_to_stderr=False,
            raise_on_fail=False,
            out=shell_tools.TeeCapture(),
            err=shell_tools.TeeCapture())


@only_in_python3_on_posix
def test_pytest_changed_files_file_selection():

    result = run(script_file='check/pytest-changed-files',
                 arg='HEAD~1',
                 setup='touch file.py\n'
                       'git add -A\n'
                       'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err == ("Comparing against revision 'HEAD~1'.\n"
                          "Found 0 differing files with associated tests.\n")

    result = run(script_file='check/pytest-changed-files',
                 arg='HEAD~1',
                 setup='touch file_test.py\n'
                       'git add -A\n'
                       'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == 'INTERCEPTED pytest file_test.py\n'
    assert result.err == ("Comparing against revision 'HEAD~1'.\n"
                          "Found 1 differing files with associated tests.\n")

    result = run(script_file='check/pytest-changed-files',
                 arg='HEAD~1',
                 setup='touch file.py file_test.py\n'
                       'git add -A\n'
                       'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == 'INTERCEPTED pytest file_test.py\n'
    assert result.err == ("Comparing against revision 'HEAD~1'.\n"
                          "Found 1 differing files with associated tests.\n")

    result = run(script_file='check/pytest-changed-files',
                 arg='HEAD',
                 setup='touch file.py file_test.py\n'
                       'git add -A\n'
                       'git commit -m test --quiet --no-gpg-sign\n'
                       'echo x > file_test.py\n')
    assert result.exit_code == 0
    assert result.out == 'INTERCEPTED pytest file_test.py\n'
    assert result.err == ("Comparing against revision 'HEAD'.\n"
                          "Found 1 differing files with associated tests.\n")

    result = run(script_file='check/pytest-changed-files',
                 arg='HEAD',
                 setup='touch file.py file_test.py\n'
                       'git add -A\n'
                       'git commit -m test --quiet --no-gpg-sign\n'
                       'echo x > file.py\n')
    assert result.exit_code == 0
    assert result.out == 'INTERCEPTED pytest file_test.py\n'
    assert result.err == ("Comparing against revision 'HEAD'.\n"
                          "Found 1 differing files with associated tests.\n")


@only_in_python3_on_posix
def test_pytest_changed_files_branch_selection():

    result = run(script_file='check/pytest-changed-files', arg='HEAD')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err == ("Comparing against revision 'HEAD'.\n"
                          "Found 0 differing files with associated tests.\n")

    result = run(script_file='check/pytest-changed-files', arg='HEAD~9999')
    assert result.exit_code == 1
    assert result.out == ''
    assert "No revision 'HEAD~9999'." in result.err

    result = run(script_file='check/pytest-changed-files')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err == ("Comparing against revision 'master'.\n"
                          "Found 0 differing files with associated tests.\n")

    result = run(script_file='check/pytest-changed-files',
                 setup='git branch origin/master')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err == ("Comparing against revision 'origin/master'.\n"
                          "Found 0 differing files with associated tests.\n")

    result = run(script_file='check/pytest-changed-files',
                 setup='git branch upstream/master')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err == ("Comparing against revision 'upstream/master'.\n"
                          "Found 0 differing files with associated tests.\n")

    result = run(script_file='check/pytest-changed-files',
                 setup='git branch upstream/master; git branch origin/master')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err == ("Comparing against revision 'upstream/master'.\n"
                          "Found 0 differing files with associated tests.\n")

    result = run(script_file='check/pytest-changed-files',
                 arg='file',
                 setup='git checkout -b other --quiet\n'
                       'git branch -D master --quiet\n')
    assert result.exit_code == 1
    assert result.out == ''
    assert "No revision 'file'." in result.err

    # Fails on file.
    result = run(script_file='check/pytest-changed-files',
                 arg='file',
                 setup='touch file\n'
                       'git add -A\n'
                       'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 1
    assert result.out == ''
    assert "No revision 'file'." in result.err

    # Works when ambiguous between revision and file.
    result = run(script_file='check/pytest-changed-files',
                 arg='HEAD',
                 setup='touch HEAD\n'
                       'git add -A\n'
                       'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err == ("Comparing against revision 'HEAD'.\n"
                          "Found 0 differing files with associated tests.\n")

    result = run(script_file='check/pytest-changed-files',
                 setup='touch master\n'
                       'git add -A\n'
                       'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err == ("Comparing against revision 'master'.\n"
                          "Found 0 differing files with associated tests.\n")

    # Works on remotes.
    result = run(script_file='check/pytest-changed-files',
                 setup='mkdir alt\n'
                       'cd alt\n'
                       'git init --quiet\n'
                       'git commit -m tes --quiet --allow-empty --no-gpg-sign\n'
                       'cd ..\n'
                       'git remote add origin alt\n'
                       'git fetch origin master --quiet 2> /dev/null\n')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err == ("Comparing against revision 'origin/master'.\n"
                          "Found 0 differing files with associated tests.\n")


@only_in_python3_on_posix
def test_pytest_and_incremental_coverage_branch_selection():

    result = run(script_file='check/pytest-and-incremental-coverage',
                 arg='HEAD')
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED python '
        'dev_tools/run_pytest_and_incremental_coverage.py HEAD\n')
    assert result.err == "Comparing against revision 'HEAD'.\n"

    result = run(script_file='check/pytest-and-incremental-coverage',
                 arg='HEAD~9999')
    assert result.exit_code == 1
    assert result.out == ''
    assert "No revision 'HEAD~9999'." in result.err

    result = run(script_file='check/pytest-and-incremental-coverage')
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED python '
        'dev_tools/run_pytest_and_incremental_coverage.py master\n')
    assert result.err == "Comparing against revision 'master'.\n"

    result = run(script_file='check/pytest-and-incremental-coverage',
                 setup='git branch origin/master')
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED python '
        'dev_tools/run_pytest_and_incremental_coverage.py origin/master\n')
    assert result.err == "Comparing against revision 'origin/master'.\n"

    result = run(script_file='check/pytest-and-incremental-coverage',
                 setup='git branch upstream/master')
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED python '
        'dev_tools/run_pytest_and_incremental_coverage.py upstream/master\n')
    assert result.err == "Comparing against revision 'upstream/master'.\n"

    result = run(script_file='check/pytest-and-incremental-coverage',
                 setup='git branch upstream/master; git branch origin/master')
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED python '
        'dev_tools/run_pytest_and_incremental_coverage.py upstream/master\n')
    assert result.err == "Comparing against revision 'upstream/master'.\n"

    result = run(script_file='check/pytest-and-incremental-coverage',
                 setup='git checkout -b other --quiet\n'
                       'git branch -D master --quiet\n')
    assert result.exit_code == 1
    assert result.out == ''
    assert 'No default revision found to compare against' in result.err

    # Works when ambiguous between revision and file.
    result = run(script_file='check/pytest-and-incremental-coverage',
                 arg='HEAD',
                 setup='touch HEAD\n'
                       'git add -A\n'
                       'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED python '
        'dev_tools/run_pytest_and_incremental_coverage.py HEAD\n')
    assert result.err == "Comparing against revision 'HEAD'.\n"

    result = run(script_file='check/pytest-and-incremental-coverage',
                 setup='touch master\n'
                       'git add -A\n'
                       'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED python '
        'dev_tools/run_pytest_and_incremental_coverage.py master\n')
    assert result.err == "Comparing against revision 'master'.\n"
