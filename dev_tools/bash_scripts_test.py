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
from typing import TYPE_CHECKING, Iterable

from dev_tools import shell_tools

if TYPE_CHECKING:
    import _pytest


def only_on_posix(func):
    if os.name != 'posix':
        return None
    return func


def run(
        *,
        script_file: str,
        tmpdir_factory: '_pytest.tmpdir.TempdirFactory',
        arg: str = '',
        setup: str = '',
        additional_intercepts: Iterable[str] = (),
) -> shell_tools.CommandOutput:
    """Invokes the given script within a temporary test environment."""

    with open(script_file) as f:
        script_lines = f.readlines()

    # Create a unique temporary directory
    dir_path = tmpdir_factory.mktemp('tmp', numbered=True)
    file_path = os.path.join(dir_path, 'test-script.sh')

    intercepted = [
        'python',
        'python3',
        'pylint',
        'pytest',
        'mypy',
        'yapf',
        *additional_intercepts,
    ]
    assert script_lines[0] == '#!/usr/bin/env bash\n'
    for e in intercepted:
        script_lines.insert(1, e + '() {\n  echo INTERCEPTED ' + e + ' $@\n}\n')

    with open(file_path, 'w') as f:
        f.writelines(script_lines)

    cmd = r"""
dir=$(git rev-parse --show-toplevel)
cd {}
git init --quiet
git commit -m init --allow-empty --quiet --no-gpg-sign
{}
chmod +x ./test-script.sh
./test-script.sh {}
""".format(dir_path, setup, arg)
    return shell_tools.run_shell(cmd=cmd,
                                 log_run_to_stderr=False,
                                 raise_on_fail=False,
                                 out=shell_tools.TeeCapture(),
                                 err=shell_tools.TeeCapture())


@only_on_posix
def test_pytest_changed_files_file_selection(tmpdir_factory):

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD~1',
                 setup='touch file.py\n'
                 'git add -A\n'
                 'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err.split() == (
        "Comparing against revision 'HEAD~1'.\n"
        "Found 0 test files associated with changes.\n").split()

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD~1',
                 setup='touch file_test.py\n'
                 'git add -A\n'
                 'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == 'INTERCEPTED pytest file_test.py\n'
    assert result.err.split() == (
        "Comparing against revision 'HEAD~1'.\n"
        "Found 1 test files associated with changes.\n").split()

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD~1',
                 setup='touch file.py file_test.py\n'
                 'git add -A\n'
                 'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == 'INTERCEPTED pytest file_test.py\n'
    assert result.err.split() == (
        "Comparing against revision 'HEAD~1'.\n"
        "Found 1 test files associated with changes.\n").split()

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD',
                 setup='touch file.py file_test.py\n'
                 'git add -A\n'
                 'git commit -m test --quiet --no-gpg-sign\n'
                 'echo x > file_test.py\n')
    assert result.exit_code == 0
    assert result.out == 'INTERCEPTED pytest file_test.py\n'
    assert result.err.split() == (
        "Comparing against revision 'HEAD'.\n"
        "Found 1 test files associated with changes.\n").split()

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD',
                 setup='touch file.py file_test.py\n'
                 'git add -A\n'
                 'git commit -m test --quiet --no-gpg-sign\n'
                 'echo x > file.py\n')
    assert result.exit_code == 0
    assert result.out == 'INTERCEPTED pytest file_test.py\n'
    assert result.err.split() == (
        "Comparing against revision 'HEAD'.\n"
        "Found 1 test files associated with changes.\n").split()

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD',
                 setup='touch __init__.py\n'
                 'git add -A\n'
                 'git commit -m test --quiet --no-gpg-sign\n'
                 'echo x > __init__.py\n')
    assert result.exit_code == 0
    assert result.out == ('INTERCEPTED pytest docs/docs_coverage_test.py '
                          'cirq/protocols/json_serialization_test.py\n')
    assert result.err.split() == (
        "Comparing against revision 'HEAD'.\n"
        "Found 2 test files associated with changes.\n").split()


@only_on_posix
def test_pytest_changed_files_branch_selection(tmpdir_factory):

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err.split() == (
        "Comparing against revision 'HEAD'.\n"
        "Found 0 test files associated with changes.\n").split()

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD~999999')
    assert result.exit_code == 1
    assert result.out == ''
    assert "No revision 'HEAD~999999'." in result.err

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory)
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err.split() == (
        "Comparing against revision 'master'.\n"
        "Found 0 test files associated with changes.\n").split()

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 setup='git branch origin/master')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err.split() == (
        "Comparing against revision 'origin/master'.\n"
        "Found 0 test files associated with changes.\n").split()

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 setup='git branch upstream/master')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err.split() == (
        "Comparing against revision 'upstream/master'.\n"
        "Found 0 test files associated with changes.\n").split()

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 setup='git branch upstream/master; git branch origin/master')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err.split() == (
        "Comparing against revision 'upstream/master'.\n"
        "Found 0 test files associated with changes.\n").split()

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 arg='file',
                 setup='git checkout -b other --quiet\n'
                 'git branch -D master --quiet\n')
    assert result.exit_code == 1
    assert result.out == ''
    assert "No revision 'file'." in result.err

    # Fails on file.
    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 arg='file',
                 setup='touch file\n'
                 'git add -A\n'
                 'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 1
    assert result.out == ''
    assert "No revision 'file'." in result.err

    # Works when ambiguous between revision and file.
    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD',
                 setup='touch HEAD\n'
                 'git add -A\n'
                 'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err.split() == (
        "Comparing against revision 'HEAD'.\n"
        "Found 0 test files associated with changes.\n").split()

    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 setup='touch master\n'
                 'git add -A\n'
                 'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err.split() == (
        "Comparing against revision 'master'.\n"
        "Found 0 test files associated with changes.\n").split()

    # Works on remotes.
    result = run(script_file='check/pytest-changed-files',
                 tmpdir_factory=tmpdir_factory,
                 setup='mkdir alt\n'
                 'cd alt\n'
                 'git init --quiet\n'
                 'git commit -m tes --quiet --allow-empty --no-gpg-sign\n'
                 'cd ..\n'
                 'git remote add origin alt\n'
                 'git fetch origin master --quiet 2> /dev/null\n')
    assert result.exit_code == 0
    assert result.out == ''
    assert result.err.split() == (
        "Comparing against revision 'origin/master'.\n"
        "Found 0 test files associated with changes.\n").split()


@only_on_posix
def test_pytest_and_incremental_coverage_branch_selection(tmpdir_factory):
    result = run(script_file='check/pytest-and-incremental-coverage',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD',
                 additional_intercepts=['check/pytest'])
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED check/pytest '
        '. --actually-quiet --cov --cov-report=annotate '
        '--cov-config=dev_tools/conf/.coveragerc --benchmark-skip\n'
        'INTERCEPTED '
        'python dev_tools/check_incremental_coverage_annotations.py HEAD\n')
    assert result.err == "Comparing against revision 'HEAD'.\n"

    result = run(script_file='check/pytest-and-incremental-coverage',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD~999999')
    assert result.exit_code == 1
    assert result.out == ''
    assert "No revision 'HEAD~999999'." in result.err

    result = run(script_file='check/pytest-and-incremental-coverage',
                 tmpdir_factory=tmpdir_factory,
                 additional_intercepts=['check/pytest'])
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED check/pytest '
        '. --actually-quiet --cov --cov-report=annotate '
        '--cov-config=dev_tools/conf/.coveragerc --benchmark-skip\n'
        'INTERCEPTED python '
        'dev_tools/check_incremental_coverage_annotations.py master\n')
    assert result.err == "Comparing against revision 'master'.\n"

    result = run(script_file='check/pytest-and-incremental-coverage',
                 tmpdir_factory=tmpdir_factory,
                 setup='git branch origin/master',
                 additional_intercepts=['check/pytest'])
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED check/pytest '
        '. --actually-quiet --cov --cov-report=annotate '
        '--cov-config=dev_tools/conf/.coveragerc --benchmark-skip\n'
        'INTERCEPTED python '
        'dev_tools/check_incremental_coverage_annotations.py origin/master\n')
    assert result.err == "Comparing against revision 'origin/master'.\n"

    result = run(script_file='check/pytest-and-incremental-coverage',
                 tmpdir_factory=tmpdir_factory,
                 setup='git branch upstream/master',
                 additional_intercepts=['check/pytest'])
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED check/pytest '
        '. --actually-quiet --cov --cov-report=annotate '
        '--cov-config=dev_tools/conf/.coveragerc --benchmark-skip\n'
        'INTERCEPTED python '
        'dev_tools/check_incremental_coverage_annotations.py upstream/master\n')
    assert result.err == "Comparing against revision 'upstream/master'.\n"

    result = run(script_file='check/pytest-and-incremental-coverage',
                 tmpdir_factory=tmpdir_factory,
                 setup='git branch upstream/master; git branch origin/master',
                 additional_intercepts=['check/pytest'])
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED check/pytest '
        '. --actually-quiet --cov --cov-report=annotate '
        '--cov-config=dev_tools/conf/.coveragerc --benchmark-skip\n'
        'INTERCEPTED python '
        'dev_tools/check_incremental_coverage_annotations.py upstream/master\n')
    assert result.err == "Comparing against revision 'upstream/master'.\n"

    result = run(script_file='check/pytest-and-incremental-coverage',
                 tmpdir_factory=tmpdir_factory,
                 setup='git checkout -b other --quiet\n'
                 'git branch -D master --quiet\n',
                 additional_intercepts=['check/pytest'])
    assert result.exit_code == 1
    assert result.out == ''
    assert 'No default revision found to compare against' in result.err

    # Works when ambiguous between revision and file.
    result = run(script_file='check/pytest-and-incremental-coverage',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD',
                 setup='touch HEAD\n'
                 'git add -A\n'
                 'git commit -m test --quiet --no-gpg-sign\n',
                 additional_intercepts=['check/pytest'])
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED check/pytest '
        '. --actually-quiet --cov --cov-report=annotate '
        '--cov-config=dev_tools/conf/.coveragerc --benchmark-skip\n'
        'INTERCEPTED python '
        'dev_tools/check_incremental_coverage_annotations.py HEAD\n')
    assert result.err == "Comparing against revision 'HEAD'.\n"

    result = run(script_file='check/pytest-and-incremental-coverage',
                 tmpdir_factory=tmpdir_factory,
                 setup='touch master\n'
                 'git add -A\n'
                 'git commit -m test --quiet --no-gpg-sign\n',
                 additional_intercepts=['check/pytest'])
    assert result.exit_code == 0
    assert result.out == (
        'INTERCEPTED check/pytest '
        '. --actually-quiet --cov --cov-report=annotate '
        '--cov-config=dev_tools/conf/.coveragerc --benchmark-skip\n'
        'INTERCEPTED python '
        'dev_tools/check_incremental_coverage_annotations.py master\n')
    assert result.err == "Comparing against revision 'master'.\n"

    result = run(script_file='check/pytest-and-incremental-coverage',
                 tmpdir_factory=tmpdir_factory,
                 setup='touch master\n'
                 'git add -A\n'
                 'git commit -q -m test --no-gpg-sign\n'
                 'git branch alt\n'
                 'touch master2\n'
                 'git add -A\n'
                 'git commit -q -m test2 --no-gpg-sign\n'
                 'git checkout -q alt\n',
                 additional_intercepts=['check/pytest'])
    assert result.exit_code == 0
    assert result.out.startswith(
        'INTERCEPTED check/pytest '
        '. --actually-quiet --cov --cov-report=annotate '
        '--cov-config=dev_tools/conf/.coveragerc --benchmark-skip\n'
        'INTERCEPTED python '
        'dev_tools/check_incremental_coverage_annotations.py ')
    assert result.err.startswith(
        "Comparing against revision 'master' (merge base ")


@only_on_posix
def test_incremental_format_branch_selection(tmpdir_factory):
    result = run(script_file='check/format-incremental',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD')
    assert result.exit_code == 0
    assert result.out == (
        '\x1b[32mNo formatting needed on changed lines\x1b[0m.\n')
    assert result.err == "Comparing against revision 'HEAD'.\n"

    result = run(script_file='check/format-incremental',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD~9999')
    assert result.exit_code == 1
    assert result.out == ''
    assert "No revision 'HEAD~9999'." in result.err

    result = run(script_file='check/format-incremental',
                 tmpdir_factory=tmpdir_factory)
    assert result.exit_code == 0
    assert result.out == (
        '\x1b[32mNo formatting needed on changed lines\x1b[0m.\n')
    assert result.err == "Comparing against revision 'master'.\n"

    result = run(script_file='check/format-incremental',
                 tmpdir_factory=tmpdir_factory,
                 setup='git branch origin/master')
    assert result.exit_code == 0
    assert result.out == (
        '\x1b[32mNo formatting needed on changed lines\x1b[0m.\n')
    assert result.err == "Comparing against revision 'origin/master'.\n"

    result = run(script_file='check/format-incremental',
                 tmpdir_factory=tmpdir_factory,
                 setup='git branch upstream/master')
    assert result.exit_code == 0
    assert result.out == (
        '\x1b[32mNo formatting needed on changed lines\x1b[0m.\n')
    assert result.err == "Comparing against revision 'upstream/master'.\n"

    result = run(script_file='check/format-incremental',
                 tmpdir_factory=tmpdir_factory,
                 setup='git branch upstream/master; git branch origin/master')
    assert result.exit_code == 0
    assert result.out == (
        '\x1b[32mNo formatting needed on changed lines\x1b[0m.\n')
    assert result.err == "Comparing against revision 'upstream/master'.\n"

    result = run(script_file='check/format-incremental',
                 tmpdir_factory=tmpdir_factory,
                 setup='git checkout -b other --quiet\n'
                 'git branch -D master --quiet\n')
    assert result.exit_code == 1
    assert result.out == ''
    assert 'No default revision found to compare against' in result.err

    # Works when ambiguous between revision and file.
    result = run(script_file='check/format-incremental',
                 tmpdir_factory=tmpdir_factory,
                 arg='HEAD',
                 setup='touch HEAD.py\n'
                 'git add -A\n'
                 'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == (
        '\x1b[32mNo formatting needed on changed lines\x1b[0m.\n')
    assert result.err == "Comparing against revision 'HEAD'.\n"

    result = run(script_file='check/format-incremental',
                 tmpdir_factory=tmpdir_factory,
                 setup='touch master.py\n'
                 'git add -A\n'
                 'git commit -m test --quiet --no-gpg-sign\n')
    assert result.exit_code == 0
    assert result.out == (
        '\x1b[32mNo formatting needed on changed lines\x1b[0m.\n')
    assert result.err == "Comparing against revision 'master'.\n"

    result = run(script_file='check/format-incremental',
                 tmpdir_factory=tmpdir_factory,
                 setup='touch master.py\n'
                 'git add -A\n'
                 'git commit -q -m test --no-gpg-sign\n'
                 'git branch alt\n'
                 'touch master2.py\n'
                 'git add -A\n'
                 'git commit -q -m test2 --no-gpg-sign\n'
                 'git checkout -q alt\n'
                 'touch alt.py\n'
                 'git add -A\n'
                 'git commit -q -m test3 --no-gpg-sign\n')
    assert result.exit_code == 1
    assert result.out == (
        '\n'
        '\x1b[31mChanges in alt.py require formatting:\x1b[0m\n'
        'INTERCEPTED yapf --style=google --diff alt.py\n'
        '\x1b[31mSome formatting needed on changed lines\x1b[0m.\n')
    assert result.err.startswith(
        "Comparing against revision 'master' (merge base ")
