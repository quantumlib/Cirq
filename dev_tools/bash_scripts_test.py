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
import subprocess
from typing import Iterable

import pytest

from dev_tools import shell_tools
from dev_tools.test_utils import only_on_posix


def run(
    *,
    script_file: str,
    tmpdir_factory: pytest.TempdirFactory,
    arg: str = '',
    setup: str = '',
    additional_intercepts: Iterable[str] = (),
) -> subprocess.CompletedProcess:
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
        'env',
        'pytest',
        'mypy',
        'black',
        *additional_intercepts,
    ]
    assert script_lines[0] == '#!/usr/bin/env bash\n'
    for e in intercepted:
        script_lines.insert(1, e + '() {\n  echo INTERCEPTED ' + e + ' $@\n}\n')

    with open(file_path, 'w') as f:
        f.writelines(script_lines)

    cmd = f"""
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_SYSTEM=/dev/null
dir=$(git rev-parse --show-toplevel)
cd {dir_path}
git init --quiet --initial-branch master
git config --local user.name 'Me'
git config --local user.email '<>'
git commit -m init --allow-empty --quiet --no-gpg-sign
{setup}
mkdir -p dev_tools
touch dev_tools/pypath
chmod +x ./test-script.sh
./test-script.sh {arg}
"""
    return shell_tools.run(
        cmd, log_run_to_stderr=False, shell=True, check=False, capture_output=True
    )


@only_on_posix
def test_pytest_changed_files_file_selection(tmpdir_factory):
    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD~1',
        setup='touch file.py\ngit add -A\ngit commit -m test --quiet --no-gpg-sign\n',
    )
    assert result.returncode == 0
    assert result.stdout == ''
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'HEAD~1'.\nFound 0 test files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD~1',
        setup='touch file_test.py\ngit add -A\ngit commit -m test --quiet --no-gpg-sign\n',
    )
    assert result.returncode == 0
    assert result.stdout == 'INTERCEPTED pytest file_test.py\n'
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'HEAD~1'.\nFound 1 test files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD~1',
        setup='touch file.py file_test.py\n'
        'git add -A\n'
        'git commit -m test --quiet --no-gpg-sign\n',
    )
    assert result.returncode == 0
    assert result.stdout == 'INTERCEPTED pytest file_test.py\n'
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'HEAD~1'.\nFound 1 test files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD',
        setup='touch file.py file_test.py\n'
        'git add -A\n'
        'git commit -m test --quiet --no-gpg-sign\n'
        'echo x > file_test.py\n',
    )
    assert result.returncode == 0
    assert result.stdout == 'INTERCEPTED pytest file_test.py\n'
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'HEAD'.\nFound 1 test files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD',
        setup='touch file.py file_test.py\n'
        'git add -A\n'
        'git commit -m test --quiet --no-gpg-sign\n'
        'echo x > file.py\n',
    )
    assert result.returncode == 0
    assert result.stdout == 'INTERCEPTED pytest file_test.py\n'
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'HEAD'.\nFound 1 test files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD',
        setup='touch __init__.py\n'
        'git add -A\n'
        'git commit -m test --quiet --no-gpg-sign\n'
        'echo x > __init__.py\n',
    )
    assert result.returncode == 0
    assert result.stdout == (
        'INTERCEPTED pytest cirq-core/cirq/protocols/json_serialization_test.py\n'
    )
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'HEAD'.\nFound 1 test files associated with changes.\n"
        ).split()
    )


@only_on_posix
def test_pytest_changed_files_branch_selection(tmpdir_factory):
    result = run(
        script_file='check/pytest-changed-files', tmpdir_factory=tmpdir_factory, arg='HEAD'
    )
    assert result.returncode == 0
    assert result.stdout == ''
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'HEAD'.\nFound 0 test files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pytest-changed-files', tmpdir_factory=tmpdir_factory, arg='HEAD~999999'
    )
    assert result.returncode == 1
    assert result.stdout == ''
    assert "No revision 'HEAD~999999'." in result.stderr

    result = run(script_file='check/pytest-changed-files', tmpdir_factory=tmpdir_factory)
    assert result.returncode == 0
    assert result.stdout == ''
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'master'.\nFound 0 test files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        setup='git branch origin/master',
    )
    assert result.returncode == 0
    assert result.stdout == ''
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'origin/master'.\n"
            "Found 0 test files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        setup='git branch upstream/master',
    )
    assert result.returncode == 0
    assert result.stdout == ''
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'upstream/master'.\n"
            "Found 0 test files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        setup='git branch upstream/master; git branch origin/master',
    )
    assert result.returncode == 0
    assert result.stdout == ''
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'upstream/master'.\n"
            "Found 0 test files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='file',
        setup='git checkout -b other --quiet\ngit branch -D master --quiet\n',
    )
    assert result.returncode == 1
    assert result.stdout == ''
    assert "No revision 'file'." in result.stderr

    # Fails on file.
    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='file',
        setup='touch file\ngit add -A\ngit commit -m test --quiet --no-gpg-sign\n',
    )
    assert result.returncode == 1
    assert result.stdout == ''
    assert "No revision 'file'." in result.stderr

    # Works when ambiguous between revision and file.
    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD',
        setup='touch HEAD\ngit add -A\ngit commit -m test --quiet --no-gpg-sign\n',
    )
    assert result.returncode == 0
    assert result.stdout == ''
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'HEAD'.\nFound 0 test files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        setup='touch master\ngit add -A\ngit commit -m test --quiet --no-gpg-sign\n',
    )
    assert result.returncode == 0
    assert result.stdout == ''
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'master'.\nFound 0 test files associated with changes.\n"
        ).split()
    )

    # Works on remotes.
    result = run(
        script_file='check/pytest-changed-files',
        tmpdir_factory=tmpdir_factory,
        setup='mkdir alt\n'
        'cd alt\n'
        'git init --quiet --initial-branch master\n'
        'git config --local user.name \'Me\'\n'
        'git config --local user.email \'<>\'\n'
        'git commit -m tes --quiet --allow-empty --no-gpg-sign\n'
        'cd ..\n'
        'git remote add origin alt\n'
        'git fetch origin master --quiet 2> /dev/null\n',
    )
    assert result.returncode == 0
    assert result.stdout == ''
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'origin/master'.\n"
            "Found 0 test files associated with changes.\n"
        ).split()
    )


@only_on_posix
def test_pytest_and_incremental_coverage_branch_selection(tmpdir_factory):
    result = run(
        script_file='check/pytest-and-incremental-coverage',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD',
        additional_intercepts=['check/pytest'],
    )
    assert result.returncode == 0
    assert result.stdout == (
        'INTERCEPTED check/pytest '
        '--cov --cov-config=dev_tools/conf/.coveragerc\n'
        'The annotate command will be removed in a future version.\n'
        'Get in touch if you still use it: ned@nedbatchelder.com\n'
        'No data to report.\n'
        'INTERCEPTED '
        'python dev_tools/check_incremental_coverage_annotations.py HEAD\n'
    )
    assert result.stderr == "Comparing against revision 'HEAD'.\n"

    result = run(
        script_file='check/pytest-and-incremental-coverage',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD~999999',
    )
    assert result.returncode == 1
    assert result.stdout == ''
    assert "No revision 'HEAD~999999'." in result.stderr

    result = run(
        script_file='check/pytest-and-incremental-coverage',
        tmpdir_factory=tmpdir_factory,
        additional_intercepts=['check/pytest'],
    )
    assert result.returncode == 0
    assert result.stdout == (
        'INTERCEPTED check/pytest '
        '--cov --cov-config=dev_tools/conf/.coveragerc\n'
        'The annotate command will be removed in a future version.\n'
        'Get in touch if you still use it: ned@nedbatchelder.com\n'
        'No data to report.\n'
        'INTERCEPTED '
        'python dev_tools/check_incremental_coverage_annotations.py master\n'
    )
    assert result.stderr == "Comparing against revision 'master'.\n"

    result = run(
        script_file='check/pytest-and-incremental-coverage',
        tmpdir_factory=tmpdir_factory,
        setup='git branch origin/master',
        additional_intercepts=['check/pytest'],
    )
    assert result.returncode == 0
    assert result.stdout == (
        'INTERCEPTED check/pytest '
        '--cov --cov-config=dev_tools/conf/.coveragerc\n'
        'The annotate command will be removed in a future version.\n'
        'Get in touch if you still use it: ned@nedbatchelder.com\n'
        'No data to report.\n'
        'INTERCEPTED '
        'python dev_tools/check_incremental_coverage_annotations.py origin/master\n'
    )
    assert result.stderr == "Comparing against revision 'origin/master'.\n"

    result = run(
        script_file='check/pytest-and-incremental-coverage',
        tmpdir_factory=tmpdir_factory,
        setup='git branch upstream/master',
        additional_intercepts=['check/pytest'],
    )
    assert result.returncode == 0
    assert result.stdout == (
        'INTERCEPTED check/pytest '
        '--cov --cov-config=dev_tools/conf/.coveragerc\n'
        'The annotate command will be removed in a future version.\n'
        'Get in touch if you still use it: ned@nedbatchelder.com\n'
        'No data to report.\n'
        'INTERCEPTED '
        'python dev_tools/check_incremental_coverage_annotations.py upstream/master\n'
    )
    assert result.stderr == "Comparing against revision 'upstream/master'.\n"

    result = run(
        script_file='check/pytest-and-incremental-coverage',
        tmpdir_factory=tmpdir_factory,
        setup='git branch upstream/master; git branch origin/master',
        additional_intercepts=['check/pytest'],
    )
    assert result.returncode == 0
    assert result.stdout == (
        'INTERCEPTED check/pytest '
        '--cov --cov-config=dev_tools/conf/.coveragerc\n'
        'The annotate command will be removed in a future version.\n'
        'Get in touch if you still use it: ned@nedbatchelder.com\n'
        'No data to report.\n'
        'INTERCEPTED '
        'python dev_tools/check_incremental_coverage_annotations.py upstream/master\n'
    )
    assert result.stderr == "Comparing against revision 'upstream/master'.\n"

    result = run(
        script_file='check/pytest-and-incremental-coverage',
        tmpdir_factory=tmpdir_factory,
        setup='git checkout -b other --quiet\ngit branch -D master --quiet\n',
        additional_intercepts=['check/pytest'],
    )
    assert result.returncode == 1
    assert result.stdout == ''
    assert 'No default revision found to compare against' in result.stderr

    # Works when ambiguous between revision and file.
    result = run(
        script_file='check/pytest-and-incremental-coverage',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD',
        setup='touch HEAD\ngit add -A\ngit commit -m test --quiet --no-gpg-sign\n',
        additional_intercepts=['check/pytest'],
    )
    assert result.returncode == 0
    assert result.stdout == (
        'INTERCEPTED check/pytest '
        '--cov --cov-config=dev_tools/conf/.coveragerc\n'
        'The annotate command will be removed in a future version.\n'
        'Get in touch if you still use it: ned@nedbatchelder.com\n'
        'No data to report.\n'
        'INTERCEPTED '
        'python dev_tools/check_incremental_coverage_annotations.py HEAD\n'
    )
    assert result.stderr == "Comparing against revision 'HEAD'.\n"

    result = run(
        script_file='check/pytest-and-incremental-coverage',
        tmpdir_factory=tmpdir_factory,
        setup='touch master\ngit add -A\ngit commit -m test --quiet --no-gpg-sign\n',
        additional_intercepts=['check/pytest'],
    )
    assert result.returncode == 0
    assert result.stdout == (
        'INTERCEPTED check/pytest '
        '--cov --cov-config=dev_tools/conf/.coveragerc\n'
        'The annotate command will be removed in a future version.\n'
        'Get in touch if you still use it: ned@nedbatchelder.com\n'
        'No data to report.\n'
        'INTERCEPTED '
        'python dev_tools/check_incremental_coverage_annotations.py master\n'
    )
    assert result.stderr == "Comparing against revision 'master'.\n"

    result = run(
        script_file='check/pytest-and-incremental-coverage',
        tmpdir_factory=tmpdir_factory,
        setup='touch master\n'
        'git add -A\n'
        'git commit -q -m test --no-gpg-sign\n'
        'git branch alt\n'
        'touch master2\n'
        'git add -A\n'
        'git commit -q -m test2 --no-gpg-sign\n'
        'git checkout -q alt\n',
        additional_intercepts=['check/pytest'],
    )
    assert result.returncode == 0
    assert result.stdout.startswith(
        'INTERCEPTED check/pytest '
        '--cov --cov-config=dev_tools/conf/.coveragerc\n'
        'The annotate command will be removed in a future version.\n'
        'Get in touch if you still use it: ned@nedbatchelder.com\n'
        'No data to report.\n'
        'INTERCEPTED '
        'python dev_tools/check_incremental_coverage_annotations.py '
    )
    assert result.stderr.startswith("Comparing against revision 'master' (merge base ")


@only_on_posix
def test_incremental_format_branch_selection(tmpdir_factory):
    result = run(script_file='check/format-incremental', tmpdir_factory=tmpdir_factory, arg='HEAD')
    assert result.returncode == 0
    assert "No files to format" in result.stdout
    assert "Comparing against revision 'HEAD'." in result.stderr

    result = run(
        script_file='check/format-incremental', tmpdir_factory=tmpdir_factory, arg='HEAD~9999'
    )
    assert result.returncode == 1
    assert result.stdout == ''
    assert "No revision 'HEAD~9999'." in result.stderr

    result = run(script_file='check/format-incremental', tmpdir_factory=tmpdir_factory)
    assert result.returncode == 0
    assert "No files to format" in result.stdout
    assert "Comparing against revision 'master'." in result.stderr

    result = run(
        script_file='check/format-incremental',
        tmpdir_factory=tmpdir_factory,
        setup='git branch origin/master',
    )
    assert result.returncode == 0
    assert "No files to format" in result.stdout
    assert "Comparing against revision 'origin/master'." in result.stderr

    result = run(
        script_file='check/format-incremental',
        tmpdir_factory=tmpdir_factory,
        setup='git branch upstream/master',
    )
    assert result.returncode == 0
    assert "No files to format" in result.stdout
    assert "Comparing against revision 'upstream/master'." in result.stderr

    result = run(
        script_file='check/format-incremental',
        tmpdir_factory=tmpdir_factory,
        setup='git branch upstream/master; git branch origin/master',
    )
    assert result.returncode == 0
    assert "No files to format" in result.stdout
    assert "Comparing against revision 'upstream/master'." in result.stderr

    result = run(
        script_file='check/format-incremental',
        tmpdir_factory=tmpdir_factory,
        setup='git checkout -b other --quiet\ngit branch -D master --quiet\n',
    )
    assert result.returncode == 1
    assert result.stdout == ''

    assert 'No default revision found to compare against' in result.stderr

    # Works when ambiguous between revision and file.
    result = run(
        script_file='check/format-incremental',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD',
        setup='touch HEAD.py\ngit add -A\ngit commit -m test --quiet --no-gpg-sign\n',
    )
    assert result.returncode == 0
    assert "No files to format" in result.stdout
    assert "Comparing against revision 'HEAD'." in result.stderr

    result = run(
        script_file='check/format-incremental',
        tmpdir_factory=tmpdir_factory,
        setup='touch master.py\ngit add -A\ngit commit -m test --quiet --no-gpg-sign\n',
    )
    assert result.returncode == 0
    assert "No files to format" in result.stdout
    assert "Comparing against revision 'master'." in result.stderr

    result = run(
        script_file='check/format-incremental',
        tmpdir_factory=tmpdir_factory,
        setup='touch master.py\n'
        'git add -A\n'
        'git commit -q -m test --no-gpg-sign\n'
        'git branch alt\n'
        'touch master2.py\n'
        'git add -A\n'
        'git commit -q -m test2 --no-gpg-sign\n'
        'git checkout -q alt\n'
        'echo " print(1)" > alt.py\n'
        'git add -A\n'
        'git commit -q -m test3 --no-gpg-sign\n',
    )
    assert result.returncode == 0
    assert 'INTERCEPTED black --color --check --diff alt.py' in result.stdout
    assert result.stderr.startswith("Comparing against revision 'master' (merge base ")


@only_on_posix
def test_pylint_changed_files_file_selection(tmpdir_factory):
    result = run(
        script_file='check/pylint-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD~1',
        setup='touch file.py\ngit add -A\ngit commit -m test --quiet --no-gpg-sign\n',
    )
    assert result.returncode == 0
    assert result.stdout == ''
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'HEAD~1'.\n"
            "Found 0 lintable files associated with changes.\n"
        ).split()
    )

    intercepted_prefix = (
        'INTERCEPTED env PYTHONPATH=dev_tools pylint --jobs=0 --rcfile=dev_tools/conf/.pylintrc '
    )

    result = run(
        script_file='check/pylint-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD~1',
        setup='mkdir cirq\n'
        'touch cirq/file.py\n'
        'git add -A\n'
        'git commit -m test --quiet --no-gpg-sign\n',
    )
    assert result.returncode == 0
    assert result.stdout == intercepted_prefix + 'cirq/file.py\n'
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'HEAD~1'.\n"
            "Found 1 lintable files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pylint-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD~1',
        setup='mkdir cirq\n'
        'touch ignore.py cirq/file.py\n'
        'git add -A\n'
        'git commit -m test --quiet --no-gpg-sign\n',
    )
    assert result.returncode == 0
    assert result.stdout == intercepted_prefix + 'cirq/file.py\n'
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'HEAD~1'.\n"
            "Found 1 lintable files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pylint-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD',
        setup='mkdir cirq\n'
        'touch ignore.py cirq/file.py\n'
        'git add -A\n'
        'git commit -m test --quiet --no-gpg-sign\n'
        'echo x > cirq/file.py',
    )
    assert result.returncode == 0
    assert result.stdout == intercepted_prefix + 'cirq/file.py\n'
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'HEAD'.\n"
            "Found 1 lintable files associated with changes.\n"
        ).split()
    )

    result = run(
        script_file='check/pylint-changed-files',
        tmpdir_factory=tmpdir_factory,
        arg='HEAD~1',
        setup='mkdir -p cirq dev_tools examples ignore\n'
        'touch cirq/file.py dev_tools/file.py examples/file.py\n'
        'touch ignore/ignore.py\n'
        'git add -A\n'
        'git commit -m test --quiet --no-gpg-sign\n',
    )
    assert result.returncode == 0
    assert result.stdout == intercepted_prefix + (
        'cirq/file.py dev_tools/file.py examples/file.py\n'
    )
    assert (
        result.stderr.split()
        == (
            "Comparing against revision 'HEAD~1'.\n"
            "Found 3 lintable files associated with changes.\n"
        ).split()
    )
