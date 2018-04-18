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

import subprocess

import pytest

from python_ci_utils import env_tools


def test_failure_and_success():
    assert env_tools.sub_run(
        'true',
        silent=True
    ) == (None, None, 0)

    assert env_tools.sub_run(
        'true',
        raise_error_if_process_fails=True,
        silent=True
    ) == (None, None, 0)

    assert env_tools.sub_run(
        'false',
        raise_error_if_process_fails=False,
        silent=True
    ) == (None, None, 1)

    with pytest.raises(subprocess.CalledProcessError):
        env_tools.sub_run(
            'false',
            raise_error_if_process_fails=True,
            silent=True)


def test_capture_stdout():
    assert env_tools.sub_run(
        'echo', 'test',
        silent=True
    ) == (None, None, 0)

    assert env_tools.sub_run(
        'echo', 'test',
        capture_stdout=False,
        silent=True
    ) == (None, None, 0)

    assert env_tools.sub_run(
        'echo', 'test',
        capture_stdout=True,
        silent=True
    ) == ('test\n', None, 0)


def test_capture_shell_stderr():
    assert env_tools.sub_run(
        'echo', 'test', '1>&2',
        silent=True,
        shell=True,
    ) == (None, None, 0)

    assert env_tools.sub_run(
        'echo', 'test', '1>&2',
        silent=True,
        shell=True,
        capture_stderr=True,
    ) == (None, 'test\n', 0)
