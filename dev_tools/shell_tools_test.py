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

import contextlib
import io
import subprocess

import pytest

from dev_tools import shell_tools
from dev_tools.test_utils import only_on_posix


def run(*args, **kwargs):
    return shell_tools.run(*args, log_run_to_stderr=False, **kwargs)


@only_on_posix
def test_run_raises_on_failure():
    assert run('true').returncode == 0
    with pytest.raises(subprocess.CalledProcessError):
        run('false')
    assert run('false', check=False).returncode == 1


def test_run_returns_string_output():
    result = run(['echo', 'hello', 'world'], capture_output=True)
    assert result.stdout == 'hello world\n'


def test_run_with_command_logging():
    catch_stderr = io.StringIO()
    kw = {'stdout': subprocess.DEVNULL}
    with contextlib.redirect_stderr(catch_stderr):
        shell_tools.run(['echo', '-n', 'a', 'b'], **kw)
    assert catch_stderr.getvalue() == "run: ('echo', '-n', 'a', 'b')\n"
    catch_stderr = io.StringIO()
    with contextlib.redirect_stderr(catch_stderr):
        shell_tools.run(['echo', '-n', 'a', 'b'], abbreviate_non_option_arguments=True, **kw)
    assert catch_stderr.getvalue() == "run: ('echo', '-n', '[...]')\n"


@only_on_posix
def test_output_of():
    assert shell_tools.output_of('true') == ''
    with pytest.raises(subprocess.CalledProcessError):
        _ = shell_tools.output_of('false')
    assert shell_tools.output_of(['echo', 'test']) == 'test'
    # filtering of the None arguments was removed.  check this now fails
    with pytest.raises(TypeError):
        _ = shell_tools.output_of(['echo', 'test', None, 'duck'])
    assert shell_tools.output_of('pwd', cwd='/tmp') in ['/tmp', '/private/tmp']
