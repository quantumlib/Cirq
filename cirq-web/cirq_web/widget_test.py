# Copyright 2021 The Cirq Developers
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

import pytest
from IPython.testing.globalipapp import get_ipython
import cirq_web

"""A global mock IPython environment."""
ip = get_ipython()


def test_to_script_tag(tmp_path):
    # setup test data
    tempfile = tmp_path / "tempfile"
    content = "console.log('test')"
    tempfile.write_text(content)

    # call the tested method/class
    result = cirq_web.to_script_tag(tempfile)

    # compare actual with expected
    expected = f"<script>{content}</script>"
    assert result == expected


def test_determine_non_notebook_env():
    assert cirq_web.determine_env() == cirq_web.widget.Env.OTHER


@pytest.mark.parametrize(
    'mock_env, result',
    [
        ('ZMQInteractiveShell', cirq_web.widget.Env.JUPYTER),
        ('google.colab_shell', cirq_web.widget.Env.COLAB),
    ],
)
def test_determine_notebook_env(mock_env, result):
    # Set the class name to a predetermined value.
    # Note that after this, the global state as a whole is altered
    ip.__class__.__name__ = mock_env
    assert cirq_web.determine_env() == result


def test_write_output_file(tmpdir):
    path = tmpdir.mkdir('dir')

    file_name = 'tempfile.txt'
    content = "this is a test file"

    cirq_web.write_output_file(str(path), file_name, content)

    new_file = path.join(file_name)
    new_file_content = new_file.read()

    assert new_file_content == content
