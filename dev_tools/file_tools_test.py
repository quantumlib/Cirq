# Copyright 2024 The Cirq Developers
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

import tempfile

import pytest

from .file_tools import read_file_filtered

start_skip = '.. ▶︎─── start github-only'
end_skip = '.. ▶︎─── end github-only'


def output_from_read_file_filtered(content):
    """Call `read_file_filtered` using a temp file to store `content`."""
    with tempfile.NamedTemporaryFile(mode='w+') as tf:
        tf.write(content)
        tf.seek(0)
        return read_file_filtered(tf.name, start_skip, end_skip)


def test_valid():
    content = '''Cirq is great.

.. ▶︎─── start github-only
.. raw:: html

   <p>
.. ▶︎─── end github-only

Cirq is still great.
'''
    result = output_from_read_file_filtered(content)
    assert result == 'Cirq is great.\n\n\nCirq is still great.\n'


def test_valid_multiple_regions():
    content = '''Cirq is great.

.. ▶︎─── start github-only
.. raw:: html

   <p>
.. ▶︎─── end github-only

Cirq is still great.

.. ▶︎─── start github-only --- some text here ---
.. raw:: html

   <p>
.. ▶︎─── end github-only --- some more random text here ---
'''
    result = output_from_read_file_filtered(content)
    assert result == 'Cirq is great.\n\n\nCirq is still great.\n\n'


def test_exception_missing_begin():
    content = '''Cirq is great.
.. ▶︎─── end github-only
Cirq is still great.
'''
    with pytest.raises(Exception, match=r'^\[.*?\] Encountered .* without'):
        _ = output_from_read_file_filtered(content)


def test_exception_two_begins():
    content = '''Cirq is great.
.. ▶︎─── start github-only
Cirq is still great.
.. ▶︎─── start github-only
Yup, still great.
'''
    with pytest.raises(Exception, match=r'while already skipping'):
        _ = output_from_read_file_filtered(content)


def test_exception_missing_end():
    content = '''Cirq is great.
.. ▶︎─── start github-only
Cirq is still great.
'''
    with pytest.raises(Exception, match=r'^Missing final '):
        _ = output_from_read_file_filtered(content)
