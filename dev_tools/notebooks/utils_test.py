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

import filecmp
import os
import shutil
import tempfile

import pytest

import dev_tools.notebooks as dt


def write_test_data(ipynb_txt, tst_txt):
    directory = tempfile.mkdtemp()
    ipynb_path = os.path.join(directory, 'test.ipynb')
    with open(ipynb_path, 'w') as f:
        f.write(ipynb_txt)

    tst_path = os.path.join(directory, 'test.tst')
    with open(tst_path, 'w') as f:
        f.write(tst_txt)

    return directory, ipynb_path


def test_rewrite_notebook():
    directory, ipynb_path = write_test_data('d = 5\nd = 4', 'd = 5->d = 3')

    path = dt.rewrite_notebook(ipynb_path)

    assert path != ipynb_path
    with open(path, 'r') as f:
        rewritten = f.read()
        assert rewritten == 'd = 3\nd = 4'

    os.remove(path)
    shutil.rmtree(directory)


def test_rewrite_notebook_multiple():
    directory, ipynb_path = write_test_data('d = 5\nd = 4', 'd = 5->d = 3\nd = 4->d = 1')

    path = dt.rewrite_notebook(ipynb_path)

    with open(path, 'r') as f:
        rewritten = f.read()
        assert rewritten == 'd = 3\nd = 1'

    os.remove(path)
    shutil.rmtree(directory)


def test_rewrite_notebook_ignore_non_seperator_lines():
    directory, ipynb_path = write_test_data('d = 5\nd = 4', 'd = 5->d = 3\n# comment')

    path = dt.rewrite_notebook(ipynb_path)

    with open(path, 'r') as f:
        rewritten = f.read()
        assert rewritten == 'd = 3\nd = 4'

    os.remove(path)
    shutil.rmtree(directory)


def test_rewrite_notebook_no_tst_file():
    directory = tempfile.mkdtemp()
    ipynb_path = os.path.join(directory, 'test.ipynb')
    with open(ipynb_path, 'w') as f:
        f.write('d = 5\nd = 4')

    path = dt.rewrite_notebook(ipynb_path)
    assert path != ipynb_path
    assert filecmp.cmp(path, ipynb_path)

    os.remove(path)
    shutil.rmtree(directory)


def test_rewrite_notebook_extra_seperator():
    directory, ipynb_path = write_test_data('d = 5\nd = 4', 'd = 5->d = 3->d = 1')

    with pytest.raises(AssertionError, match='only contain one'):
        _ = dt.rewrite_notebook(ipynb_path)

    shutil.rmtree(directory)


def test_rewrite_notebook_unused_patterns():
    directory, ipynb_path = write_test_data('d = 5\nd = 4', 'd = 2->d = 3')

    with pytest.raises(AssertionError, match='re.compile'):
        _ = dt.rewrite_notebook(ipynb_path)

    shutil.rmtree(directory)
