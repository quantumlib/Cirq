# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest

import nbformat


def find_examples_jupyter_notebook_paths():
    examples_folder = os.path.dirname(__file__)
    for filename in os.listdir(examples_folder):
        if not filename.endswith('.ipynb'):
            continue
        yield os.path.join(examples_folder, filename)


@pytest.mark.parametrize('path', find_examples_jupyter_notebook_paths())

def test_can_run_examples_jupyter_notebook(path):
    notebook = nbformat.read(path, nbformat.NO_CONVERT)
    state = {}  # type: Dict[str, Any]

    for cell in notebook.cells:
        if cell.cell_type == 'code' and not is_matplotlib_cell(cell):
            try:
                exec(strip_magics_and_shows(cell.source), state)
            # coverage: ignore
            except:
                print('Failed to run {}.'.format(path))
                raise


def is_matplotlib_cell(cell):
    return "%matplotlib" in cell.source


def strip_magics_and_shows(text):
    """Remove Jupyter magics and pyplot show commands."""
    lines = [line for line in text.split('\n')
             if not contains_magic_or_show(line)]
    return '\n'.join(lines)

def contains_magic_or_show(line):
    return (line.strip().startswith('%') or
            'pyplot.show(' in line or
            'plt.show(' in line)
