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

from pathlib import Path
from enum import Enum
import IPython
import cirq_web


class Env(Enum):
    JUPYTER = 1
    COLAB = 2
    OTHER = 3


def to_script_tag(path):
    """Dumps the contents of a particular bundle file into a script tag.

    Args:
        path: the path to the bundle file
    """
    bundle_file_path = path
    bundle_file = open(bundle_file_path, 'r')
    bundle_file_contents = bundle_file.read()
    bundle_file.close()
    bundle_html = f'<script>{bundle_file_contents}</script>'

    return bundle_html


def determine_env():
    """Determines if a Widget is being run in a Jupyter notebook

    The return types of IPython().get_ipython().__class__.__name__
    we care about are only "ZMQInteractiveShell", and potentially,
    "google.colab_shell", since those are the only environments that
    we're supporting at this stage.
    """
    env = IPython.get_ipython().__class__.__name__
    if env == 'ZMQInteractiveShell':
        return Env.JUPYTER
    elif env == 'google.colab_shell':
        return Env.COLAB
    else:
        return Env.OTHER


def write_output_file(output_directory, file_name, contents):
    """Writes the output file and returns its absolute path.

    Args:
        output_directory: the directory in which the output file will be
        generated.

        file_name: the name of the output file. Default is 'bloch_sphere'

        contents: the contents of the file
    """
    # Ensure that the user enters a trailing slash
    file_path = Path(output_directory).joinpath(file_name)

    file_to_write_in = open(str(file_path), 'w')
    file_to_write_in.write(contents)
    file_to_write_in.close()

    path_string = str(file_path)
    print(f'File can be found at: {path_string}')
    return file_path


def resolve_path():
    # Go go levels up from the __init__ file
    cirq_path = Path(cirq_web.__file__).parents[1]
    return cirq_path


class Widget:
    """Parent class for all widgets."""
