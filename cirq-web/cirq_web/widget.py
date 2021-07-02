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
import uuid

import cirq_web

# Resolve the path so the bundle file can be accessed properly
_DIST_PATH = Path(cirq_web.__file__).parents[1] / "cirq_ts" / "dist"


class Env(Enum):
    JUPYTER = 1
    COLAB = 2
    OTHER = 3


def _to_script_tag(path: str) -> str:
    """Dumps the contents of a particular bundle file into a script tag.

    Args:
        path: the path to the bundle file

    Returns:
        The bundle file as string (readable by browser) wrapped in HTML script tags.
    """
    bundle_file_path = path
    bundle_file = open(bundle_file_path, 'r', encoding='utf-8')
    bundle_file_contents = bundle_file.read()
    bundle_file.close()
    bundle_html = f'<script>{bundle_file_contents}</script>'

    return bundle_html


def write_output_file(output_directory: str, file_name: str, contents: str) -> Path:
    """Writes the output file and returns its absolute path.

    Args:
        output_directory: the directory in which the output file will be
        generated.

        file_name: the name of the output file. Default is 'bloch_sphere'

        contents: the contents of the file
    Returns:
        The path of the file as a Path object
    """
    # Ensure that the user enters a trailing slash
    file_path = Path(output_directory).joinpath(file_name)

    file_to_write_in = open(str(file_path), 'w', encoding='utf-8')
    file_to_write_in.write(contents)
    file_to_write_in.close()

    path_string = str(file_path)
    print(f'File can be found at: {path_string}')
    return file_path


class Widget:
    """Parent class for all widgets."""

    def __init__(self, bundle_file_path: str):
        """Initializes a Widget.

        Args:
            bundle_file_path: The relative path of the widget's bundle file starting from cirq_ts/
        """
        self.bundle_file_path = f'{_DIST_PATH}/{bundle_file_path}'

        # Generate a unique UUID for every instance of a Bloch sphere.
        # This helps with adding visualizations to scenes, etc.
        self.id = str(uuid.uuid1())

    def get_bundle_script(self):
        """Returns the bundle script of a widget"""
        return _to_script_tag(self.bundle_file_path)

    def create_html_content(self, client_code: str) -> str:
        div = f"""
        <meta charset="UTF-8">
        <div id="{self.id}"></div>
        """

        bundle_script = _to_script_tag(self.bundle_file_path)

        return div + bundle_script + client_code
