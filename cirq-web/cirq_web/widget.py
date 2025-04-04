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

import os
import uuid
import webbrowser
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import cirq_web

# Resolve the path so the bundle file can be accessed properly
_DIST_PATH = Path(cirq_web.__file__).parents[1] / "cirq_ts" / "dist"


class Env(Enum):
    JUPYTER = 1
    COLAB = 2
    OTHER = 3


class Widget(ABC):
    """Abstract class for all widgets."""

    def __init__(self):
        """Initializes a Widget.

        Gives a widget a unique ID.
        """
        # Generate a unique UUID for every instance of a Widget.
        # This helps with adding visualizations to scenes, etc.
        self.id = str(uuid.uuid1())

    @abstractmethod
    def get_client_code(self) -> str:
        """Returns HTML code to render the widget."""
        raise NotImplementedError()

    @abstractmethod
    def get_widget_bundle_name(self) -> str:
        """Returns the name of the Javascript library file for this widget."""
        raise NotImplementedError()

    def _repr_html_(self):
        """Allows the object's html to be easily displayed in a notebook
        by using the display() method.
        """
        client_code = self.get_client_code()
        return self._create_html_content(client_code)

    def generate_html_file(
        self,
        output_directory: str = './',
        file_name: str = 'bloch_sphere.html',
        open_in_browser: bool = False,
    ) -> str:
        """Generates a portable HTML file of the widget that
        can be run anywhere. Prints out the absolute path of the file to the console.

        Args:
            output_directory: the directory in which the output file will be
            generated. The default is the current directory ('./')

            file_name: the name of the output file. Default is 'bloch_sphere'

            open_in_browser: if True, opens the newly generated file automatically in the browser.

        Returns:
            The path of the HTML file in as a Path object.
        """
        client_code = self.get_client_code()
        contents = self._create_html_content(client_code)
        path_of_html_file = os.path.join(output_directory, file_name)
        with open(path_of_html_file, 'w', encoding='utf-8') as f:
            f.write(contents)

        if open_in_browser:
            webbrowser.open(path_of_html_file, new=2)

        return path_of_html_file

    def _get_bundle_script(self):
        """Returns the bundle script of a widget"""
        bundle_filename = self.get_widget_bundle_name()
        return _to_script_tag(bundle_filename)

    def _create_html_content(self, client_code: str) -> str:
        div = f"""
        <meta charset="UTF-8">
        <div id="{self.id}"></div>
        """

        bundle_script = self._get_bundle_script()

        return div + bundle_script + client_code


def _to_script_tag(bundle_filename: str) -> str:
    """Dumps the contents of a particular bundle file into a script tag.

    Args:
        bundle_filename: the path to the bundle file

    Returns:
        The bundle file as string (readable by browser) wrapped in HTML script tags.
    """
    bundle_file_path = os.path.join(_DIST_PATH, bundle_filename)
    bundle_file = open(bundle_file_path, 'r', encoding='utf-8')
    bundle_file_contents = bundle_file.read()
    bundle_file.close()
    bundle_html = f'<script>{bundle_file_contents}</script>'

    return bundle_html
