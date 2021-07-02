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

import webbrowser

from pathlib import Path

from cirq_web import widget

from cirq.qis.states import bloch_vector_from_state_vector
from cirq.qis.states import STATE_VECTOR_LIKE


class BlochSphere(widget.Widget):
    def __init__(
        self,
        sphere_radius: int = 5,
        state_vector: STATE_VECTOR_LIKE = None,
    ):
        """Initializes a BlochSphere, gathering all the user information and
        converting to JSON for output.

        Also initializes it's parent class Widget with the bundle file provided.

        Args:
            sphere_radius: the radius of the bloch sphere in the three.js diagram.
            The default value is 5.

            state_vector: a state vector to pass in to be represented.
        """

        super().__init__('bloch_sphere.bundle.js')

        if sphere_radius <= 0:
            raise ValueError('You must input a positive radius for the sphere')
        self.sphere_radius = sphere_radius

        if state_vector is None:
            raise ValueError('No state vector given in BlochSphere initialization')
        self.bloch_vector = bloch_vector_from_state_vector(state_vector, 0)

    def _repr_html_(self):
        """Allows the object's html to be easily displayed in a notebook
        by using the display() method.

        If display() is called from the command line, [INSERT HERE]
        """

        client_code = f"""
        <script>
        renderBlochSphere('{self.id}', {self.sphere_radius})
            .addVector({self.bloch_vector[0]}, {self.bloch_vector[1]}, {self.bloch_vector[2]});
        </script>
        """

        return super().create_html_content(client_code)

    def generate_html_file(
        self,
        output_directory: str = './',
        file_name: str = 'bloch_sphere.html',
        open_in_browser: bool = False,
    ) -> Path:
        """Generates a portable HTML file of the bloch sphere that
        can be run anywhere. Prints out the absolute path of the file to the console.

        Args:
            output_directory: the directory in which the output file will be
            generated. The default is the current directory ('./')

            file_name: the name of the output file. Default is 'bloch_sphere'

            open: if True, opens the newly generated file automatically in the browser.

        Returns:
            The path of the HTML file in as a Path object.
        """

        client_code = f"""
        <script>
        renderBlochSphere('{self.id}', {self.sphere_radius})
            .addVector({self.bloch_vector[0]}, {self.bloch_vector[1]}, {self.bloch_vector[2]});
        </script>
        """

        contents = super().create_html_content(client_code)
        path_of_html_file = widget.write_output_file(output_directory, file_name, contents)

        if open_in_browser:
            webbrowser.open(str(path_of_html_file), new=2)  # 2 opens in a new tab if possible

        return path_of_html_file
