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

import math
import webbrowser
import uuid

from typing import Union
from pathlib import PosixPath, WindowsPath

from numpy import ndarray

from cirq_web import widget

from cirq.testing import random_superposition
from cirq.qis import to_valid_state_vector
from cirq.qis.states import bloch_vector_from_state_vector
from cirq.protocols import to_json


class BlochSphere(widget.Widget):
    def __init__(
        self,
        sphere_radius: int = 5,
        state_vector: ndarray = to_valid_state_vector([math.sqrt(2) / 2, math.sqrt(2) / 2]),
        random: bool = False,
    ):
        """Initializes a BlochSphere, gathering all the user information and
        converting to JSON for output.

        Also initializes it's parent class Widget with the bundle file provided.

        Args:
            sphere_radius: the radius of the bloch sphere in the three.js diagram.
            The default value is 5.

            state_vector: a state vector to pass in to be represented. The default
            vector is 1/sqrt(2) (|0⟩ +|1⟩) (the plus state)
        """

        super().__init__('cirq_ts/dist/bloch_sphere.bundle.js')

        self.sphere_json = self._convert_sphere_input(sphere_radius)
        self.bloch_vector = (
            self._create_random_vector() if random else self._create_vector(state_vector)
        )
        self.vector_json = self._serialize_vector(*self.bloch_vector, sphere_radius)

        # Generate a unique UUID for every instance of a Bloch sphere.
        # This helps with adding visualizations to scenes, etc.
        self.id = str(uuid.uuid1())

    def _repr_html_(self):
        """Allows the object's html to be easily displayed in a notebook
        by using the display() method.

        If display() is called from the command line, [INSERT HERE]
        """

        bundle_script = super().get_bundle_script()
        return f"""
        <meta charset="UTF-8">
        <div id="{self.id}"></div>
        {bundle_script}
        <script>
        renderBlochSphere('{self.sphere_json}', '{self.id}').addVector('{self.vector_json}');
        </script>
        """

    def get_id(self):
        return self.id

    def generate_html_file(
        self,
        output_directory: str = './',
        file_name: str = 'bloch_sphere.html',
        open_in_browser: bool = False,
    ) -> Union[PosixPath, WindowsPath]:
        """Generates a portable HTML file of the bloch sphere that
        can be run anywhere. Prints out the absolute path of the file to the console.

        Args:
            output_directory: the directory in which the output file will be
            generated. The default is the current directory ('./')

            file_name: the name of the output file. Default is 'bloch_sphere'

            open: if True, opens the newly generated file automatically in the browser.

        Returns:
            The path of the HTML file in either PosixPath or WindowsPath form, depending on the
            operating system.

        For now, if ran in a notebook, this function just returns. Support for downloading
        the HTML file via the browser can be added later.
        """

        template_div = f"""
        <meta charset="UTF-8">
        <div id="{self.id}"></div>
        """

        template_script = f"""
        <script>
        renderBlochSphere('{self.sphere_json}', '{self.id}').addVector('{self.vector_json}');
        </script>
        """

        bundle_script = super().get_bundle_script()
        contents = template_div + bundle_script + template_script
        path_of_html_file = widget.write_output_file(output_directory, file_name, contents)

        if open_in_browser:
            webbrowser.open(str(path_of_html_file), new=2)  # 2 opens in a new tab if possible

        return path_of_html_file

    def _convert_sphere_input(self, radius: int) -> str:
        if radius <= 0:
            raise (BaseException('You must input a positive radius for the sphere'))

        obj = {'radius': radius}
        return to_json(obj, indent=None)

    def _create_vector(self, state_vector: ndarray) -> ndarray:
        """Any state_vector input will need to come from cirq.to_valid_state_vector,
        so we can assume that a valid state_vector will be passed in.
        """
        bloch_vector = bloch_vector_from_state_vector(state_vector, 0)

        return bloch_vector

    def _create_random_vector(self) -> ndarray:
        random_vector = random_superposition(2)
        state_vector = to_valid_state_vector(random_vector)
        bloch_vector = bloch_vector_from_state_vector(state_vector, 0)
        return bloch_vector

    def _serialize_vector(self, x: float, y: float, z: float, length: float = 5) -> str:
        # .item() because input is of type float32, need to convert to serializable type
        obj = {
            'x': x.item(),
            'y': y.item(),
            'z': z.item(),
            'length': length,
        }
        return to_json(obj, indent=None)
