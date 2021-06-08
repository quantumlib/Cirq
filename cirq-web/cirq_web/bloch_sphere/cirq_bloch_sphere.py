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

import json
import math
import webbrowser

from cirq_web import widget

from cirq.testing import random_superposition
from cirq.qis import to_valid_state_vector
from cirq.qis.states import bloch_vector_from_state_vector


class BlochSphere(widget.Widget):
    def __init__(
        self,
        sphere_radius=5,
        state_vector=to_valid_state_vector([math.sqrt(2) / 2, math.sqrt(2) / 2]),
        random=False,
    ):
        """Initializes a CirqBlochSphere, gathering all the user information and
        converting to JSON for output

        Args:
            sphere_radius: the radius of the bloch sphere in the three.js diagram.
            The default value is 5.

            state_vector: a state vector to pass in to be represented. The default
            vector is 1/sqrt(2) (|0⟩ +|1⟩) (the plus state)
        """

        self.sphere_json = self._convertSphereInput(sphere_radius)
        self.bloch_vector = (
            self._createRandomVector() if random else self._createVector(state_vector)
        )
        self.vector_json = self._serializeVector(*self.bloch_vector, sphere_radius)

    def _repr_html_(self):
        """Allows the object's html to be easily displayed in a notebook
        by using the display() method.

        If display() is called from the command line, [INSERT HERE]
        """

        absolute_path_prefix = widget.resolve_path()
        bundle_file_path = f'{absolute_path_prefix}/cirq_ts/dist/bloch_sphere.bundle.js'
        bundle_script = widget.to_script_tag(bundle_file_path)

        return f"""
        <div id="container"></div>
        {bundle_script}
        <script>
        createSphere.showSphere('{self.sphere_json}', '{self.vector_json}');
        </script>
        """

    def generate_HTML_file(
        self, output_directory='./', file_name='bloch_sphere.html', open_in_browser=False
    ):
        """Generates a portable HTML file of the bloch sphere that
        can be run anywhere. Prints out the absolute path of the file to the console.

        Args:
            output_directory: the directory in which the output file will be
            generated. The default is the current directory ('./')

            file_name: the name of the output file. Default is 'bloch_sphere'

            open: if True, opens the newly generated file automatically in the browser.

        Returns the path of the HTML file.

        For now, if ran in a notebook, this function just returns. Support for downloading
        the HTML file via the browser can be added later.
        """
        
        template_div = f"""
        <div id="container"></div>
        """

        template_script = f"""
        <script>
        createSphere.showSphere('{self.sphere_json}', '{self.vector_json}');
        </script>
        """

        absolute_path_prefix = widget.resolve_path()
        # Spit out the bundle.js into a script tag to then serve to the user
        bundle_file_path = f'{absolute_path_prefix}/cirq_ts/dist/bloch_sphere.bundle.js'
        bundle_script = widget.to_script_tag(bundle_file_path)

        contents = template_div + bundle_script + template_script
        path_of_html_file = widget.write_output_file(output_directory, file_name, contents)

        if open_in_browser:
            webbrowser.open(str(path_of_html_file), new=2)  # 2 opens in a new tab if possible

        return path_of_html_file

    def _convertSphereInput(self, radius):
        if radius <= 0:
            raise (BaseException('You must input a positive radius for the sphere'))

        obj = {'radius': radius}
        return json.dumps(obj)

    def _createVector(self, state_vector):
        """Any state_vector input will need to come from cirq.to_valid_state_vector,
        so we can assume that a valid state_vector will be passed in.
        """
        bloch_vector = bloch_vector_from_state_vector(state_vector, 0)

        return bloch_vector

    def _createRandomVector(self):
        random_vector = random_superposition(2)
        state_vector = to_valid_state_vector(random_vector)
        bloch_vector = bloch_vector_from_state_vector(state_vector, 0)
        return bloch_vector

    def _serializeVector(self, x, y, z, length=5):
        # .item() bc input is of type float32, need to convert to seralizable type
        obj = {
            'x': x.item(),
            'y': y.item(),
            'z': z.item(),
            'v_length': length,
        }
        return json.dumps(obj)
