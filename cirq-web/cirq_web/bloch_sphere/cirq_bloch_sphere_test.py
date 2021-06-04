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
import math
import json
import numpy as np

import cirq_web

from cirq.qis import to_valid_state_vector
from cirq.qis.states import bloch_vector_from_state_vector

def test_init_bloch_sphere_type():
    bloch_sphere = cirq_web.CirqBlochSphere()
    assert isinstance(bloch_sphere, cirq_web.CirqBlochSphere)


@pytest.mark.parametrize('sphere_radius', [5, 0.2, 100])
def test_valid_bloch_sphere_radius_json_info(sphere_radius):
    bloch_sphere = cirq_web.CirqBlochSphere(sphere_radius=sphere_radius)
    expected_object = {
        'radius': sphere_radius
    } 
    expected = json.dumps(expected_object)
    assert expected == bloch_sphere.sphere_json

@pytest.mark.parametrize('sphere_radius', [0, -1])
def test_invalid_bloch_sphere_radius_json_info(sphere_radius):
    with pytest.raises(BaseException):
        bloch_sphere = cirq_web.CirqBlochSphere(sphere_radius=sphere_radius)

@pytest.mark.parametrize('state_vector',
    [to_valid_state_vector([math.sqrt(2)/2, math.sqrt(2)/2])])
def test_valid_bloch_sphere_vector_json(state_vector):
    bloch_sphere = cirq_web.CirqBlochSphere(state_vector=state_vector)
    bloch_vector = bloch_vector_from_state_vector(state_vector, 0)
    expected_object = {
        'x': bloch_vector[0].item(),
        'y': bloch_vector[1].item(),
        'z': bloch_vector[2].item(),
        'v_length': 5, # This is the default value
    }
    expected = json.dumps(expected_object)
    assert expected == bloch_sphere.vector_json


@pytest.mark.skip(reason="Find good parameters for this")
def test_invalid_bloch_sphere_vector_json(state_vector):
    with pytest.raises(ValueError):
        cirq_web.CirqBlochSphere(state_vector=state_vector)

def test_repr_html():
    # This tests more of the path rather than the contents.
    # Add more contents later
    bloch_sphere = cirq_web.CirqBlochSphere()
    # Need to call this from the root directory
    bundle_file_path = f'cirq-web/cirq_ts/dist/bloch_sphere.bundle.js'
    bundle_script = cirq_web.to_script_tag(bundle_file_path)
    expected = f"""
        <div id="container"></div>
        {bundle_script}
        <script>
        createSphere.showSphere('{bloch_sphere.sphere_json}', '{bloch_sphere.vector_json}');
        </script>
        """
    assert expected == bloch_sphere._repr_html_()


def test_generate_HTML_file(tmpdir):
    path = tmpdir.mkdir('dir')

    bloch_sphere = cirq_web.CirqBlochSphere()
    bundle_file_path = f'cirq-web/cirq_ts/dist/bloch_sphere.bundle.js'
    bundle_script = cirq_web.to_script_tag(bundle_file_path)

    template_div = f"""
        <div id="container"></div>
        """

    template_script = f"""
        <script>
        createSphere.showSphere('{bloch_sphere.sphere_json}', '{bloch_sphere.vector_json}');
        </script>
        """

    file_name = 'test.html'
    contents = template_div + bundle_script + template_script

    new_path = cirq_web.write_output_file(str(path), file_name, contents)

    expected = template_div + bundle_script + template_script
    actual = open(str(new_path), 'r').read()

    assert expected == actual
