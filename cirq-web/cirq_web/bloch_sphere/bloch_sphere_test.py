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
import json

import pytest
import numpy as np

import cirq_web

from cirq.qis import to_valid_state_vector
from cirq.qis.states import bloch_vector_from_state_vector


def get_bundle_file_path():
    # Need to call this from the root directory
    absolute_path = cirq_web.resolve_path()
    bundle_file_path = f'{absolute_path}/cirq_ts/dist/bloch_sphere.bundle.js'
    bundle_script = cirq_web.to_script_tag(bundle_file_path)
    return bundle_script


def test_init_bloch_sphere_type():
    bloch_sphere = cirq_web.BlochSphere()
    assert isinstance(bloch_sphere, cirq_web.BlochSphere)


@pytest.mark.parametrize('sphere_radius', [5, 0.2, 100])
def test_valid_bloch_sphere_radius_json_info(sphere_radius):
    bloch_sphere = cirq_web.BlochSphere(sphere_radius=sphere_radius)
    expected_object = {'radius': sphere_radius}
    expected = json.dumps(expected_object)
    assert expected == bloch_sphere.sphere_json


@pytest.mark.parametrize('sphere_radius', [0, -1])
def test_invalid_bloch_sphere_radius_json_info(sphere_radius):
    with pytest.raises(BaseException):
        cirq_web.BlochSphere(sphere_radius=sphere_radius)


@pytest.mark.parametrize(
    'state_vector', [to_valid_state_vector([math.sqrt(2) / 2, math.sqrt(2) / 2])]
)
def test_valid_bloch_sphere_vector_json(state_vector):
    bloch_sphere = cirq_web.BlochSphere(state_vector=state_vector)
    bloch_vector = bloch_vector_from_state_vector(state_vector, 0)
    expected_object = {
        'x': bloch_vector[0].item(),
        'y': bloch_vector[1].item(),
        'z': bloch_vector[2].item(),
        'v_length': 5,  # This is the default value
    }
    expected = json.dumps(expected_object)
    assert expected == bloch_sphere.vector_json


def test_random_vector():
    """Checks if we generated a random vector by checking the return type."""
    bloch_sphere = cirq_web.BlochSphere(random=True)
    vec = bloch_sphere.bloch_vector
    did_create_random_vector = isinstance(vec, np.ndarray) and len(vec) == 3
    assert did_create_random_vector


def test_repr_html():
    # This tests more of the path rather than the contents.
    # Add more contents later
    bloch_sphere = cirq_web.BlochSphere()
    bundle_script = get_bundle_file_path()
    expected = f"""
        <meta charset="UTF-8">
        <div id="container"></div>
        {bundle_script}
        <script>
        CirqTS.showSphere('{bloch_sphere.sphere_json}', '{bloch_sphere.vector_json}');
        </script>
        """
    assert expected == bloch_sphere._repr_html_()


def test_generate_HTML_file_with_browser(tmpdir):
    path = tmpdir.mkdir('dir')

    bloch_sphere = cirq_web.BlochSphere()
    test_path = bloch_sphere.generate_HTML_file(
        output_directory=str(path), file_name='test.html', open_in_browser=True
    )

    template_div = f"""
        <meta charset="UTF-8">
        <div id="container"></div>
        """
    bundle_script = get_bundle_file_path()

    template_script = f"""
        <script>
        CirqTS.showSphere('{bloch_sphere.sphere_json}', '{bloch_sphere.vector_json}');
        </script>
        """

    expected = template_div + bundle_script + template_script
    actual = open(str(test_path), 'r', encoding='utf-8').read()

    assert expected == actual
