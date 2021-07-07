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

import pytest
import numpy as np

import cirq_web

from cirq.qis import to_valid_state_vector
from cirq.qis.states import bloch_vector_from_state_vector


def test_init_bloch_sphere_type():
    state_vector = to_valid_state_vector([math.sqrt(2) / 2, math.sqrt(2) / 2])
    bloch_sphere = cirq_web.BlochSphere(state_vector=state_vector)
    assert isinstance(bloch_sphere, cirq_web.BlochSphere)


@pytest.mark.parametrize('sphere_radius', [5, 0.2, 100])
def test_valid_bloch_sphere_radius(sphere_radius):
    state_vector = to_valid_state_vector([math.sqrt(2) / 2, math.sqrt(2) / 2])
    bloch_sphere = cirq_web.BlochSphere(sphere_radius, state_vector)
    assert sphere_radius == bloch_sphere.sphere_radius


@pytest.mark.parametrize('sphere_radius', [0, -1])
def test_invalid_bloch_sphere_radius(sphere_radius):
    with pytest.raises(ValueError):
        cirq_web.BlochSphere(sphere_radius=sphere_radius)


@pytest.mark.parametrize(
    'state_vector', [to_valid_state_vector([math.sqrt(2) / 2, math.sqrt(2) / 2])]
)
def test_valid_bloch_sphere_vector(state_vector):
    bloch_sphere = cirq_web.BlochSphere(state_vector=state_vector)
    bloch_vector = bloch_vector_from_state_vector(state_vector, 0)
    assert np.array_equal(bloch_vector, bloch_sphere.bloch_vector)


def test_no_state_vector_given():
    with pytest.raises(ValueError):
        cirq_web.BlochSphere()


def test_bloch_sphere_default_client_code():
    state_vector = to_valid_state_vector([math.sqrt(2) / 2, math.sqrt(2) / 2])
    bloch_sphere = cirq_web.BlochSphere(state_vector=state_vector)

    expected_client_code = f"""
        <script>
        renderBlochSphere('{bloch_sphere.id}', 5)
            .addVector(1.0, 0.0, 0.0);
        </script>
        """

    assert expected_client_code == bloch_sphere.get_client_code()


def test_bloch_sphere_default_bundle_name():
    state_vector = to_valid_state_vector([math.sqrt(2) / 2, math.sqrt(2) / 2])
    bloch_sphere = cirq_web.BlochSphere(state_vector=state_vector)

    assert bloch_sphere.get_widget_bundle_name() == 'bloch_sphere.bundle.js'
