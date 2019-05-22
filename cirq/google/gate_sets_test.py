# Copyright 2019 The Cirq Developers
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

import numpy as np
import pytest

import cirq
import cirq.google as cg


@pytest.mark.parametrize(
    ('gate', 'axis_half_turns', 'half_turns'),
    ((cirq.X, 0.0, 1.0),
     (cirq.X ** 0.2, 0.0, 0.2),
     (cirq.Y, 0.5, 1.0),
     (cirq.Y ** 0.2, 0.5, 0.2),
     (cirq.PhasedXPowGate(exponent=0.1, phase_exponent=0.2), 0.2, 0.1),
     (cirq.Rx(0.1 * np.pi), 0.0, 0.1),
     (cirq.Ry(0.2 * np.pi), 0.5, 0.2))
)
def test_serialize_exp_w(gate, axis_half_turns, half_turns):
    q = cirq.GridQubit(1, 2)
    assert cg.XMON.serialize_op(gate.on(q)) == {
        'gate': {
            'id': 'exp_w'
        },
        'args': {
            'axis_half_turns': {
                'arg_value': {
                    'float_value': axis_half_turns
                }
            },
            'half_turns': {
                'arg_value': {
                    'float_value': half_turns
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }

