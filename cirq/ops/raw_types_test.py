# Copyright 2018 The Cirq Developers
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

import cirq

class ValiGate(cirq.Gate):
    def num_qubits(self):
        return 2

    def validate_args(self, qubits):
        if len(qubits) == 3:
            raise ValueError()

q00 = cirq.NamedQubit('q00')
q01 = cirq.NamedQubit('q01')
q10 = cirq.NamedQubit('q10')

def test_gate():
    g = ValiGate()
    assert g.num_qubits() == 2

    _ = g.on(q00, q10)
    with pytest.raises(ValueError):
        _ = g.on(q00, q10, q01)

    _ = g(q00)
    _ = g(q00, q10)
    with pytest.raises(ValueError):
        _ = g(q10, q01, q00)

def test_control():
    g = ValiGate()
    controlled_g = g.__control__()
    assert controlled_g.sub_gate == g
    assert controlled_g.control_qubits == []
    assert controlled_g.num_unspecified_control_qubits == 1
    specified_controlled_g = g.__control__([q00, q01])
    assert specified_controlled_g.sub_gate == g
    assert specified_controlled_g.control_qubits == [q00, q01]
    assert specified_controlled_g.num_unspecified_control_qubits == 0

def test_op():
    g = ValiGate()
    op = g(q00)
    with pytest.raises(ValueError):
        _ = op.__control__()
    controlled_op = op.__control__([q01, q10])
    assert controlled_op.sub_operation.sub_operation == op
    assert controlled_op.sub_operation.control == q01
    assert controlled_op.control == q10