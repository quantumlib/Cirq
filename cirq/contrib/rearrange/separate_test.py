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

from cirq import ops, Circuit
from cirq.ops import CZ, H, CNOT, X, Y, Z, SWAP

from cirq.contrib.rearrange.separate import convert_circuit, non_clifford_half, clifford_half

from cirq.testing import assert_allclose_up_to_global_phase


def _toffoli_circuit():
    q0, q1, q2 = (ops.NamedQubit('q{}'.format(i)) for i in range(3))
    return Circuit.from_ops(
        Y(q2) ** 0.5,
        X(q2),
        CNOT(q1, q2),
        Z(q2) ** -0.25,
        CNOT(q1, q2),
        CNOT(q2, q1),
        CNOT(q1, q2),
        CNOT(q0, q1),
        CNOT(q1, q2),
        CNOT(q2, q1),
        CNOT(q1, q2),
        Z(q2) ** 0.25,
        CNOT(q1, q2),
        Z(q2) ** -0.25,
        CNOT(q1, q2),
        CNOT(q2, q1),
        CNOT(q1, q2),
        CNOT(q0, q1),
        CNOT(q1, q2),
        CNOT(q2, q1),
        CNOT(q1, q2),
        Z(q2) ** 0.25,
        Z(q1) ** 0.25,
        CNOT(q0, q1),
        Z(q0) ** 0.25,
        Z(q1) ** -0.25,
        CNOT(q0, q1),
        Y(q2) ** 0.5,
        X(q2),
    )

def test_toffoli_separate():
    circuit_orig = _toffoli_circuit()
    expected = circuit_orig.to_unitary_matrix()

    circuit = convert_circuit(circuit_orig)
    c_left = non_clifford_half(circuit)
    c_right = clifford_half(circuit)

    print(c_left)
    print(c_right)

    actual = (c_left + c_right).to_unitary_matrix()
    assert_allclose_up_to_global_phase(actual, expected, atol=1e-7)


