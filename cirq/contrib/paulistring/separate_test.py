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

import cirq
from cirq.contrib.paulistring import (
    convert_and_separate_circuit,)


def test_toffoli_separate():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.testing.nonoptimal_toffoli_circuit(q0, q1, q2)

    c_left, c_right = convert_and_separate_circuit(circuit)

    cirq.testing.assert_allclose_up_to_global_phase(
        circuit.unitary(),
        (c_left + c_right).unitary(),
        atol=1e-7,
    )

    assert all(
        isinstance(op, cirq.PauliStringPhasor)
        for op in c_left.all_operations())
    assert all(
        isinstance(op, cirq.GateOperation) and
        isinstance(op.gate, (cirq.SingleQubitCliffordGate, cirq.CZPowGate))
        for op in c_right.all_operations())
