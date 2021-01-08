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
import numpy as np
import pytest

import cirq


class OtherX(cirq.SingleQubitGate):
    def _unitary_(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]])

    def _decompose_(self, qubits):
        # Coverage explicitly ignored since we are checking that we don't
        # run this line and fall into an infinite loop.
        return OtherOtherX().on(*qubits)  # coverage:ignore


class OtherOtherX(cirq.SingleQubitGate):
    def _decompose_(self, qubits):
        return OtherX().on(*qubits)


class NonNativeGate(cirq.SingleQubitGate):
    pass


def test_avoids_infinite_cycle_when_matrix_available():
    q = cirq.GridQubit(0, 0)
    c = cirq.Circuit(OtherX().on(q), OtherOtherX().on(q))
    cirq.google.ConvertToXmonGates().optimize_circuit(c)
    cirq.testing.assert_has_diagram(c, '(0, 0): ───PhX(1)───PhX(1)───')
    cirq.protocols.decompose(c)


def test_bad_operation():
    qubits = cirq.GridQubit.rect(1, 3)
    c = cirq.Circuit(NonNativeGate().on(qubits[0]))
    with pytest.raises(TypeError):
        cirq.google.ConvertToXmonGates().optimize_circuit(c)
