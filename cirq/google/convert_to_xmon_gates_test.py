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

import cirq


class OtherX(cirq.KnownMatrixGate, cirq.CompositeGate):
    def matrix(self):
        return np.array([[0, 1], [1, 0]])

    def default_decompose(self, qubits):
        return OtherOtherX().on(*qubits)


class OtherOtherX(cirq.KnownMatrixGate, cirq.CompositeGate):
    def matrix(self):
        return np.array([[0, 1], [1, 0]])

    def default_decompose(self, qubits):
        return OtherX().on(*qubits)


def test_avoids_infinite_cycle_when_matrix_available():
    q = cirq.google.XmonQubit(0, 0)
    c = cirq.Circuit.from_ops(OtherX().on(q), OtherOtherX().on(q))
    cirq.google.ConvertToXmonGates().optimize_circuit(c)
    assert c.to_text_diagram() == '(0, 0): ───X───X───'
