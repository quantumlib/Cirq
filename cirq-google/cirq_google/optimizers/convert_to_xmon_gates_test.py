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
import cirq_google


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
    with cirq.testing.assert_deprecated("Use cirq.optimize_for_target_gateset", deadline='v1.0'):
        cirq_google.ConvertToXmonGates().optimize_circuit(c)
    cirq.testing.assert_has_diagram(c, '(0, 0): ───PhX(1)───PhX(1)───')

    cirq.protocols.decompose(c)


q = cirq.GridQubit.rect(1, 3)
matrix_gate = cirq.MatrixGate(cirq.testing.random_unitary(2))


def test_bad_operation():
    c = cirq.Circuit(NonNativeGate().on(q[0]))
    with pytest.raises(TypeError):
        with cirq.testing.assert_deprecated(
            "Use cirq.optimize_for_target_gateset", deadline='v1.0'
        ):
            cirq_google.ConvertToXmonGates().optimize_circuit(c)


@pytest.mark.parametrize(
    'op, is_valid',
    [
        (cirq.CircuitOperation(cirq.FrozenCircuit(matrix_gate(q[0]))), False),
        (matrix_gate(q[0]), True),
        (matrix_gate(q[0]).with_tags('test_tags'), True),
        (matrix_gate(q[0]).controlled_by(q[1]), True),
        (matrix_gate(q[0]).controlled_by(q[1]).with_tags('test_tags'), True),
        (matrix_gate(q[0]).with_tags('test_tags').controlled_by(q[1]), True),
    ],
)
def test_supported_operation(op, is_valid):
    c = cirq.Circuit(op)
    with cirq.testing.assert_deprecated("Use cirq.optimize_for_target_gateset", deadline='v1.0'):
        assert (cirq_google.ConvertToXmonGates().optimization_at(c, 0, op) is not None) == is_valid
