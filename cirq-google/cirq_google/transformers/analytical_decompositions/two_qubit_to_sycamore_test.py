# Copyright 2022 The Cirq Developers
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
import pytest
import numpy as np
import cirq_google as cg
import sympy

EXPECTED_TARGET_GATESET = cirq.Gateset(cirq.AnyUnitaryGateFamily(1), cg.SYC)


def assert_implements(circuit: cirq.Circuit, target_op: cirq.Operation):
    assert all(op in EXPECTED_TARGET_GATESET for op in circuit.all_operations())
    assert sum(1 for _ in circuit.findall_operations(lambda e: len(e.qubits) > 2)) <= 6
    circuit.append(cirq.I.on_each(*target_op.qubits))
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(circuit), cirq.unitary(target_op), atol=1e-7
    )


_THETA = sympy.Symbol('theta')
_ALL_EXPS = np.linspace(0, 1, 10)
_QUBITS = cirq.LineQubit.range(2)


@pytest.mark.parametrize(
    'op, theta_range',
    [
        (
            cirq.CircuitOperation(
                cirq.FrozenCircuit(cirq.SWAP(*_QUBITS), cirq.ZZ(*_QUBITS) ** _THETA)
            ),
            _ALL_EXPS,
        ),
        (
            cirq.CircuitOperation(
                cirq.FrozenCircuit(cirq.ZZ(*_QUBITS) ** _THETA, cirq.SWAP(*_QUBITS))
            ),
            _ALL_EXPS,
        ),
        (cirq.PhasedISwapPowGate(exponent=1, phase_exponent=_THETA).on(*_QUBITS), _ALL_EXPS),
        (cirq.PhasedISwapPowGate(exponent=_THETA, phase_exponent=0.25).on(*_QUBITS), _ALL_EXPS),
        (cirq.CNOT(*_QUBITS) ** _THETA, _ALL_EXPS),
        (cirq.CZ(*_QUBITS) ** _THETA, _ALL_EXPS),
        (cirq.ZZ(*_QUBITS) ** _THETA, _ALL_EXPS),
        (cirq.SWAP(*_QUBITS) ** _THETA, [1]),
        (cirq.ISWAP(*_QUBITS) ** _THETA, [1]),
    ],
)
def test_known_two_qubit_op_decomposition(op, theta_range):
    for theta_val in theta_range:
        op_resolved = cirq.resolve_parameters(op, {'theta': theta_val}, recursive=False)
        known_2q_circuit = cirq.Circuit(cg.known_2q_op_to_sycamore_operations(op_resolved))
        matrix_2q_circuit = cirq.Circuit(
            cg.two_qubit_matrix_to_sycamore_operations(
                _QUBITS[0], _QUBITS[1], cirq.unitary(op_resolved)
            )
        )
        assert_implements(known_2q_circuit, op_resolved)
        assert_implements(matrix_2q_circuit, op_resolved)


@pytest.mark.parametrize(
    'op',
    [
        cirq.CircuitOperation(
            cirq.FrozenCircuit(cirq.SWAP(*_QUBITS), cirq.ZZ(*_QUBITS), cirq.SWAP(*_QUBITS))
        ),
        cirq.X(_QUBITS[0]),
        cirq.XX(*_QUBITS) ** _THETA,
        cirq.FSimGate(0.25, 0.85).on(*_QUBITS),
        cirq.XX(*_QUBITS),
        cirq.YY(*_QUBITS),
        *[cirq.testing.random_unitary(4, random_state=1234) for _ in range(10)],
    ],
)
def test_unknown_two_qubit_op_decomposition(op):
    assert cg.known_2q_op_to_sycamore_operations(op) is None
    if cirq.has_unitary(op) and cirq.num_qubits(op) == 2:
        matrix_2q_circuit = cirq.Circuit(
            cg.two_qubit_matrix_to_sycamore_operations(_QUBITS[0], _QUBITS[1], cirq.unitary(op))
        )
        assert_implements(matrix_2q_circuit, op)
