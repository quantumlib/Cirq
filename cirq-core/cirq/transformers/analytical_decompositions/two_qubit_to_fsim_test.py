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

import itertools
import random
from typing import Any

import numpy as np
import pytest

import cirq
from cirq.transformers.analytical_decompositions.two_qubit_to_fsim import (
    _decompose_two_qubit_interaction_into_two_b_gates,
    _decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops,
    _sticky_0_to_1,
    _B,
)

ALLOW_DEPRECATION_IN_TEST = 'ALLOW_DEPRECATION_IN_TEST'


def test_deprecated_submodule():
    with cirq.testing.assert_deprecated(
        "Use cirq.transformers.analytical_decompositions.two_qubit_to_fsim instead",
        deadline="v0.16",
    ):
        _ = cirq.optimizers.two_qubit_to_fsim.decompose_two_qubit_interaction_into_four_fsim_gates


UNITARY_OBJS = [
    cirq.IdentityGate(2),
    cirq.XX ** 0.25,
    cirq.CNOT,
    cirq.CNOT(*cirq.LineQubit.range(2)),
    cirq.CNOT(*cirq.LineQubit.range(2)[::-1]),
    cirq.ISWAP,
    cirq.SWAP,
    cirq.FSimGate(theta=np.pi / 6, phi=np.pi / 6),
] + [cirq.testing.random_unitary(4) for _ in range(5)]

FEASIBLE_FSIM_GATES = [
    cirq.ISWAP,
    cirq.FSimGate(np.pi / 2, 0),
    cirq.FSimGate(-np.pi / 2, 0),
    cirq.FSimGate(np.pi / 2, np.pi / 6),
    cirq.FSimGate(np.pi / 2, -np.pi / 6),
    cirq.FSimGate(5 * np.pi / 9, -np.pi / 6),
    cirq.FSimGate(5 * np.pi / 9, 0),
    cirq.FSimGate(4 * np.pi / 9, -np.pi / 6),
    cirq.FSimGate(4 * np.pi / 9, 0),
    cirq.FSimGate(-4 * np.pi / 9, 0),
    # Extreme points.
    cirq.FSimGate(np.pi * 3 / 8, -np.pi / 4),
    cirq.FSimGate(np.pi * 5 / 8, -np.pi / 4),
    cirq.FSimGate(np.pi * 3 / 8, +np.pi / 4),
    cirq.FSimGate(np.pi * 5 / 8, +np.pi / 4),
] + [
    cirq.FSimGate(
        theta=random.uniform(np.pi * 3 / 8, np.pi * 5 / 8),
        phi=random.uniform(-np.pi / 4, np.pi / 4),
    )
    for _ in range(5)
]


@pytest.mark.parametrize('obj', UNITARY_OBJS)
def test_decompose_two_qubit_interaction_into_two_b_gates(obj: Any):
    circuit = cirq.Circuit(
        _decompose_two_qubit_interaction_into_two_b_gates(obj, qubits=cirq.LineQubit.range(2))
    )
    desired_unitary = obj if isinstance(obj, np.ndarray) else cirq.unitary(obj)
    for operation in circuit.all_operations():
        assert len(operation.qubits) < 2 or operation.gate == _B
    assert cirq.approx_eq(cirq.unitary(circuit), desired_unitary, atol=1e-6)


def test_decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops_fail():
    c = _decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops(
        qubits=cirq.LineQubit.range(2),
        fsim_gate=cirq.FSimGate(theta=np.pi / 2, phi=0),
        canonical_x_kak_coefficient=np.pi / 4,
        canonical_y_kak_coefficient=np.pi / 8,
    )
    np.testing.assert_allclose(
        cirq.kak_decomposition(cirq.Circuit(c)).interaction_coefficients, [np.pi / 4, np.pi / 8, 0]
    )

    with pytest.raises(ValueError, match='Failed to synthesize'):
        _ = _decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops(
            qubits=cirq.LineQubit.range(2),
            fsim_gate=cirq.FSimGate(theta=np.pi / 5, phi=0),
            canonical_x_kak_coefficient=np.pi / 4,
            canonical_y_kak_coefficient=np.pi / 8,
        )


@pytest.mark.parametrize('obj,fsim_gate', itertools.product(UNITARY_OBJS, FEASIBLE_FSIM_GATES))
def test_decompose_two_qubit_interaction_into_four_fsim_gates_equivalence(
    obj: Any, fsim_gate: cirq.FSimGate
):
    qubits = obj.qubits if isinstance(obj, cirq.Operation) else cirq.LineQubit.range(2)
    circuit = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(obj, fsim_gate=fsim_gate)
    desired_unitary = obj if isinstance(obj, np.ndarray) else cirq.unitary(obj)
    for operation in circuit.all_operations():
        assert len(operation.qubits) < 2 or operation.gate == fsim_gate
    assert len(circuit) <= 4 * 3 + 5
    assert cirq.approx_eq(circuit.unitary(qubit_order=qubits), desired_unitary, atol=1e-6)


def test_decompose_two_qubit_interaction_into_four_fsim_gates_validate():
    iswap = cirq.FSimGate(theta=np.pi / 2, phi=0)
    with pytest.raises(ValueError, match='fsim_gate.theta'):
        cirq.decompose_two_qubit_interaction_into_four_fsim_gates(
            np.eye(4), fsim_gate=cirq.FSimGate(theta=np.pi / 10, phi=0)
        )
    with pytest.raises(ValueError, match='fsim_gate.phi'):
        cirq.decompose_two_qubit_interaction_into_four_fsim_gates(
            np.eye(4), fsim_gate=cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 3)
        )
    with pytest.raises(ValueError, match='pair of qubits'):
        cirq.decompose_two_qubit_interaction_into_four_fsim_gates(
            np.eye(4), fsim_gate=iswap, qubits=cirq.LineQubit.range(3)
        )


def test_decompose_two_qubit_interaction_into_four_fsim_gates():
    iswap = cirq.FSimGate(theta=np.pi / 2, phi=0)

    # Defaults to line qubits.
    c = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(np.eye(4), fsim_gate=iswap)
    assert set(c.all_qubits()) == set(cirq.LineQubit.range(2))

    # Infers from operation but not gate.
    c = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(cirq.CZ, fsim_gate=iswap)
    assert set(c.all_qubits()) == set(cirq.LineQubit.range(2))
    c = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(
        cirq.CZ(*cirq.LineQubit.range(20, 22)), fsim_gate=iswap
    )
    assert set(c.all_qubits()) == set(cirq.LineQubit.range(20, 22))

    # Can override.
    c = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(
        np.eye(4), fsim_gate=iswap, qubits=cirq.LineQubit.range(10, 12)
    )
    assert set(c.all_qubits()) == set(cirq.LineQubit.range(10, 12))
    c = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(
        cirq.CZ(*cirq.LineQubit.range(20, 22)),
        fsim_gate=iswap,
        qubits=cirq.LineQubit.range(10, 12),
    )
    assert set(c.all_qubits()) == set(cirq.LineQubit.range(10, 12))


def test_sticky_0_to_1():
    assert _sticky_0_to_1(-1, atol=1e-8) is None

    assert _sticky_0_to_1(-1e-6, atol=1e-8) is None
    assert _sticky_0_to_1(-1e-10, atol=1e-8) == 0
    assert _sticky_0_to_1(0, atol=1e-8) == 0
    assert _sticky_0_to_1(1e-10, atol=1e-8) == 1e-10
    assert _sticky_0_to_1(1e-6, atol=1e-8) == 1e-6

    assert _sticky_0_to_1(0.5, atol=1e-8) == 0.5

    assert _sticky_0_to_1(1 - 1e-6, atol=1e-8) == 1 - 1e-6
    assert _sticky_0_to_1(1 - 1e-10, atol=1e-8) == 1 - 1e-10
    assert _sticky_0_to_1(1, atol=1e-8) == 1
    assert _sticky_0_to_1(1 + 1e-10, atol=1e-8) == 1
    assert _sticky_0_to_1(1 + 1e-6, atol=1e-8) is None

    assert _sticky_0_to_1(2, atol=1e-8) is None

    assert _sticky_0_to_1(-0.1, atol=0.5) == 0
