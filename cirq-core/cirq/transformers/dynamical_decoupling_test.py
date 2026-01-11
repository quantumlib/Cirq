# Copyright 2024 The Cirq Developers
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

from __future__ import annotations

from collections.abc import Sequence
from unittest import mock

import numpy as np
import pytest

import cirq
from cirq import add_dynamical_decoupling, CNOT, CZ, CZPowGate, H, I, measure, X, Y, Z
from cirq.transformers.dynamical_decoupling import _CellType, _Grid


def assert_sim_eq(circuit1: cirq.AbstractCircuit, circuit2: cirq.AbstractCircuit) -> None:
    # Simulate 2 circuits and compare final states.
    sampler = cirq.Simulator(dtype=np.complex128)
    psi0 = sampler.simulate(cirq.drop_terminal_measurements(circuit1)).final_state_vector
    psi1 = sampler.simulate(cirq.drop_terminal_measurements(circuit2)).final_state_vector

    assert np.isclose(np.abs(np.vdot(psi0, psi1)) ** 2, 1.0)


def assert_dd(
    input_circuit: cirq.AbstractCircuit,
    expected_circuit: str | cirq.AbstractCircuit,
    schema: str | tuple[cirq.Gate, ...] = 'DEFAULT',
    single_qubit_gate_moments_only: bool = True,
) -> None:
    transformed_circuit = add_dynamical_decoupling(
        input_circuit, schema=schema, single_qubit_gate_moments_only=single_qubit_gate_moments_only
    ).freeze()
    if isinstance(expected_circuit, str):
        cirq.testing.assert_has_diagram(transformed_circuit, expected_circuit)
    else:
        cirq.testing.assert_same_circuits(transformed_circuit, expected_circuit)
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
        cirq.drop_terminal_measurements(input_circuit),
        cirq.drop_terminal_measurements(transformed_circuit),
        {q: q for q in input_circuit.all_qubits()},
    )
    assert_sim_eq(input_circuit, transformed_circuit)


def test_classically_controlled_no_update_succeeds():
    """Test case diagrams.
    Input:
    a: ───M───I───
          ║   ║
    a: ═══@═══^═══
    """
    a = cirq.NamedQubit('a')

    input_circuit = cirq.Circuit(
        cirq.Moment(measure(a, key="a")), cirq.Moment(I(a).with_classical_controls("a"))
    )
    output_circuit = add_dynamical_decoupling(input_circuit)
    cirq.testing.assert_same_circuits(input_circuit, output_circuit)


def test_no_insertion() -> None:
    """Test case diagrams.
    Input:
    a: ───H───@───────
              │
    b: ───────X───H───
    Output:
    a: ───H───@───────
              │
    b: ───────X───H───
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert_dd(
        input_circuit=cirq.Circuit(cirq.Moment(H(a)), cirq.Moment(CNOT(a, b)), cirq.Moment(H(b))),
        expected_circuit=cirq.Circuit(
            cirq.Moment(H(a)), cirq.Moment(CNOT(a, b)), cirq.Moment(H(b))
        ),
        schema='XX_PAIR',
        single_qubit_gate_moments_only=False,
    )


@pytest.mark.parametrize(
    'schema,inserted_gates',
    [('XX_PAIR', (X, X)), ('X_XINV', (X, X**-1)), ('YY_PAIR', (Y, Y)), ('Y_YINV', (Y, Y**-1))],
)
def test_insert_provided_schema(schema: str, inserted_gates: Sequence[cirq.Gate]) -> None:
    """Test case diagrams.
    Input:
    a: ───H───@───────────M───
              │
    b: ───────X───@───@───M───
                  │   │
    c: ───────────X───X───M───
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    input_circuit = cirq.Circuit(
        cirq.Moment(H(a)),
        cirq.Moment(CNOT(a, b)),
        cirq.Moment(CNOT(b, c)),
        cirq.Moment(CNOT(b, c)),
        cirq.Moment([cirq.M(qubit) for qubit in [a, b, c]]),
    )
    expected_circuit = cirq.Circuit(
        cirq.Moment(H(a)),
        cirq.Moment(CNOT(a, b)),
        cirq.Moment(CNOT(b, c), inserted_gates[0](a)),
        cirq.Moment(CNOT(b, c), inserted_gates[1](a)),
        cirq.Moment([cirq.M(qubit) for qubit in [a, b, c]]),
    )

    # Insert one dynamical decoupling sequence in idle moments.
    assert_dd(input_circuit, expected_circuit, schema=schema, single_qubit_gate_moments_only=False)


def test_insert_by_customized_dd_sequence() -> None:
    """Test case diagrams.
        Input:
    a: ───H───@───────────────────H───
              │
    b: ───────X───@───@───@───@───H───
                  │   │   │   │
    c: ───────────X───X───X───X───H───
    Output:
    a: ───H───@───X───X───Y───Y───H───
              │
    b: ───────X───@───@───@───@───H───
                  │   │   │   │
    c: ───────────X───X───X───X───H───
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(H(a)),
            cirq.Moment(CNOT(a, b)),
            cirq.Moment(CNOT(b, c)),
            cirq.Moment(CNOT(b, c)),
            cirq.Moment(CNOT(b, c)),
            cirq.Moment(CNOT(b, c)),
            cirq.Moment([H(qubit) for qubit in [a, b, c]]),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(H(a)),
            cirq.Moment(CNOT(a, b)),
            cirq.Moment(CNOT(b, c), X(a)),
            cirq.Moment(CNOT(b, c), X(a)),
            cirq.Moment(CNOT(b, c), Y(a)),
            cirq.Moment(CNOT(b, c), Y(a)),
            cirq.Moment([H(qubit) for qubit in [a, b, c]]),
        ),
        schema=(X, X, Y, Y),
        single_qubit_gate_moments_only=False,
    )


@pytest.mark.parametrize('single_qubit_gate_moments_only', [True, False])
def test_pull_through_h_gate_case1(single_qubit_gate_moments_only: bool) -> None:
    """Test case diagrams.
    Input:
    a: ───H───────H───────@───
                          │
    b: ───H───H───H───H───X───
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(H(a), H(b)),
            cirq.Moment(H(b)),
            cirq.Moment(H(a), H(b)),
            cirq.Moment(H(b)),
            cirq.Moment(CNOT(a, b)),
        ),
        expected_circuit="""
a: ───H───X───H───PhXZ(a=-0.5,x=0,z=-1)───@───
                                          │
b: ───H───H───H───H───────────────────────X───
""",
        schema="XX_PAIR",
        single_qubit_gate_moments_only=single_qubit_gate_moments_only,
    )


@pytest.mark.parametrize('single_qubit_gate_moments_only', [True, False])
def test_pull_through_h_gate_case2(single_qubit_gate_moments_only: bool) -> None:
    """Test case diagrams.
    Input:
    a: ───H───────H───────H───

    b: ───H───H───H───H───H───
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(H(a), H(b)),
            cirq.Moment(H(b)),
            cirq.Moment(H(a), H(b)),
            cirq.Moment(H(b)),
            cirq.Moment(H(a), H(b)),
        ),
        expected_circuit="""
a: ───H───X───H───X───PhXZ(a=0.5,x=0.5,z=-1)───

b: ───H───H───H───H───H────────────────────────
""",
        schema="XX_PAIR",
        single_qubit_gate_moments_only=single_qubit_gate_moments_only,
    )


@pytest.mark.parametrize(
    'schema,error_msg_regex',
    [
        ('INVALID_SCHEMA', 'Invalid schema name.'),
        ([X], 'Invalid dynamical decoupling sequence. Expect more than one gates.'),
        (
            [X, Y],
            (
                'Invalid dynamical decoupling sequence. Expect sequence product equals identity'
                ' up to a global phase, got'
            ),
        ),
        (
            [H, H],
            (
                'Dynamical decoupling sequence should only contain gates that are essentially'
                ' Pauli gates.'
            ),
        ),
    ],
)
def test_invalid_dd_schema(schema: str | tuple[cirq.Gate, ...], error_msg_regex) -> None:
    a = cirq.NamedQubit('a')
    input_circuit = cirq.Circuit(H(a))
    with pytest.raises(ValueError, match=error_msg_regex):
        add_dynamical_decoupling(input_circuit, schema=schema, single_qubit_gate_moments_only=False)


def test_single_qubit_gate_moments_only_no_updates_succeeds() -> None:
    qubits = cirq.LineQubit.range(9)
    input_circuit = cirq.Circuit(
        cirq.Moment([H(qubits[i]) for i in [3, 4, 5]]),
        cirq.Moment(CZ(*qubits[4:6])),
        cirq.Moment(CZ(*qubits[3:5])),
        cirq.Moment([H(qubits[i]) for i in [2, 3, 5, 6]]),
        cirq.Moment(CZ(*qubits[2:4]), CNOT(*qubits[5:7])),
        cirq.Moment([H(qubits[i]) for i in [1, 2, 6, 7]]),
        cirq.Moment(CZ(*qubits[1:3]), CNOT(*qubits[6:8])),
        cirq.Moment([H(qubits[i]) for i in [0, 1, 7, 8]]),
        cirq.Moment(CZ(*qubits[0:2]), CNOT(*qubits[7:])),
    )
    add_dynamical_decoupling(input_circuit, schema='X_XINV', single_qubit_gate_moments_only=True)


def test_scattered_circuit() -> None:
    """Test case diagrams.
    Input:
    0: ───────────────────────────────H───@───H───
                                          │
    1: ───────────────────────H───@───H───@───H───
                                  │
    2: ───────────────H───@───H───@───────────H───
                          │
    3: ───H───────@───H───@───────────────────H───
                  │
    4: ───H───@───@───────────────────────────H───
              │
    5: ───H───@───────H───@───────────────────H───
                          │
    6: ───────────────H───@───H───@───────────H───
                                  │
    7: ───────────────────────H───@───H───@───H───
                                          │
    8: ───────────────────────────────H───@───H───

    Output (single_qubit_gate_moment_only_on):
    0: ───────────────────────────────H───@───H────────────────────────
                                          │
    1: ───────────────────────H───@───H───@───H────────────────────────
                                  │
    2: ───────────────H───@───H───@───X───────PhXZ(a=-0.5,x=0.5,z=0)───
                          │
    3: ───H───────@───H───@───X───────Y───────PhXZ(a=0.5,x=0.5,z=0)────
                  │
    4: ───H───@───@───X───────Y───────X───────PhXZ(a=0.5,x=0.5,z=-1)───
              │
    5: ───H───@───────H───@───X───────Y───────PhXZ(a=0.5,x=0.5,z=0)────
                          │
    6: ───────────────H───@───H───@───X───────PhXZ(a=-0.5,x=0.5,z=0)───
                                  │
    7: ───────────────────────H───@───H───@───H────────────────────────
                                          │
    8: ───────────────────────────────H───@───H────────────────────────

    Output (single_qubit_gate_moment_only_off):
    0: ───────────────────────────────H───@───H───────────────────────
                                          │
    1: ───────────────────────H───@───H───@───H───────────────────────
                                  │
    2: ───────────────H───@───H───@───X───Y───PhXZ(a=0.5,x=0.5,z=0)───
                          │
    3: ───H───X───@───H───@───Y───X───Y───X───PhXZ(a=0.5,x=0.5,z=0)───
                  │
    4: ───H───@───@───X───Y───X───Y───X───Y───H───────────────────────
              │
    5: ───H───@───X───H───@───Y───X───Y───X───PhXZ(a=0.5,x=0.5,z=0)───
                          │
    6: ───────────────H───@───H───@───X───Y───PhXZ(a=0.5,x=0.5,z=0)───
                                  │
    7: ───────────────────────H───@───H───@───H───────────────────────
                                          │
    8: ───────────────────────────────H───@───H───────────────────────
    """
    qubits = cirq.LineQubit.range(9)
    input_circuit = cirq.Circuit(
        cirq.Moment([H(qubits[i]) for i in [3, 4, 5]]),
        cirq.Moment(CZ(*qubits[4:6])),
        cirq.Moment(CZ(*qubits[3:5])),
        cirq.Moment([H(qubits[i]) for i in [2, 3, 5, 6]]),
        cirq.Moment(CZ(*qubits[2:4]), CZ(*qubits[5:7])),
        cirq.Moment([H(qubits[i]) for i in [1, 2, 6, 7]]),
        cirq.Moment(CZ(*qubits[1:3]), CZ(*qubits[6:8])),
        cirq.Moment([H(qubits[i]) for i in [0, 1, 7, 8]]),
        cirq.Moment(CZ(*qubits[0:2]), CZ(*qubits[7:])),
        cirq.Moment([H(q) for q in qubits]),
    )
    expected_circuit_single_qubit_gate_on = cirq.Circuit(
        cirq.Moment([H(qubits[i]) for i in [3, 4, 5]]),
        cirq.Moment(CZ(*qubits[4:6])),
        cirq.Moment(CZ(*qubits[3:5])),
        cirq.Moment([H(qubits[i]) for i in [2, 3, 5, 6]] + [X(qubits[4])]),
        cirq.Moment(CZ(*qubits[2:4]), CZ(*qubits[5:7])),
        cirq.Moment(
            [H(qubits[i]) for i in [1, 2, 6, 7]] + [X(qubits[i]) for i in [3, 5]] + [Y(qubits[4])]
        ),
        cirq.Moment(CZ(*qubits[1:3]), CZ(*qubits[6:8])),
        cirq.Moment(
            [H(qubits[i]) for i in [0, 1, 7, 8]]
            + [X(qubits[i]) for i in [2, 4, 6]]
            + [Y(qubits[i]) for i in [3, 5]]
        ),
        cirq.Moment(CZ(*qubits[0:2]), CZ(*qubits[7:])),
        cirq.Moment(
            [H(qubits[i]) for i in [0, 1, 7, 8]]
            + [
                cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=0).on(
                    qubits[2]
                ),
                cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0).on(
                    qubits[3]
                ),
                cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=-1).on(
                    qubits[4]
                ),
                cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0).on(
                    qubits[5]
                ),
                cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=0).on(
                    qubits[6]
                ),
            ]
        ),
    )
    expected_circuit_single_qubit_gates_off = cirq.Circuit(
        cirq.Moment([H(qubits[i]) for i in [3, 4, 5]]),
        cirq.Moment(CZ(*qubits[4:6]), X(qubits[3])),
        cirq.Moment(CZ(*qubits[3:5]), X(qubits[5])),
        cirq.Moment([H(qubits[i]) for i in [2, 3, 5, 6]] + [X(qubits[i]) for i in [4]]),
        cirq.Moment(CZ(*qubits[2:4]), CZ(*qubits[5:7]), Y(qubits[4])),
        cirq.Moment(
            [H(qubits[i]) for i in [1, 2, 6, 7]] + [Y(qubits[i]) for i in [3, 5]] + [X(qubits[4])]
        ),
        cirq.Moment(
            [CZ(*qubits[1:3]), CZ(*qubits[6:8])] + [X(qubits[i]) for i in [3, 5]] + [Y(qubits[4])]
        ),
        cirq.Moment(
            [H(qubits[i]) for i in [0, 1, 7, 8]]
            + [X(qubits[i]) for i in [2, 4, 6]]
            + [Y(qubits[i]) for i in [3, 5]]
        ),
        cirq.Moment(
            [CZ(*qubits[0:2]), CZ(*qubits[7:])]
            + [X(qubits[i]) for i in [3, 5]]
            + [Y(qubits[i]) for i in [2, 4, 6]]
        ),
        cirq.Moment(
            [H(qubits[i]) for i in [0, 1, 4, 7, 8]]
            + [
                cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0).on(
                    qubits[i]
                )
                for i in [2, 3, 5, 6]
            ]
        ),
    )
    assert_dd(
        input_circuit,
        expected_circuit_single_qubit_gate_on,
        schema='DEFAULT',
        single_qubit_gate_moments_only=True,
    )
    assert_dd(
        input_circuit,
        expected_circuit_single_qubit_gates_off,
        schema='DEFAULT',
        single_qubit_gate_moments_only=False,
    )


def test_scattered_circuit2() -> None:
    """Test case diagrams.
    Input:
    0: ───────────────────@───
                          │
    1: ───────────────@───@───
                      │
    2: ───────────@───@───────
                  │
    3: ───────@───@───────────
              │
    4: ───@───@───────────────
          │
    5: ───@───────@───────────
                  │
    6: ───────────@───@───────
                      │
    7: ───────────────@───@───
                          │
    8: ───────────────────@───
    """
    qubits = cirq.LineQubit.range(9)
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(CZ(*qubits[4:6])),
            cirq.Moment(CZ(*qubits[3:5])),
            cirq.Moment(CZ(*qubits[2:4]), CZ(*qubits[5:7])),
            cirq.Moment(CZ(*qubits[1:3]), CZ(*qubits[6:8])),
            cirq.Moment(CZ(*qubits[0:2]), CZ(*qubits[7:])),
        ),
        expected_circuit="""
0: ───────────────────@───
                      │
1: ───────────────@───@───
                  │
2: ───────────@───@───────
              │
3: ───────@───@───────────
          │
4: ───@───@───────────────
      │
5: ───@───I───@───────────
              │
6: ───────────@───@───────
                  │
7: ───────────────@───@───
                      │
8: ───────────────────@───
""",
        schema="XX_PAIR",
        single_qubit_gate_moments_only=False,
    )


def test_pull_through_chain() -> None:
    """Test case diagrams.
    Input:
    0: ───X───────×───────────X───
                  │
    1: ───────Y───×───×───────X───
                      │
    2: ───────────────×───×───X───
                          │
    3: ───────────────────×───X───
    """
    qubits = cirq.LineQubit.range(4)
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(X(qubits[0])),
            cirq.Moment(Y(qubits[1])),
            cirq.Moment(cirq.SWAP(*qubits[0:2])),
            cirq.Moment(cirq.SWAP(*qubits[1:3])),
            cirq.Moment(cirq.SWAP(*qubits[2:4])),
            cirq.Moment([X(qubits[i]) for i in range(4)]),
        ),
        expected_circuit="""
0: ───X───X───×───X───X───X───
              │
1: ───────Y───×───×───X───I───
                  │
2: ───────────────×───×───X───
                      │
3: ───────────────────×───I───
""",
        schema='XX_PAIR',
        single_qubit_gate_moments_only=False,
    )


def test_multiple_clifford_pieces_case1() -> None:
    """Test case diagrams.
    Input:
    a: ───H───────H───────@───────────H───────H───
                          │
    b: ───H───H───H───H───@^0.5───H───H───H───H───
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(H(a), H(b)),
            cirq.Moment(H(b)),
            cirq.Moment(H(a), H(b)),
            cirq.Moment(H(b)),
            cirq.Moment(CZPowGate(exponent=0.5).on(a, b)),
            cirq.Moment(H(b)),
            cirq.Moment(H(a), H(b)),
            cirq.Moment(H(b)),
            cirq.Moment(H(a), H(b)),
        ),
        expected_circuit="""
a: ───H───X───H───PhXZ(a=-0.5,x=0,z=-1)───@───────X───H───X───PhXZ(a=0.5,x=0.5,z=-1)───
                                          │
b: ───H───H───H───H───────────────────────@^0.5───H───H───H───H────────────────────────
""",
        schema="XX_PAIR",
    )


def test_multiple_clifford_pieces_case2() -> None:
    """Test case diagrams.
    Input:
    a: ───@───PhXZ(a=0.3,x=0.2,z=0)───PhXZ(a=0.3,x=0.2,z=0)───PhXZ(a=0.3,x=0.2,z=0)───@───
          │                                                                           │
    b: ───@───────────────────────────────────────────────────────────────────────────@───
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    phased_xz_gate = cirq.PhasedXZGate(axis_phase_exponent=0.3, x_exponent=0.2, z_exponent=0)

    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(CZ(a, b)),
            cirq.Moment(phased_xz_gate.on(a)),
            cirq.Moment(phased_xz_gate.on(a)),
            cirq.Moment(phased_xz_gate.on(a)),
            cirq.Moment(CZ(a, b)),
        ),
        expected_circuit="""
a: ───@───PhXZ(a=0.3,x=0.2,z=0)───PhXZ(a=0.3,x=0.2,z=0)───PhXZ(a=0.3,x=0.2,z=0)───@───
      │                                                                           │
b: ───@───X───────────────────────X───────────────────────I───────────────────────@───
""",
        schema='XX_PAIR',
        single_qubit_gate_moments_only=False,
    )


def test_absorb_remaining_dd_sequence() -> None:
    """Test case diagrams.
    Input:
    a: ───H───────H───@───@───────
                      │   │
    b: ───H───H───H───X───@^0.5───

    c: ───H───────────────H───────
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(H(a), H(b), H(c)),
            cirq.Moment(H(b)),
            cirq.Moment(H(b), H(a)),
            cirq.Moment(CNOT(a, b)),
            cirq.Moment(CZPowGate(exponent=0.5).on(a, b), H(c)),
        ),
        expected_circuit="""
a: ───H───X───PhXZ(a=-0.5,x=0.5,z=0)───@───@───────
                                       │   │
b: ───H───H───H────────────────────────X───@^0.5───

c: ───H───X───X────────────────────────────H───────        
""",
        schema="XX_PAIR",
    )


def test_with_non_clifford_measurements() -> None:
    """Test case diagrams.
    Input:
    0: ───────────H───@───H───M───
                      │
    1: ───H───@───────@───────M───
              │
    2: ───H───@───H───@───────M───
                      │
    3: ───────────H───@───H───M───
    """
    qubits = cirq.LineQubit.range(4)
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment([H(qubits[i]) for i in [1, 2]]),
            cirq.Moment(CZ(*qubits[1:3])),
            cirq.Moment([H(qubits[i]) for i in [0, 2, 3]]),
            cirq.Moment(CZ(*qubits[0:2]), CZ(*qubits[2:])),
            cirq.Moment([H(qubits[i]) for i in [0, 3]]),
            cirq.Moment([cirq.M(qubits[i]) for i in [0, 1, 2, 3]]),
        ),
        expected_circuit="""
0: ───────────H───@───PhXZ(a=0.5,x=0.5,z=0)───M───
                  │
1: ───H───@───X───@───X───────────────────────M───
          │
2: ───H───@───H───@───I───────────────────────M───
                  │
3: ───────────H───@───H───────────────────────M───
""",
        schema="XX_PAIR",
        single_qubit_gate_moments_only=True,
    )


def test_cross_clifford_pieces_filling_merge() -> None:
    """Test case diagrams.
    Input:
    0: ─────────────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───H───
                                                                  │                                                   │
    1: ─────────────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───H───

    2: ───PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────────H───
                                    │                             │                                                   │
    3: ─────────────────────────────┼───PhXZ(a=0.2,x=0.2,z=0.1)───@───────────────────────────────────────────────────@─────────────────────────────H───
                                    │
    4: ─────────────────────────────┼─────────────────────────────@─────────────────────────────────────────────────────────────────────────────────H───
                                    │                             │
    5: ───PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───H───
                                                                                                                      │
    6: ───────────────────────────────────────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)─────────────────────────────@───PhXZ(a=0.2,x=0.2,z=0.1)───H───
    """  # noqa: E501
    qubits = cirq.LineQubit.range(7)
    phased_xz_gate = cirq.PhasedXZGate(axis_phase_exponent=0.2, x_exponent=0.2, z_exponent=0.1)
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment([phased_xz_gate.on(qubits[i]) for i in [2, 5]]),
            cirq.Moment(CZ(qubits[2], qubits[5])),
            cirq.Moment([phased_xz_gate.on(qubits[i]) for i in [0, 1, 2, 3, 5]]),
            cirq.Moment(
                [CZ(qubits[i0], qubits[i1]) for i0, i1 in [(0, 1), (2, 3), (4, 5)]]
                + [phased_xz_gate.on(qubits[6])]
            ),
            cirq.Moment([phased_xz_gate.on(qubits[i]) for i in [0, 1, 2, 5]]),
            cirq.Moment([CZ(qubits[i0], qubits[i1]) for i0, i1 in [(0, 1), (2, 3), (5, 6)]]),
            cirq.Moment([phased_xz_gate.on(qubits[i]) for i in [0, 1, 5, 6]]),
            cirq.Moment([H.on(q) for q in qubits]),
        ),
        expected_circuit="""
0: ─────────────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───H────────────────────────
                                                              │                                                   │
1: ─────────────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───H────────────────────────

2: ───PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───X─────────────────────────PhXZ(a=0.5,x=0.5,z=-1)───
                                │                             │                                                   │
3: ─────────────────────────────┼───PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────X─────────────────────────@───Y─────────────────────────PhXZ(a=0.5,x=0.5,z=0)────
                                │
4: ─────────────────────────────┼─────────────────────────────@─────────────────────────X─────────────────────────────Y─────────────────────────PhXZ(a=0.5,x=0.5,z=0)────
                                │                             │
5: ───PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───H────────────────────────
                                                                                                                  │
6: ───────────────────────────────────────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───I─────────────────────────@───PhXZ(a=0.2,x=0.2,z=0.1)───H────────────────────────
""",  # noqa: E501
    )


def test_pull_through_phxz_gate_case1() -> None:
    """Test case diagrams.

    Input:
    a: ───H───────PhXZ(a=0.25,x=-1,z=0)───────@───
                                              │
    b: ───H───H───H───────────────────────H───X───
    Output: expected circuit diagram below.
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    phxz = cirq.PhasedXZGate(axis_phase_exponent=0.25, x_exponent=-1, z_exponent=0)
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(H(a), H(b)),
            cirq.Moment(H(b)),
            cirq.Moment(phxz(a), H(b)),
            cirq.Moment(H(b)),
            cirq.Moment(CNOT(a, b)),
        ),
        expected_circuit="""
a: ───H───X───PhXZ(a=0.25,x=-1,z=0)───PhXZ(a=0.5,x=1,z=0)───@───
                                                            │
b: ───H───H───H───────────────────────H─────────────────────X───
""",
        schema="XX_PAIR",
    )


def test_pull_through_phxz_gate_case2() -> None:
    """Test case diagrams.

    Input:
    a: ───H───────PhXZ(a=0.2,x=-1,z=0)───────@───
                                              │
    b: ───H───H───H───────────────────────H───X───
    Output: expected circuit diagram below.
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    phxz = cirq.PhasedXZGate(axis_phase_exponent=0.2, x_exponent=-1, z_exponent=0)
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(H(a), H(b)),
            cirq.Moment(H(b)),
            cirq.Moment(phxz(a), H(b)),
            cirq.Moment(H(b)),
            cirq.Moment(CNOT(a, b)),
        ),
        expected_circuit="""
a: ───H───I───PhXZ(a=0.2,x=-1,z=0)───I───@───
                                         │
b: ───H───H───H──────────────────────H───X───
""",
        schema="XX_PAIR",
    )


def test_merge_before_non_cliffords() -> None:
    """Test case diagrams.
    Input circuit:
    0: ───X──────────────────────────────────────────────────M───

    1: ───X───────PhXZ(a=-1,x=0,z=-0.5)───FSim(0, 0.0637π)───M───
                                          │
    2: ───X───X───S───────────────────────FSim(0, 0.0637π)───M───
    """
    q0, q1, q2 = cirq.LineQubit.range(3)
    input_circuit = cirq.Circuit(
        cirq.Moment([X(q) for q in [q0, q1, q2]]),
        cirq.Moment(X(q2)),
        cirq.Moment(
            cirq.PhasedXZGate(axis_phase_exponent=-1, x_exponent=0, z_exponent=-0.5).on(q1),
            (Z**0.5).on(q2),
        ),
        cirq.Moment(cirq.FSimGate(theta=0, phi=0.2).on(q1, q2)),
        cirq.Moment([cirq.M(q) for q in [q0, q1, q2]]),
    )
    assert_dd(
        input_circuit=input_circuit,
        expected_circuit="""
0: ───X───X───X──────────────────────────────────────────M───

1: ───X───X───PhXZ(a=-1.25,x=1,z=0)───FSim(0, 0.0637π)───M───
                                      │
2: ───X───X───S───────────────────────FSim(0, 0.0637π)───M───
""",
        schema="XX_PAIR",
    )


def test_runtime_error_if_pulled_through_not_empty_mocked() -> None:
    """Tests that a RuntimeError is raised if pulled_through is not empty at the end.

    This test explicitly mocks the internal state to simulate a scenario where
    the `pulled_through` PauliString is not empty after processing all moments.
    Under normal operation, the `_Grid` and `add_dynamical_decoupling`
    logic should ensure `pulled_through` is always empty at the end, making
    this RuntimeError theoretically unreachable. This test verifies the
    defensive check itself.
    """
    q0: cirq.Qid = cirq.NamedQubit('q0')
    circuit = cirq.FrozenCircuit(cirq.Moment(cirq.I(q0)))  # A minimal circuit

    # Create a mock _Grid instance that would lead to an unabsorbed Pauli.
    # We need an INSERTABLE slot, but no STOP label, and no self-cancellation.
    # To achieve this, we'll mock the `pulled_through.after` method to *not* clear it.
    # This is a deep mock to hit the specific RuntimeError line.

    # First, create a _Grid that allows insertion but no stopping.
    # This is a hypothetical scenario that `_Grid.from_circuit` should prevent.
    mock_gate_types = {q0: {0: _CellType.INSERTABLE}}
    mock_need_to_stop = {q0: {0: False}}  # Crucially, no stop gate

    mock_labeled_circuit = _Grid(
        gate_types=mock_gate_types, need_to_stop=mock_need_to_stop, circuit=circuit
    )

    # Mock _Grid.from_circuit to return our custom mock
    with mock.patch(
        'cirq.transformers.dynamical_decoupling._Grid.from_circuit',
        return_value=mock_labeled_circuit,
    ):
        # Mock the PauliString.after method to ensure `pulled_through` remains non-empty.
        with mock.patch('cirq.ops.PauliString.after', return_value=cirq.PauliString({q0: cirq.X})):
            with pytest.raises(
                RuntimeError, match="Expect empty remaining Paulis after the dd insertion."
            ):
                add_dynamical_decoupling(
                    circuit, schema='XX_PAIR', single_qubit_gate_moments_only=True
                )


def test_labeled_circuit_str():
    """Input circuit:
    0: ───X──────────────────────────────────────────────────M───

    1: ───X───────PhXZ(a=-1,x=0,z=-0.5)───FSim(0, 0.0637π)───M───
                                          │
    2: ───X───X───S───────────────────────FSim(0, 0.0637π)───M───
    """
    q0, q1, q2 = cirq.LineQubit.range(3)
    input_circuit = cirq.Circuit(
        cirq.Moment([X(q) for q in [q0, q1, q2]]),
        cirq.Moment(X(q2)),
        cirq.Moment(
            cirq.PhasedXZGate(axis_phase_exponent=-1, x_exponent=0, z_exponent=-0.5).on(q1),
            (Z**0.5).on(q2),
        ),
        cirq.Moment(cirq.FSimGate(theta=0, phi=0.2).on(q1, q2)),
        cirq.Moment([cirq.M(q) for q in [q0, q1, q2]]),
    )
    labeled_circuit = _Grid.from_circuit(input_circuit, single_qubit_gate_moments_only=True)
    assert str(labeled_circuit) == (
        """Grid Repr:
     |  0  |  1  |  2  |  3  |  4  |
-----+-----+-----+-----+-----+-----+
q(0) |  d  |  i  | i,s |  d  |  w  |
q(1) |  d  |  i  | d,s |  w  |  w  |
q(2) |  d  |  d  | d,s |  w  |  w  |"""
    )


def test_labeled_circuit_str_empty():
    """Tests the __str__ method of _Grid for empty and no-qubit circuits."""
    # Test case for an empty circuit (no moments, no qubits)
    empty_circuit = cirq.Circuit()
    labeled_empty = _Grid.from_circuit(empty_circuit, single_qubit_gate_moments_only=True)
    assert str(labeled_empty) == "Grid(empty)"


def test_add_dynamical_decoupling_with_deep_context_raises_error():
    """Tests that add_dynamical_decoupling raises an error with deep context."""
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.H(q))
    context = cirq.TransformerContext(deep=True)
    with pytest.raises(ValueError, match="Deep transformation is not supported."):
        add_dynamical_decoupling(circuit, context=context)


def test_context_logger():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.H(q))
    mock_logger = mock.MagicMock(spec=cirq.TransformerLogger)
    context = cirq.TransformerContext(logger=mock_logger)

    add_dynamical_decoupling(circuit, context=context)

    mock_logger.log.assert_called_once()
    assert "Preprocessed input circuit grid repr:" in mock_logger.log.call_args[0][0]
