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

from typing import Sequence, Tuple, Union
import cirq
from cirq import add_dynamical_decoupling
import pytest
import numpy as np


def assert_sim_eq(circuit1: 'cirq.AbstractCircuit', circuit2: 'cirq.AbstractCircuit'):
    # Simulate 2 circuits and compare final states.
    sampler = cirq.Simulator(dtype=np.complex128)
    psi0 = sampler.simulate(cirq.drop_terminal_measurements(circuit1)).final_state_vector
    psi1 = sampler.simulate(cirq.drop_terminal_measurements(circuit2)).final_state_vector

    assert np.isclose(np.abs(np.vdot(psi0, psi1)) ** 2, 1.0)


def assert_dd(
    input_circuit: 'cirq.AbstractCircuit',
    expected_circuit: Union[str, 'cirq.AbstractCircuit'],
    schema: Union[str, Tuple['cirq.Gate', ...]] = 'DEFAULT',
    single_qubit_gate_moments_only: bool = True,
):
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


def test_no_insertion():
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
        input_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a)), cirq.Moment(cirq.CNOT(a, b)), cirq.Moment(cirq.H(b))
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a)), cirq.Moment(cirq.CNOT(a, b)), cirq.Moment(cirq.H(b))
        ),
        schema='XX_PAIR',
        single_qubit_gate_moments_only=False,
    )


@pytest.mark.parametrize(
    'schema,inserted_gates',
    [
        ('XX_PAIR', (cirq.X, cirq.X)),
        ('X_XINV', (cirq.X, cirq.X**-1)),
        ('YY_PAIR', (cirq.Y, cirq.Y)),
        ('Y_YINV', (cirq.Y, cirq.Y**-1)),
    ],
)
def test_insert_provided_schema(schema: str, inserted_gates: Sequence['cirq.Gate']):
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
        cirq.Moment(cirq.H(a)),
        cirq.Moment(cirq.CNOT(a, b)),
        cirq.Moment(cirq.CNOT(b, c)),
        cirq.Moment(cirq.CNOT(b, c)),
        cirq.Moment([cirq.M(qubit) for qubit in [a, b, c]]),
    )
    expected_circuit = cirq.Circuit(
        cirq.Moment(cirq.H(a)),
        cirq.Moment(cirq.CNOT(a, b)),
        cirq.Moment(cirq.CNOT(b, c), inserted_gates[0](a)),
        cirq.Moment(cirq.CNOT(b, c), inserted_gates[1](a)),
        cirq.Moment([cirq.M(qubit) for qubit in [a, b, c]]),
    )

    # Insert one dynamical decoupling sequence in idle moments.
    assert_dd(input_circuit, expected_circuit, schema=schema, single_qubit_gate_moments_only=False)


def test_insert_by_customized_dd_sequence():
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
            cirq.Moment(cirq.H(a)),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(cirq.CNOT(b, c)),
            cirq.Moment(cirq.CNOT(b, c)),
            cirq.Moment(cirq.CNOT(b, c)),
            cirq.Moment(cirq.CNOT(b, c)),
            cirq.Moment([cirq.H(qubit) for qubit in [a, b, c]]),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a)),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(cirq.CNOT(b, c), cirq.X(a)),
            cirq.Moment(cirq.CNOT(b, c), cirq.X(a)),
            cirq.Moment(cirq.CNOT(b, c), cirq.Y(a)),
            cirq.Moment(cirq.CNOT(b, c), cirq.Y(a)),
            cirq.Moment([cirq.H(qubit) for qubit in [a, b, c]]),
        ),
        schema=[cirq.X, cirq.X, cirq.Y, cirq.Y],
        single_qubit_gate_moments_only=False,
    )


@pytest.mark.parametrize('single_qubit_gate_moments_only', [True, False])
def test_pull_through_h_gate_case1(single_qubit_gate_moments_only: bool):
    """Test case diagrams.
    Input:
    a: ───H───────H───────@───
                          │
    b: ───H───H───H───H───X───
    Output:
    a: ───H───X───H───X───@───Y───
                          │
    b: ───H───H───H───H───X───X───
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b)),
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b)),
            cirq.Moment(cirq.CNOT(a, b)),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b), cirq.X(a)),
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b), cirq.X(a)),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(cirq.Y(a), cirq.X(b)),
        ),
        schema="XX_PAIR",
        single_qubit_gate_moments_only=single_qubit_gate_moments_only,
    )


@pytest.mark.parametrize('single_qubit_gate_moments_only', [True, False])
def test_pull_through_h_gate_case2(single_qubit_gate_moments_only: bool):
    """Test case diagrams.
    Input:
    a: ───H───────H───────H───

    b: ───H───H───H───H───H───
    Output:
    a: ───H───X───H───X───PhXZ(a=0.5,x=0.5,z=1)───

    b: ───H───H───H───H───H───────────────────────
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b)),
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b)),
            cirq.Moment(cirq.H(a), cirq.H(b)),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b), cirq.X(a)),
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b), cirq.X(a)),
            cirq.Moment(
                cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=1).on(a),
                cirq.H(b),
            ),
        ),
        schema="XX_PAIR",
        single_qubit_gate_moments_only=single_qubit_gate_moments_only,
    )


@pytest.mark.parametrize(
    'schema,error_msg_regex',
    [
        ('INVALID_SCHEMA', 'Invalid schema name.'),
        ([cirq.X], 'Invalid dynamical decoupling sequence. Expect more than one gates.'),
        (
            [cirq.X, cirq.Y],
            'Invalid dynamical decoupling sequence. Expect sequence production equals identity'
            ' up to a global phase, got',
        ),
        (
            [cirq.H, cirq.H],
            'Dynamical decoupling sequence should only contain gates that are essentially'
            ' Pauli gates.',
        ),
    ],
)
def test_invalid_dd_schema(schema: Union[str, Tuple['cirq.Gate', ...]], error_msg_regex):
    a = cirq.NamedQubit('a')
    input_circuit = cirq.Circuit(cirq.H(a))
    with pytest.raises(ValueError, match=error_msg_regex):
        add_dynamical_decoupling(input_circuit, schema=schema, single_qubit_gate_moments_only=False)


def test_single_qubit_gate_moments_only_no_updates_succeeds():
    qubits = cirq.LineQubit.range(9)
    input_circuit = cirq.Circuit(
        cirq.Moment([cirq.H(qubits[i]) for i in [3, 4, 5]]),
        cirq.Moment(cirq.CZ(*qubits[4:6])),
        cirq.Moment(cirq.CZ(*qubits[3:5])),
        cirq.Moment([cirq.H(qubits[i]) for i in [2, 3, 5, 6]]),
        cirq.Moment(cirq.CZ(*qubits[2:4]), cirq.CNOT(*qubits[5:7])),
        cirq.Moment([cirq.H(qubits[i]) for i in [1, 2, 6, 7]]),
        cirq.Moment(cirq.CZ(*qubits[1:3]), cirq.CNOT(*qubits[6:8])),
        cirq.Moment([cirq.H(qubits[i]) for i in [0, 1, 7, 8]]),
        cirq.Moment(cirq.CZ(*qubits[0:2]), cirq.CNOT(*qubits[7:])),
    )
    add_dynamical_decoupling(input_circuit, schema='X_XINV', single_qubit_gate_moments_only=True)


def test_scattered_circuit():
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
        cirq.Moment([cirq.H(qubits[i]) for i in [3, 4, 5]]),
        cirq.Moment(cirq.CZ(*qubits[4:6])),
        cirq.Moment(cirq.CZ(*qubits[3:5])),
        cirq.Moment([cirq.H(qubits[i]) for i in [2, 3, 5, 6]]),
        cirq.Moment(cirq.CZ(*qubits[2:4]), cirq.CZ(*qubits[5:7])),
        cirq.Moment([cirq.H(qubits[i]) for i in [1, 2, 6, 7]]),
        cirq.Moment(cirq.CZ(*qubits[1:3]), cirq.CZ(*qubits[6:8])),
        cirq.Moment([cirq.H(qubits[i]) for i in [0, 1, 7, 8]]),
        cirq.Moment(cirq.CZ(*qubits[0:2]), cirq.CZ(*qubits[7:])),
        cirq.Moment([cirq.H(q) for q in qubits]),
    )
    expected_circuit_single_qubit_gate_on = cirq.Circuit(
        cirq.Moment([cirq.H(qubits[i]) for i in [3, 4, 5]]),
        cirq.Moment(cirq.CZ(*qubits[4:6])),
        cirq.Moment(cirq.CZ(*qubits[3:5])),
        cirq.Moment([cirq.H(qubits[i]) for i in [2, 3, 5, 6]] + [cirq.X(qubits[4])]),
        cirq.Moment(cirq.CZ(*qubits[2:4]), cirq.CZ(*qubits[5:7])),
        cirq.Moment(
            [cirq.H(qubits[i]) for i in [1, 2, 6, 7]]
            + [cirq.X(qubits[i]) for i in [3, 5]]
            + [cirq.Y(qubits[4])]
        ),
        cirq.Moment(cirq.CZ(*qubits[1:3]), cirq.CZ(*qubits[6:8])),
        cirq.Moment(
            [cirq.H(qubits[i]) for i in [0, 1, 7, 8]]
            + [cirq.X(qubits[i]) for i in [2, 4, 6]]
            + [cirq.Y(qubits[i]) for i in [3, 5]]
        ),
        cirq.Moment(cirq.CZ(*qubits[0:2]), cirq.CZ(*qubits[7:])),
        cirq.Moment(
            [cirq.H(qubits[i]) for i in [0, 1, 7, 8]]
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
        cirq.Moment([cirq.H(qubits[i]) for i in [3, 4, 5]]),
        cirq.Moment(cirq.CZ(*qubits[4:6]), cirq.X(qubits[3])),
        cirq.Moment(cirq.CZ(*qubits[3:5]), cirq.X(qubits[5])),
        cirq.Moment([cirq.H(qubits[i]) for i in [2, 3, 5, 6]] + [cirq.X(qubits[i]) for i in [4]]),
        cirq.Moment(cirq.CZ(*qubits[2:4]), cirq.CZ(*qubits[5:7]), cirq.Y(qubits[4])),
        cirq.Moment(
            [cirq.H(qubits[i]) for i in [1, 2, 6, 7]]
            + [cirq.Y(qubits[i]) for i in [3, 5]]
            + [cirq.X(qubits[4])]
        ),
        cirq.Moment(
            [cirq.CZ(*qubits[1:3]), cirq.CZ(*qubits[6:8])]
            + [cirq.X(qubits[i]) for i in [3, 5]]
            + [cirq.Y(qubits[4])]
        ),
        cirq.Moment(
            [cirq.H(qubits[i]) for i in [0, 1, 7, 8]]
            + [cirq.X(qubits[i]) for i in [2, 4, 6]]
            + [cirq.Y(qubits[i]) for i in [3, 5]]
        ),
        cirq.Moment(
            [cirq.CZ(*qubits[0:2]), cirq.CZ(*qubits[7:])]
            + [cirq.X(qubits[i]) for i in [3, 5]]
            + [cirq.Y(qubits[i]) for i in [2, 4, 6]]
        ),
        cirq.Moment(
            [cirq.H(qubits[i]) for i in [0, 1, 4, 7, 8]]
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


def test_scattered_circuit2():
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
    Output:
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
    5: ───@───X───@───X───────
                  │
    6: ───────────@───@───Z───
                      │
    7: ───────────────@───@───
                          │
    8: ───────────────────@───
    """
    qubits = cirq.LineQubit.range(9)
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(cirq.CZ(*qubits[4:6])),
            cirq.Moment(cirq.CZ(*qubits[3:5])),
            cirq.Moment(cirq.CZ(*qubits[2:4]), cirq.CZ(*qubits[5:7])),
            cirq.Moment(cirq.CZ(*qubits[1:3]), cirq.CZ(*qubits[6:8])),
            cirq.Moment(cirq.CZ(*qubits[0:2]), cirq.CZ(*qubits[7:])),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.CZ(*qubits[4:6])),
            cirq.Moment(cirq.CZ(*qubits[3:5]), cirq.X(qubits[5])),
            cirq.Moment(cirq.CZ(*qubits[2:4]), cirq.CZ(*qubits[5:7])),
            cirq.Moment(cirq.CZ(*qubits[1:3]), cirq.CZ(*qubits[6:8]), cirq.X(qubits[5])),
            cirq.Moment(cirq.CZ(*qubits[0:2]), cirq.CZ(*qubits[7:]), cirq.Z(qubits[6])),
        ),
        schema="XX_PAIR",
        single_qubit_gate_moments_only=False,
    )


def test_pull_through_chain():
    """Test case diagrams.
    Input:
    0: ───X───────×───────────X───
                  │
    1: ───────Y───×───×───────X───
                      │
    2: ───────────────×───×───X───
                          │
    3: ───────────────────×───X───
    Output:
    0: ───X───X───×───X───X───X───
                  │
    1: ───────Y───×───×───X───I───
                      │
    2: ───────────────×───×───X───
                          │
    3: ───────────────────×───I───
    """
    qubits = cirq.LineQubit.range(4)
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(cirq.X(qubits[0])),
            cirq.Moment(cirq.Y(qubits[1])),
            cirq.Moment(cirq.SWAP(*qubits[0:2])),
            cirq.Moment(cirq.SWAP(*qubits[1:3])),
            cirq.Moment(cirq.SWAP(*qubits[2:4])),
            cirq.Moment([cirq.X(qubits[i]) for i in range(4)]),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.X(qubits[0])),
            cirq.Moment(cirq.Y(qubits[1]), cirq.X(qubits[0])),
            cirq.Moment(cirq.SWAP(*qubits[0:2])),
            cirq.Moment([cirq.SWAP(*qubits[1:3])] + [cirq.X(qubits[0])]),
            cirq.Moment([cirq.SWAP(*qubits[2:4])] + [cirq.X(qubits[0]), cirq.X(qubits[1])]),
            cirq.Moment(cirq.X(qubits[0]), cirq.I(qubits[1]), cirq.X(qubits[2]), cirq.I(qubits[3])),
        ),
        schema='XX_PAIR',
        single_qubit_gate_moments_only=False,
    )


def test_multiple_clifford_pieces_case1():
    """Test case diagrams.
    Input:
    a: ───H───────H───────@───────────H───────H───
                          │
    b: ───H───H───H───H───@^0.5───H───H───H───H───
    Output:
    a: ───H───X───H───PhXZ(a=0.5,x=0,z=-1)───@───────X───H───X───PhXZ(a=0.5,x=0.5,z=-1)───
                                             │
    b: ───H───H───H───H──────────────────────@^0.5───H───H───H───H────────────────────────
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b)),
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b)),
            cirq.Moment(cirq.CZPowGate(exponent=0.5).on(a, b)),
            cirq.Moment(cirq.H(b)),
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b)),
            cirq.Moment(cirq.H(a), cirq.H(b)),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b), cirq.X(a)),
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(
                cirq.H(b),
                cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0, z_exponent=-1).on(a),
            ),
            cirq.Moment(cirq.CZPowGate(exponent=0.5).on(a, b)),
            cirq.Moment(cirq.H(b), cirq.X(a)),
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b), cirq.X(a)),
            cirq.Moment(
                cirq.H(b),
                cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=-1).on(a),
            ),
        ),
        schema="XX_PAIR",
    )


def test_multiple_clifford_pieces_case2():
    """Test case diagrams.
    Input:
    a: ───@───PhXZ(a=0.3,x=0.2,z=0)───PhXZ(a=0.3,x=0.2,z=0)───PhXZ(a=0.3,x=0.2,z=0)───@───
          │                                                                           │
    b: ───@───────────────────────────────────────────────────────────────────────────@───
    Output:
    a: ───@───PhXZ(a=0.3,x=0.2,z=0)───PhXZ(a=0.3,x=0.2,z=0)───PhXZ(a=0.3,x=0.2,z=0)───@───Z───
          │                                                                           │
    b: ───@───X───────────────────────X───────────────────────X───────────────────────@───X───
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    phased_xz_gate = cirq.PhasedXZGate(axis_phase_exponent=0.3, x_exponent=0.2, z_exponent=0)

    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(cirq.CZ(a, b)),
            cirq.Moment(phased_xz_gate.on(a)),
            cirq.Moment(phased_xz_gate.on(a)),
            cirq.Moment(phased_xz_gate.on(a)),
            cirq.Moment(cirq.CZ(a, b)),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.CZ(a, b)),
            cirq.Moment(phased_xz_gate.on(a), cirq.X(b)),
            cirq.Moment(phased_xz_gate.on(a), cirq.X(b)),
            cirq.Moment(phased_xz_gate.on(a), cirq.X(b)),
            cirq.Moment(cirq.CZ(a, b)),
            cirq.Moment(cirq.Z(a), cirq.X(b)),
        ),
        schema='XX_PAIR',
        single_qubit_gate_moments_only=False,
    )


def test_insert_new_moment():
    """Test case diagrams.
    Input:
    a: ───H───────H───@───@───────
                      │   │
    b: ───H───H───H───X───@^0.5───

    c: ───H───────────────H───────
    Output:
    a: ───H───X───H───@───Z───@────────────────────────
                      │       │
    b: ───H───H───H───X───────@^0.5────────────────────

    c: ───H───X───X───────X───PhXZ(a=-0.5,x=0.5,z=0)───
    """
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a), cirq.H(b), cirq.H(c)),
            cirq.Moment(cirq.H(b)),
            cirq.Moment(cirq.H(b), cirq.H(a)),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(cirq.CZPowGate(exponent=0.5).on(a, b), cirq.H(c)),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a), cirq.H(b), cirq.H(c)),
            cirq.Moment(cirq.H(b), cirq.X(a), cirq.X(c)),
            cirq.Moment(cirq.H(a), cirq.H(b), cirq.X(c)),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(cirq.Z(a), cirq.X(c)),
            cirq.Moment(
                cirq.CZPowGate(exponent=0.5).on(a, b),
                cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=0).on(c),
            ),
        ),
        schema="XX_PAIR",
    )


def test_with_non_clifford_measurements():
    """Test case diagrams.
    Input:
    0: ───────────H───@───H───M───
                      │
    1: ───H───@───────@───────M───
              │
    2: ───H───@───H───@───────M───
                      │
    3: ───────────H───@───H───M───
    Output:
    0: ───────────H───@───PhXZ(a=0.5,x=0.5,z=0)───M───
                      │
    1: ───H───@───X───@───X───────────────────────M───
              │
    2: ───H───@───H───@───I───────────────────────M───
                      │
    3: ───────────H───@───H───────────────────────M───
    """
    qubits = cirq.LineQubit.range(4)
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment([cirq.H(qubits[i]) for i in [1, 2]]),
            cirq.Moment(cirq.CZ(*qubits[1:3])),
            cirq.Moment([cirq.H(qubits[i]) for i in [0, 2, 3]]),
            cirq.Moment(cirq.CZ(*qubits[0:2]), cirq.CZ(*qubits[2:])),
            cirq.Moment([cirq.H(qubits[i]) for i in [0, 3]]),
            cirq.Moment([cirq.M(qubits[i]) for i in [0, 1, 2, 3]]),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment([cirq.H(qubits[i]) for i in [1, 2]]),
            cirq.Moment(cirq.CZ(*qubits[1:3])),
            cirq.Moment([cirq.H(qubits[i]) for i in [0, 2, 3]] + [cirq.X(qubits[1])]),
            cirq.Moment(cirq.CZ(*qubits[0:2]), cirq.CZ(*qubits[2:])),
            cirq.Moment(
                cirq.H(qubits[3]),
                cirq.I(qubits[2]),
                cirq.X(qubits[1]),
                cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0).on(
                    qubits[0]
                ),
            ),
            cirq.Moment([cirq.M(qubits[i]) for i in [0, 1, 2, 3]]),
        ),
        schema="XX_PAIR",
        single_qubit_gate_moments_only=True,
    )


def test_cross_clifford_pieces_filling_merge():
    # pylint: disable=line-too-long
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
    Output:

    0: ─────────────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)─────H────────────────────────
                                                                  │                                                   │
    1: ─────────────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)─────H────────────────────────

    2: ───PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───X───────────────────────────PhXZ(a=0.5,x=0.5,z=-1)───
                                    │                             │                                                   │
    3: ─────────────────────────────┼───PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────X─────────────────────────@───Y───────────────────────────PhXZ(a=0.5,x=0.5,z=0)────
                                    │
    4: ─────────────────────────────┼─────────────────────────────@─────────────────────────X─────────────────────────────Y───────────────────────────PhXZ(a=0.5,x=0.5,z=0)────
                                    │                             │
    5: ───PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=-0.8,x=0.2,z=-0.9)───H────────────────────────
                                                                                                                      │
    6: ───────────────────────────────────────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───X─────────────────────────@───PhXZ(a=0.8,x=0.8,z=0.5)─────H────────────────────────
    """
    # pylint: enable
    qubits = cirq.LineQubit.range(7)
    phased_xz_gate = cirq.PhasedXZGate(axis_phase_exponent=0.2, x_exponent=0.2, z_exponent=0.1)
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment([phased_xz_gate.on(qubits[i]) for i in [2, 5]]),
            cirq.Moment(cirq.CZ(qubits[2], qubits[5])),
            cirq.Moment([phased_xz_gate.on(qubits[i]) for i in [0, 1, 2, 3, 5]]),
            cirq.Moment(
                [cirq.CZ(qubits[i0], qubits[i1]) for i0, i1 in [(0, 1), (2, 3), (4, 5)]]
                + [phased_xz_gate.on(qubits[6])]
            ),
            cirq.Moment([phased_xz_gate.on(qubits[i]) for i in [0, 1, 2, 5]]),
            cirq.Moment([cirq.CZ(qubits[i0], qubits[i1]) for i0, i1 in [(0, 1), (2, 3), (5, 6)]]),
            cirq.Moment([phased_xz_gate.on(qubits[i]) for i in [0, 1, 5, 6]]),
            cirq.Moment([cirq.H.on(q) for q in qubits]),
        ),
        expected_circuit="""
0: ─────────────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)─────H────────────────────────
                                                              │                                                   │
1: ─────────────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)─────H────────────────────────

2: ───PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───X───────────────────────────PhXZ(a=0.5,x=0.5,z=-1)───
                                │                             │                                                   │
3: ─────────────────────────────┼───PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────X─────────────────────────@───Y───────────────────────────PhXZ(a=0.5,x=0.5,z=0)────
                                │
4: ─────────────────────────────┼─────────────────────────────@─────────────────────────X─────────────────────────────Y───────────────────────────PhXZ(a=0.5,x=0.5,z=0)────
                                │                             │
5: ───PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=0.2,x=0.2,z=0.1)───@─────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───@───PhXZ(a=-0.8,x=0.2,z=-0.9)───H────────────────────────
                                                                                                                  │
6: ───────────────────────────────────────────────────────────PhXZ(a=0.2,x=0.2,z=0.1)───X─────────────────────────@───PhXZ(a=0.8,x=0.8,z=0.5)─────H────────────────────────
""",
    )


def test_pull_through_phxz_gate_case1():
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
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b)),
            cirq.Moment(phxz(a), cirq.H(b)),
            cirq.Moment(cirq.H(b)),
            cirq.Moment(cirq.CNOT(a, b)),
        ),
        expected_circuit="""
a: ───H───X───PhXZ(a=0.25,x=-1,z=0)───X───@───Z───
                                          │
b: ───H───H───H───────────────────────H───X───────
""",
        schema="XX_PAIR",
    )


def test_pull_through_phxz_gate_case2():
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
            cirq.Moment(cirq.H(a), cirq.H(b)),
            cirq.Moment(cirq.H(b)),
            cirq.Moment(phxz(a), cirq.H(b)),
            cirq.Moment(cirq.H(b)),
            cirq.Moment(cirq.CNOT(a, b)),
        ),
        expected_circuit="""
a: ───H───X───PhXZ(a=0.1,x=0,z=0.4)───X───@───X───
                                          │
b: ───H───H───H───────────────────────H───X───X───
""",
        schema="XX_PAIR",
    )
