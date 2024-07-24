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

from typing import Sequence, Union
from unittest import mock
import re
import cirq
from cirq import add_dynamical_decoupling
import pytest


def assert_dd(
    input_circuit: cirq.Circuit,
    expected_circuit: Union[str, cirq.Circuit],
    schema: Union[str, Sequence['cirq.Gate']],
    single_qubit_gate_moments_only: bool = True,
):
    updated_circuit = add_dynamical_decoupling(
        input_circuit, schema=schema, single_qubit_gate_moments_only=single_qubit_gate_moments_only
    )
    if isinstance(expected_circuit, cirq.Circuit):
        cirq.testing.assert_same_circuits(updated_circuit, expected_circuit)
    else:
        cirq.testing.assert_has_diagram(updated_circuit, expected_circuit)


def test_no_insert_due_to_no_consecutive_moments():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # No insertion as there is no room for a dd sequence.
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
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    input_circuit = cirq.Circuit(
        cirq.Moment(cirq.H(a)),
        cirq.Moment(cirq.CNOT(a, b)),
        cirq.Moment(cirq.CNOT(b, c)),
        cirq.Moment(cirq.CNOT(b, c)),
        cirq.Moment(cirq.measure_each(a, b, c)),
    )
    expected_circuit = cirq.Circuit(
        cirq.Moment(cirq.H(a)),
        cirq.Moment(cirq.CNOT(a, b)),
        cirq.Moment(cirq.CNOT(b, c), inserted_gates[0](a)),
        cirq.Moment(cirq.CNOT(b, c), inserted_gates[1](a)),
        cirq.Moment(cirq.measure_each(a, b, c)),
    )

    # Insert one dynamical decoupling sequence in idle moments.
    assert_dd(input_circuit, expected_circuit, schema=schema, single_qubit_gate_moments_only=False)


def test_insert_by_customized_dd_sequence():
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
            cirq.Moment(cirq.measure_each(a, b, c)),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a)),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(cirq.CNOT(b, c), cirq.X(a)),
            cirq.Moment(cirq.CNOT(b, c), cirq.X(a)),
            cirq.Moment(cirq.CNOT(b, c), cirq.Y(a)),
            cirq.Moment(cirq.CNOT(b, c), cirq.Y(a)),
            cirq.Moment(cirq.measure_each(a, b, c)),
        ),
        schema=[cirq.X, cirq.X, cirq.Y, cirq.Y],
        single_qubit_gate_moments_only=False,
    )


@pytest.mark.parametrize(
    'schema,error_msg_regex',
    [
        ('INVALID_SCHEMA', 'Invalid schema name.'),
        ([cirq.X], 'Invalid dynamical decoupling sequence. Expect more than one gates.'),
        (
            [cirq.X, cirq.H],
            'Invalid dynamical decoupling sequence. Expect sequence production equals identity'
            ' up to a global phase, got',
        ),
    ],
)
def test_invalid_dd_schema(schema: Union[str, Sequence['cirq.Gate']], error_msg_regex):
    a = cirq.NamedQubit('a')
    input_circuit = cirq.Circuit(cirq.H(a))
    with pytest.raises(ValueError, match=error_msg_regex):
        add_dynamical_decoupling(input_circuit, schema=schema, single_qubit_gate_moments_only=False)


def test_single_qubit_gate_moments_only_no_updates_succeeds():
    qubits = cirq.LineQubit.range(9)
    input_circuit = cirq.Circuit(
        cirq.Moment([cirq.H(qubits[i]) for i in [3, 4, 5]]),
        cirq.Moment(cirq.CNOT(*qubits[4:6])),
        cirq.Moment(cirq.CNOT(*qubits[3:5])),
        cirq.Moment([cirq.H(qubits[i]) for i in [2, 3, 5, 6]]),
        cirq.Moment(cirq.CNOT(*qubits[2:4]), cirq.CNOT(*qubits[5:7])),
        cirq.Moment([cirq.H(qubits[i]) for i in [1, 2, 6, 7]]),
        cirq.Moment(cirq.CNOT(*qubits[1:3]), cirq.CNOT(*qubits[6:8])),
        cirq.Moment([cirq.H(qubits[i]) for i in [0, 1, 7, 8]]),
        cirq.Moment(cirq.CNOT(*qubits[0:2]), cirq.CNOT(*qubits[7:])),
    )
    assert_dd(input_circuit, expected_circuit=input_circuit, schema="X_XINV")


def test_single_qubit_gate_moments_only_with_updates_succeeds():
    qubits = cirq.LineQubit.range(9)
    input_circuit = cirq.Circuit(
        cirq.Moment([cirq.H(qubits[i]) for i in [3, 4, 5]]),
        cirq.Moment(cirq.CNOT(*qubits[4:6])),
        cirq.Moment(cirq.CNOT(*qubits[3:5])),
        cirq.Moment([cirq.H(qubits[i]) for i in [2, 3, 5, 6]]),
        cirq.Moment(cirq.CNOT(*qubits[2:4]), cirq.CNOT(*qubits[5:7])),
        cirq.Moment([cirq.H(qubits[i]) for i in [1, 2, 6, 7]]),
        cirq.Moment(cirq.CNOT(*qubits[1:3]), cirq.CNOT(*qubits[6:8])),
        cirq.Moment([cirq.H(qubits[i]) for i in [0, 1, 7, 8]]),
        cirq.Moment(cirq.CNOT(*qubits[0:2]), cirq.CNOT(*qubits[7:])),
        cirq.Moment([cirq.H(q) for q in qubits]),
    )
    assert_dd(
        input_circuit,
        expected_circuit="""
0: ───────────────────────────────H───@───H───────────────────────
                                      │
1: ───────────────────────H───@───H───X───H───────────────────────
                              │
2: ───────────────H───@───H───X───X───────PhXZ(a=0.5,x=0.5,z=0)───
                      │
3: ───H───────@───H───X───X───────X───────H───────────────────────
              │
4: ───H───@───X───X───────X───────X───────PhXZ(a=0.5,x=0.5,z=0)───
          │
5: ───H───X───────H───@───X───────X───────H───────────────────────
                      │
6: ───────────────H───X───H───@───X───────PhXZ(a=0.5,x=0.5,z=0)───
                              │
7: ───────────────────────H───X───H───@───H───────────────────────
                                      │
8: ───────────────────────────────H───X───H───────────────────────
""",
        schema="X_XINV",
    )


def test_single_qubit_gate_moments_only_exceptions():
    qubits = cirq.LineQubit.range(9)
    input_circuit = cirq.Circuit(
        cirq.Moment([cirq.H(qubits[i]) for i in [3, 4, 5]]),
        cirq.Moment(cirq.CNOT(*qubits[4:6])),
        cirq.Moment(cirq.CNOT(*qubits[3:5])),
        cirq.Moment([cirq.H(qubits[i]) for i in [2, 3, 5, 6]]),
        cirq.Moment(cirq.CNOT(*qubits[2:4]), cirq.CNOT(*qubits[5:7])),
        cirq.Moment([cirq.H(qubits[i]) for i in [1, 2, 6, 7]]),
        cirq.Moment(cirq.CNOT(*qubits[1:3]), cirq.CNOT(*qubits[6:8])),
        cirq.Moment([cirq.H(qubits[i]) for i in [0, 1, 7, 8]]),
        cirq.Moment(cirq.CNOT(*qubits[0:2]), cirq.CNOT(*qubits[7:])),
        cirq.Moment([cirq.H(q) for q in qubits]),
    )
    with mock.patch(
        "cirq.transformers.analytical_decompositions"
        ".single_qubit_decompositions.single_qubit_matrix_to_phxz",
        return_value=None,
    ):
        with pytest.raises(
            ValueError, match=re.compile("Can't convert .* to PhasedXZ gate.", re.DOTALL)
        ):
            add_dynamical_decoupling(input_circuit)

    with mock.patch('cirq.AbstractCircuit.next_moment_operating_on', return_value=None):
        assert_dd(input_circuit=input_circuit, expected_circuit=input_circuit, schema="X_XINV")
