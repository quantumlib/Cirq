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

from typing import Callable, List

import pytest
import sympy

import cirq


def assert_optimizes(before: cirq.Circuit, expected: cirq.Circuit):
    actual = cirq.Circuit(before)
    opt = cirq.MergeInteractions()
    opt.optimize_circuit(actual)

    # Ignore differences that would be caught by follow-up optimizations.
    followup_optimizations: List[Callable[[cirq.Circuit], None]] = [
        cirq.merge_single_qubit_gates_into_phased_x_z,
        cirq.EjectPhasedPaulis().optimize_circuit,
        cirq.EjectZ().optimize_circuit,
        cirq.DropNegligible().optimize_circuit,
        cirq.DropEmptyMoments().optimize_circuit,
    ]
    for post in followup_optimizations:
        post(actual)
        post(expected)

    assert actual == expected, f'ACTUAL {actual} : EXPECTED {expected}'


def assert_optimization_not_broken(circuit):
    """Check that the unitary matrix for the input circuit is the same (up to
    global phase and rounding error) as the unitary matrix of the optimized
    circuit."""
    u_before = circuit.unitary()
    cirq.MergeInteractions().optimize_circuit(circuit)
    u_after = circuit.unitary()

    cirq.testing.assert_allclose_up_to_global_phase(u_before, u_after, atol=1e-8)


def test_clears_paired_cnot():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.CNOT(a, b)]),
                cirq.Moment([cirq.CNOT(a, b)]),
            ]
        ),
        expected=cirq.Circuit(),
    )


def test_ignores_czs_separated_by_parameterized():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.CZ(a, b)]),
                cirq.Moment([cirq.Z(a) ** sympy.Symbol('boo')]),
                cirq.Moment([cirq.CZ(a, b)]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment([cirq.CZ(a, b)]),
                cirq.Moment([cirq.Z(a) ** sympy.Symbol('boo')]),
                cirq.Moment([cirq.CZ(a, b)]),
            ]
        ),
    )


def test_ignores_czs_separated_by_outer_cz():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.CZ(q00, q01)]),
                cirq.Moment([cirq.CZ(q00, q10)]),
                cirq.Moment([cirq.CZ(q00, q01)]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment([cirq.CZ(q00, q01)]),
                cirq.Moment([cirq.CZ(q00, q10)]),
                cirq.Moment([cirq.CZ(q00, q01)]),
            ]
        ),
    )


def test_cnots_separated_by_single_gates_correct():
    a, b = cirq.LineQubit.range(2)
    assert_optimization_not_broken(
        cirq.Circuit(
            cirq.CNOT(a, b),
            cirq.H(b),
            cirq.CNOT(a, b),
        )
    )


def test_czs_separated_by_single_gates_correct():
    a, b = cirq.LineQubit.range(2)
    assert_optimization_not_broken(
        cirq.Circuit(
            cirq.CZ(a, b),
            cirq.X(b),
            cirq.X(b),
            cirq.X(b),
            cirq.CZ(a, b),
        )
    )


def test_inefficient_circuit_correct():
    t = 0.1
    v = 0.11
    a, b = cirq.LineQubit.range(2)
    assert_optimization_not_broken(
        cirq.Circuit(
            cirq.H(b),
            cirq.CNOT(a, b),
            cirq.H(b),
            cirq.CNOT(a, b),
            cirq.CNOT(b, a),
            cirq.H(a),
            cirq.CNOT(a, b),
            cirq.Z(a) ** t,
            cirq.Z(b) ** -t,
            cirq.CNOT(a, b),
            cirq.H(a),
            cirq.Z(b) ** v,
            cirq.CNOT(a, b),
            cirq.Z(a) ** -v,
            cirq.Z(b) ** -v,
        )
    )


def test_optimizes_single_iswap():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.ISWAP(a, b))
    assert_optimization_not_broken(c)
    cirq.MergeInteractions().optimize_circuit(c)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 2


def test_optimizes_tagged_partial_cz():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit((cirq.CZ ** 0.5)(a, b).with_tags('mytag'))
    assert_optimization_not_broken(c)
    cirq.MergeInteractions(allow_partial_czs=False).optimize_circuit(c)
    assert (
        len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 2
    ), 'It should take 2 CZ gates to decompose a CZ**0.5 gate'


def test_not_decompose_czs():
    circuit = cirq.Circuit(
        cirq.CZPowGate(exponent=1, global_shift=-0.5).on(*cirq.LineQubit.range(2))
    )
    circ_orig = circuit.copy()
    cirq.MergeInteractions(allow_partial_czs=False).optimize_circuit(circuit)
    assert circ_orig == circuit


@pytest.mark.parametrize(
    'circuit',
    (
        cirq.Circuit(
            cirq.CZPowGate(exponent=0.1)(*cirq.LineQubit.range(2)),
        ),
        cirq.Circuit(
            cirq.CZPowGate(exponent=0.2)(*cirq.LineQubit.range(2)),
            cirq.CZPowGate(exponent=0.3, global_shift=-0.5)(*cirq.LineQubit.range(2)),
        ),
    ),
)
def test_decompose_partial_czs(circuit):
    optimizer = cirq.MergeInteractions(allow_partial_czs=False)
    optimizer.optimize_circuit(circuit)

    cz_gates = [
        op.gate
        for op in circuit.all_operations()
        if isinstance(op, cirq.GateOperation) and isinstance(op.gate, cirq.CZPowGate)
    ]
    num_full_cz = sum(1 for cz in cz_gates if cz.exponent % 2 == 1)
    num_part_cz = sum(1 for cz in cz_gates if cz.exponent % 2 != 1)
    assert num_full_cz == 2
    assert num_part_cz == 0


def test_not_decompose_partial_czs():
    circuit = cirq.Circuit(
        cirq.CZPowGate(exponent=0.1, global_shift=-0.5)(*cirq.LineQubit.range(2)),
    )

    optimizer = cirq.MergeInteractions(allow_partial_czs=True)
    optimizer.optimize_circuit(circuit)

    cz_gates = [
        op.gate
        for op in circuit.all_operations()
        if isinstance(op, cirq.GateOperation) and isinstance(op.gate, cirq.CZPowGate)
    ]
    num_full_cz = sum(1 for cz in cz_gates if cz.exponent % 2 == 1)
    num_part_cz = sum(1 for cz in cz_gates if cz.exponent % 2 != 1)
    assert num_full_cz == 0
    assert num_part_cz == 1


def test_post_clean_up():
    class Marker(cirq.testing.TwoQubitGate):
        pass

    a, b = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(
        cirq.CZ(a, b),
        cirq.CZ(a, b),
        cirq.CZ(a, b),
        cirq.CZ(a, b),
        cirq.CZ(a, b),
    )
    circuit = cirq.Circuit(c_orig)

    def clean_up(operations):
        yield Marker()(a, b)
        yield operations
        yield Marker()(a, b)

    optimizer = cirq.MergeInteractions(allow_partial_czs=False, post_clean_up=clean_up)
    optimizer.optimize_circuit(circuit)
    cirq.DropEmptyMoments().optimize_circuit(circuit)

    assert isinstance(circuit[0].operations[0].gate, Marker)
    assert isinstance(circuit[-1].operations[0].gate, Marker)

    u_before = c_orig.unitary()
    u_after = circuit[1:-1].unitary()
    cirq.testing.assert_allclose_up_to_global_phase(u_before, u_after, atol=1e-8)
