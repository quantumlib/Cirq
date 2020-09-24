# Copyright 2020 The Cirq Developers
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
import pytest

import cirq


def test_stratified_circuit_classifier_types():
    a, b, c, d = cirq.LineQubit.range(4)

    circuit = cirq.Circuit(
        cirq.Moment([
            cirq.X(a),
            cirq.Y(b),
            cirq.X(c)**0.5,
            cirq.X(d),
        ]),)

    gate_result = cirq.stratified_circuit(circuit, categories=[
        cirq.X,
    ])
    assert gate_result == cirq.Circuit(
        cirq.Moment([
            cirq.X(a),
            cirq.X(d),
        ]), cirq.Moment([
            cirq.Y(b),
            cirq.X(c)**0.5,
        ]))

    gate_type_result = cirq.stratified_circuit(circuit,
                                               categories=[
                                                   cirq.XPowGate,
                                               ])
    assert gate_type_result == cirq.Circuit(
        cirq.Moment([
            cirq.X(a),
            cirq.X(c)**0.5,
            cirq.X(d),
        ]), cirq.Moment([
            cirq.Y(b),
        ]))

    operation_result = cirq.stratified_circuit(circuit, categories=[
        cirq.X(a),
    ])
    assert operation_result == cirq.Circuit(
        cirq.Moment([
            cirq.X(a),
        ]), cirq.Moment([
            cirq.Y(b),
            cirq.X(c)**0.5,
            cirq.X(d),
        ]))

    operation_type_result = cirq.stratified_circuit(circuit,
                                                    categories=[
                                                        cirq.GateOperation,
                                                    ])
    assert operation_type_result == cirq.Circuit(
        cirq.Moment([
            cirq.X(a),
            cirq.Y(b),
            cirq.X(c)**0.5,
            cirq.X(d),
        ]))

    predicate_result = cirq.stratified_circuit(circuit,
                                               categories=[
                                                   lambda op: op.qubits == (b,),
                                               ])
    assert predicate_result == cirq.Circuit(
        cirq.Moment([
            cirq.Y(b),
        ]), cirq.Moment([
            cirq.X(a),
            cirq.X(d),
            cirq.X(c)**0.5,
        ]))

    with pytest.raises(TypeError, match='Unrecognized'):
        _ = cirq.stratified_circuit(circuit, categories=['unknown'])


def test_overlapping_categories():
    a, b, c, d = cirq.LineQubit.range(4)

    result = cirq.stratified_circuit(cirq.Circuit(
        cirq.Moment([
            cirq.X(a),
            cirq.Y(b),
            cirq.Z(c),
        ]),
        cirq.Moment([
            cirq.CNOT(a, b),
        ]),
        cirq.Moment([
            cirq.CNOT(c, d),
        ]),
        cirq.Moment([
            cirq.X(a),
            cirq.Y(b),
            cirq.Z(c),
        ]),
    ),
                                     categories=[
                                         lambda op: len(op.qubits) == 1 and
                                         not isinstance(op.gate, cirq.XPowGate),
                                         lambda op: len(op.qubits) == 1 and
                                         not isinstance(op.gate, cirq.ZPowGate),
                                     ])

    assert result == cirq.Circuit(
        cirq.Moment([
            cirq.Y(b),
            cirq.Z(c),
        ]),
        cirq.Moment([
            cirq.X(a),
        ]),
        cirq.Moment([
            cirq.CNOT(a, b),
            cirq.CNOT(c, d),
        ]),
        cirq.Moment([
            cirq.Y(b),
            cirq.Z(c),
        ]),
        cirq.Moment([
            cirq.X(a),
        ]),
    )


def test_empty():
    a = cirq.LineQubit(0)
    assert cirq.stratified_circuit(cirq.Circuit(),
                                   categories=[]) == cirq.Circuit()
    assert cirq.stratified_circuit(cirq.Circuit(),
                                   categories=[cirq.X]) == cirq.Circuit()
    assert cirq.stratified_circuit(cirq.Circuit(cirq.X(a)),
                                   categories=[]) == cirq.Circuit(cirq.X(a))


def test_greedy_merging():
    """Tests a tricky situation where the algorithm of "Merge single-qubit
    gates, greedily align single-qubit then 2-qubit operations" doesn't work.
    Our algorithm succeeds because we also run it in reverse order. """
    q1, q2, q3, q4 = cirq.LineQubit.range(4)
    input_circuit = cirq.Circuit(
        cirq.Moment([cirq.X(q1)]),
        cirq.Moment([cirq.SWAP(q1, q2), cirq.SWAP(q3, q4)]),
        cirq.Moment([cirq.X(q3)]), cirq.Moment([cirq.SWAP(q3, q4)]))
    expected = cirq.Circuit(cirq.Moment([cirq.SWAP(q3, q4)]),
                            cirq.Moment([cirq.X(q1), cirq.X(q3)]),
                            cirq.Moment([cirq.SWAP(q1, q2),
                                         cirq.SWAP(q3, q4)]))
    assert cirq.stratified_circuit(input_circuit,
                                   categories=[cirq.X]) == expected


def test_greedy_merging_reverse():
    """Same as the above test, except that the aligning is done in reverse."""
    q1, q2, q3, q4 = cirq.LineQubit.range(4)
    input_circuit = cirq.Circuit(
        cirq.Moment([cirq.SWAP(q1, q2), cirq.SWAP(q3, q4)]),
        cirq.Moment([cirq.X(q4)]), cirq.Moment([cirq.SWAP(q3, q4)]),
        cirq.Moment([cirq.X(q1)]))
    expected = cirq.Circuit(cirq.Moment([cirq.SWAP(q1, q2),
                                         cirq.SWAP(q3, q4)]),
                            cirq.Moment([cirq.X(q1), cirq.X(q4)]),
                            cirq.Moment([cirq.SWAP(q3, q4)]))
    assert cirq.stratified_circuit(input_circuit,
                                   categories=[cirq.X]) == expected


def test_complex_circuit():
    """Tests that a complex circuit is correctly optimized."""
    q1, q2, q3, q4, q5 = cirq.LineQubit.range(5)
    input_circuit = cirq.Circuit(
        cirq.Moment([cirq.X(q1), cirq.ISWAP(q2, q3),
                     cirq.Z(q5)]), cirq.Moment([cirq.X(q1),
                                                cirq.ISWAP(q4, q5)]),
        cirq.Moment([cirq.ISWAP(q1, q2), cirq.X(q4)]))
    expected = cirq.Circuit(
        cirq.Moment([cirq.X(q1)]), cirq.Moment([cirq.Z(q5)]),
        cirq.Moment([cirq.ISWAP(q2, q3), cirq.ISWAP(q4, q5)]),
        cirq.Moment([cirq.X(q1), cirq.X(q4)]), cirq.Moment([cirq.ISWAP(q1,
                                                                       q2)]))
    assert cirq.stratified_circuit(input_circuit,
                                   categories=[cirq.X, cirq.Z]) == expected


def test_heterogeneous_circuit():
    """Tests that a circuit that is very heterogeneous is correctly optimized"""
    q1, q2, q3, q4, q5, q6 = cirq.LineQubit.range(6)
    input_circuit = cirq.Circuit(
        cirq.Moment(
            [cirq.X(q1),
             cirq.X(q2),
             cirq.ISWAP(q3, q4),
             cirq.ISWAP(q5, q6)]),
        cirq.Moment(
            [cirq.ISWAP(q1, q2),
             cirq.ISWAP(q3, q4),
             cirq.X(q5),
             cirq.X(q6)]),
        cirq.Moment([
            cirq.X(q1),
            cirq.Z(q2),
            cirq.X(q3),
            cirq.Z(q4),
            cirq.X(q5),
            cirq.Z(q6)
        ]))
    expected = cirq.Circuit(
        cirq.Moment([cirq.ISWAP(q3, q4), cirq.ISWAP(q5, q6)]),
        cirq.Moment([cirq.X(q1), cirq.X(q2),
                     cirq.X(q5), cirq.X(q6)]),
        cirq.Moment([
            cirq.ISWAP(q1, q2),
            cirq.ISWAP(q3, q4),
        ]), cirq.Moment([cirq.Z(q2), cirq.Z(q4),
                         cirq.Z(q6)]),
        cirq.Moment([
            cirq.X(q1),
            cirq.X(q3),
            cirq.X(q5),
        ]))

    assert cirq.stratified_circuit(input_circuit,
                                   categories=[cirq.X, cirq.Z]) == expected


def test_surface_code_cycle_stratifies_without_growing():
    g = cirq.GridQubit
    circuit = cirq.Circuit(
        cirq.H(g(9, 11)),
        cirq.H(g(11, 12)),
        cirq.H(g(12, 9)),
        cirq.H(g(9, 8)),
        cirq.H(g(8, 11)),
        cirq.H(g(11, 9)),
        cirq.H(g(10, 9)),
        cirq.H(g(10, 8)),
        cirq.H(g(11, 10)),
        cirq.H(g(12, 10)),
        cirq.H(g(9, 9)),
        cirq.H(g(9, 10)),
        cirq.H(g(10, 11)),
        cirq.CZ(g(10, 9), g(9, 9)),
        cirq.CZ(g(10, 11), g(9, 11)),
        cirq.CZ(g(9, 10), g(8, 10)),
        cirq.CZ(g(11, 10), g(10, 10)),
        cirq.CZ(g(12, 9), g(11, 9)),
        cirq.CZ(g(11, 12), g(10, 12)),
        cirq.H(g(9, 11)),
        cirq.H(g(9, 9)),
        cirq.H(g(10, 10)),
        cirq.H(g(11, 9)),
        cirq.H(g(10, 12)),
        cirq.H(g(8, 10)),
        cirq.CZ(g(11, 10), g(11, 11)),
        cirq.CZ(g(10, 9), g(10, 8)),
        cirq.CZ(g(12, 9), g(12, 10)),
        cirq.CZ(g(10, 11), g(10, 10)),
        cirq.CZ(g(9, 8), g(9, 9)),
        cirq.CZ(g(9, 10), g(9, 11)),
        cirq.CZ(g(8, 11), g(8, 10)),
        cirq.CZ(g(11, 10), g(11, 9)),
        cirq.CZ(g(11, 12), g(11, 11)),
        cirq.H(g(10, 8)),
        cirq.H(g(12, 10)),
        cirq.H(g(12, 9)),
        cirq.CZ(g(9, 10), g(9, 9)),
        cirq.CZ(g(10, 9), g(10, 10)),
        cirq.CZ(g(10, 11), g(10, 12)),
        cirq.H(g(11, 11)),
        cirq.H(g(9, 11)),
        cirq.H(g(11, 9)),
        cirq.CZ(g(9, 8), g(10, 8)),
        cirq.CZ(g(11, 10), g(12, 10)),
        cirq.H(g(11, 12)),
        cirq.H(g(8, 10)),
        cirq.H(g(10, 10)),
        cirq.CZ(g(8, 11), g(9, 11)),
        cirq.CZ(g(10, 9), g(11, 9)),
        cirq.CZ(g(10, 11), g(11, 11)),
        cirq.H(g(9, 8)),
        cirq.H(g(10, 12)),
        cirq.H(g(11, 10)),
        cirq.CZ(g(9, 10), g(10, 10)),
        cirq.H(g(11, 11)),
        cirq.H(g(9, 11)),
        cirq.H(g(8, 11)),
        cirq.H(g(11, 9)),
        cirq.H(g(10, 9)),
        cirq.H(g(10, 11)),
        cirq.H(g(9, 10)),
    )
    assert len(circuit) == 8
    stratified = cirq.stratified_circuit(circuit, categories=[cirq.H, cirq.CZ])
    # Ideally, this would not grow at all, but for now the algorithm has it
    # grow to a 9. Note that this optimizer uses a fairly simple algorithm
    # that is known not to be optimal - optimal stratification is a CSP
    # problem with high dimensionality that quickly becomes intractable. See
    # https://github.com/quantumlib/Cirq/pull/2772/ for some discussion on
    # this, as well as a more optimal but much more complex and slow solution.
    assert len(stratified) == 9
