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
