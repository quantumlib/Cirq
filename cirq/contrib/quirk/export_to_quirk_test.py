# Copyright 2018 Google LLC
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
from cirq import ops
from cirq.contrib.quirk.export_to_quirk import circuit_to_quirk_url


def test_x_z_same_col():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit.from_ops(cirq.X(a), cirq.Z(b))
    assert circuit_to_quirk_url(circuit) == """
        http://algassert.com/quirk#circuit={"cols":[["X","Z"]]}
    """.strip()


def test_x_cnot_split_cols():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    circuit = cirq.Circuit.from_ops(cirq.CNOT(a, b), cirq.X(c))
    assert circuit_to_quirk_url(circuit) == """
        http://algassert.com/quirk#circuit={"cols":[["•","X"],[1,1,"X"]]}
    """.strip()


def test_cz_cnot_split_cols():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    circuit = cirq.Circuit.from_ops(cirq.CNOT(a, b), cirq.CZ(b, c))
    assert circuit_to_quirk_url(circuit) == """
        http://algassert.com/quirk#circuit={"cols":[["•","X"],[1,"•","Z"]]}
    """.strip()


def test_various_known_gate_types():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    circuit = cirq.Circuit.from_ops(
        cirq.X(a),
        cirq.X(a)**0.25,
        cirq.X(a)**-0.5,
        cirq.Z(a),
        cirq.Z(a)**0.5,
        cirq.Y(a),
        cirq.Y(a)**-0.25,
        cirq.H(a),
        cirq.MeasurementGate()(a),
        cirq.MeasurementGate()(a, b),
        cirq.SWAP(a, b),
        cirq.CNOT(a, b),
        cirq.CNOT(b, a),
        cirq.CZ(a, b),
    )
    assert circuit_to_quirk_url(circuit) == """
        http://algassert.com/quirk#circuit={"cols":[
            ["X"],
            ["X^¼"],
            ["X^-½"],
            ["Z"],
            ["Z^½"],
            ["Y"],
            ["Y^-¼"],
            ["H"],
            ["Measure"],
            ["Measure","Measure"],
            ["Swap","Swap"],
            ["•","X"],
            ["X","•"],
            ["•","Z"]]}
    """.replace('\n', '').replace(' ', '')


def test_unrecognized_single_qubit_gate_with_matrix():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit.from_ops(
        cirq.X(a)**0.2731,
    )
    assert circuit_to_quirk_url(circuit) == """
        http://algassert.com/quirk#circuit={"cols":[[{
            "id":"?",
            "matrix":"{
                {0.8269876673711053+0.37825793499568966i,
                            0.1730123326288947+-0.37825793499568966i},
                {0.1730123326288947+-0.37825793499568966i,
                            0.8269876673711053+0.37825793499568966i}
            }"}]]}
    """.replace('\n', '').replace(' ', '')


def test_unknown_gate():
    class UnknownGate(ops.Gate):
        pass
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit.from_ops(UnknownGate()(a))
    with pytest.raises(TypeError):
        _ = circuit_to_quirk_url(circuit)
    assert circuit_to_quirk_url(circuit,
                                prefer_unknown_gate_to_failure=True) == """
        http://algassert.com/quirk#circuit={"cols":[["UNKNOWN"]]}
    """.strip()
