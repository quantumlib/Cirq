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
    assert circuit_to_quirk_url(circuit, escape_url=False) == """
        http://algassert.com/quirk#circuit={"cols":[["X","Z"]]}
    """.strip()
    assert circuit_to_quirk_url(circuit) == (
        'http://algassert.com/quirk#circuit='
        '%7B%22cols%22%3A%5B%5B%22X%22%2C%22Z%22%5D%5D%7D')


def test_x_cnot_split_cols():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    circuit = cirq.Circuit.from_ops(cirq.CNOT(a, b), cirq.X(c))
    assert circuit_to_quirk_url(circuit, escape_url=False) == """
        http://algassert.com/quirk#circuit={"cols":[["•","X"],[1,1,"X"]]}
    """.strip()


def test_cz_cnot_split_cols():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    circuit = cirq.Circuit.from_ops(cirq.CNOT(a, b), cirq.CZ(b, c))
    assert circuit_to_quirk_url(circuit, escape_url=False) == """
        http://algassert.com/quirk#circuit={"cols":[["•","X"],[1,"•","Z"]]}
    """.strip()


def test_various_known_gate_types():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit.from_ops(
        cirq.google.ExpWGate(axis_half_turns=0).on(a),
        cirq.google.ExpWGate(axis_half_turns=0.5).on(a),
        cirq.X(a),
        cirq.X(a)**0.25,
        cirq.X(a)**-0.5,
        cirq.Z(a),
        cirq.Z(a)**0.5,
        cirq.Y(a),
        cirq.Y(a)**-0.25,
        cirq.RotYGate(half_turns=cirq.Symbol('t')).on(a),
        cirq.H(a),
        cirq.MeasurementGate()(a),
        cirq.MeasurementGate()(a, b),
        cirq.SWAP(a, b),
        cirq.CNOT(a, b),
        cirq.CNOT(b, a),
        cirq.CZ(a, b),
    )
    assert circuit_to_quirk_url(circuit, escape_url=False) == """
        http://algassert.com/quirk#circuit={"cols":[
            ["X"],
            ["Y"],
            ["X"],
            ["X^¼"],
            ["X^-½"],
            ["Z"],
            ["Z^½"],
            ["Y"],
            ["Y^-¼"],
            ["Y^t"],
            ["H"],
            ["Measure"],
            ["Measure","Measure"],
            ["Swap","Swap"],
            ["•","X"],
            ["X","•"],
            ["•","Z"]]}
    """.replace('\n', '').replace(' ', '')


def test_various_unknown_gate_types():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit.from_ops(
        cirq.X(a)**(1/5),
        cirq.Y(a)**(1/5),
        cirq.Z(a)**(1/5),
        cirq.CZ(a, b)**(1/5),
        cirq.google.ExpWGate(axis_half_turns=0.25)(a),
        cirq.google.ExpWGate(half_turns=1, axis_half_turns=cirq.Symbol('r'))(a)
    )
    assert circuit_to_quirk_url(circuit,
                                escape_url=False,
                                prefer_unknown_gate_to_failure=True) == """
        http://algassert.com/quirk#circuit={"cols":[
            [{"id":"?",
              "matrix":"{{0.9045084971874737+0.29389262614623657i,
                          0.09549150281252627+-0.29389262614623657i},
                         {0.09549150281252627+-0.29389262614623657i,
                          0.9045084971874737+0.29389262614623657i}}"}],
            [{"id":"?",
              "matrix":"{{0.9045084971874737+0.29389262614623657i,
                          0.29389262614623657+0.09549150281252627i},
                         {-0.29389262614623657+-0.09549150281252627i,
                          0.9045084971874737+0.29389262614623657i}}"}],
            [{"id":"?",
              "matrix":"{{1.0+0.0i,0.0+0.0i},
                         {0.0+0.0i,0.8090169943749475+0.5877852522924731i}}"}],
            ["UNKNOWN", "UNKNOWN"],
            [{"id":"?",
              "matrix":"{{0.0+6.123233995736766e-17i,
                          0.7071067811865476+0.7071067811865475i},
                         {0.7071067811865476+-0.7071067811865475i,
                          0.0+6.123233995736766e-17i}}"}],
            ["UNKNOWN"]
        ]}
    """.replace('\n', '').replace(' ', '')


def test_unrecognized_single_qubit_gate_with_matrix():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit.from_ops(
        cirq.X(a)**0.2731,
    )
    assert circuit_to_quirk_url(circuit, escape_url=False) == """
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
    with pytest.raises(TypeError):
        _ = circuit_to_quirk_url(circuit, escape_url=False)
    assert circuit_to_quirk_url(circuit,
                                prefer_unknown_gate_to_failure=True,
                                escape_url=False) == """
        http://algassert.com/quirk#circuit={"cols":[["UNKNOWN"]]}
    """.strip()
