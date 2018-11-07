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
        cirq.X(a),
        cirq.X(a)**0.25,
        cirq.X(a)**-0.5,
        cirq.Z(a),
        cirq.Z(a)**0.5,
        cirq.Y(a),
        cirq.Y(a)**-0.25,
        cirq.Y(a)**cirq.Symbol('t'),
        cirq.H(a),
        cirq.measure(a),
        cirq.measure(a, b, key='not-relevant'),
        cirq.SWAP(a, b),
        cirq.CNOT(a, b),
        cirq.CNOT(b, a),
        cirq.CZ(a, b),
    )
    assert circuit_to_quirk_url(circuit, escape_url=False) == """
        http://algassert.com/quirk#circuit={"cols":[
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


class MysteryOperation(cirq.Operation):
    def __init__(self, *qubits):
        self._qubits = qubits

    @property
    def qubits(self):
        return self._qubits

    def with_qubits(self, *new_qubits):
        return MysteryOperation(*new_qubits)


def test_various_unknown_gate_types():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit.from_ops(
        MysteryOperation(b),
        cirq.SWAP(a, b)**0.5,
        cirq.H(a)**0.5,
        cirq.SingleQubitCliffordGate.X_sqrt.merged_with(
            cirq.SingleQubitCliffordGate.Z_sqrt).on(a),
        cirq.X(a)**(1/5),
        cirq.Y(a)**(1/5),
        cirq.Z(a)**(1/5),
        cirq.CZ(a, b)**(1/5),
        cirq.PhasedXPowGate(phase_exponent=0.25)(a),
        cirq.PhasedXPowGate(exponent=1, phase_exponent=cirq.Symbol('r'))(a),
        cirq.PhasedXPowGate(exponent=0.001, phase_exponent=0.1)(a)
    )
    actual = circuit_to_quirk_url(
        circuit,
        escape_url=False,
        prefer_unknown_gate_to_failure=True)
    assert actual == """
        http://algassert.com/quirk#circuit={"cols":[
            [1,"UNKNOWN"],
            ["UNKNOWN", "UNKNOWN"],
            [{"id":"?","matrix":"{{0.853553+0.146447i,0.353553-0.353553i},
                                  {0.353553-0.353553i,0.146447+0.853553i}}"}],
            [{"id":"?","matrix":"{{0.5+0.5i,0.5+0.5i},{0.5-0.5i,-0.5+0.5i}}"}],
            [{"id":"?",
              "matrix":"{{0.904508+0.293893i, 0.095492-0.293893i},
                         {0.095492-0.293893i, 0.904508+0.293893i}}"}],
            [{"id":"?",
              "matrix":"{{0.904508+0.293893i, 0.293893+0.095492i},
                         {-0.293893-0.095492i, 0.904508+0.293893i}}"}],
            [{"id":"?",
              "matrix":"{{1, 0},
                         {0, 0.809017+0.587785i}}"}],
            ["UNKNOWN", "UNKNOWN"],
            [{"id":"?",
              "matrix":"{{0, 0.707107+0.707107i},
                         {0.707107-0.707107i, 0}}"}],
            ["UNKNOWN"],
            [{"id":"?",
              "matrix":"{{0.999998+0.001571i,0.000488-0.001493i},
                         {-0.000483-0.001495i,0.999998+0.001571i}}"}]
        ]}
    """.replace('\n', '').replace(' ', ''), actual.replace('],[', '],\n[')


def test_unrecognized_single_qubit_gate_with_matrix():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit.from_ops(
        cirq.X(a)**0.2731,
    )
    assert circuit_to_quirk_url(circuit, escape_url=False) == """
        http://algassert.com/quirk#circuit={"cols":[[{
            "id":"?",
            "matrix":"{
                {0.826988+0.378258i, 0.173012-0.378258i},
                {0.173012-0.378258i, 0.826988+0.378258i}
            }"}]]}
    """.replace('\n', '').replace(' ', '')


def test_unknown_gate():
    class UnknownGate(cirq.Gate):
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
