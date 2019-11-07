# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns


def test_controls():
    a, b = cirq.LineQubit.range(2)

    assert_url_to_circuit_returns('{"cols":[["•","X"]]}',
                                  cirq.Circuit(cirq.X(b).controlled_by(a),))
    assert_url_to_circuit_returns(
        '{"cols":[["◦","X"]]}',
        cirq.Circuit(
            cirq.X(a),
            cirq.X(b).controlled_by(a),
            cirq.X(a),
        ))

    assert_url_to_circuit_returns('{"cols":[["⊕","X"]]}',
                                  cirq.Circuit(
                                      cirq.Y(a)**0.5,
                                      cirq.X(b).controlled_by(a),
                                      cirq.Y(a)**-0.5,
                                  ),
                                  output_amplitudes_from_quirk=[
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": -0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                  ])
    assert_url_to_circuit_returns('{"cols":[["⊖","X"]]}',
                                  cirq.Circuit(
                                      cirq.Y(a)**-0.5,
                                      cirq.X(b).controlled_by(a),
                                      cirq.Y(a)**+0.5,
                                  ),
                                  output_amplitudes_from_quirk=[
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": -0.5,
                                          "i": 0
                                      },
                                  ])

    assert_url_to_circuit_returns('{"cols":[["⊗","X"]]}',
                                  cirq.Circuit(
                                      cirq.X(a)**-0.5,
                                      cirq.X(b).controlled_by(a),
                                      cirq.X(a)**+0.5,
                                  ),
                                  output_amplitudes_from_quirk=[
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0,
                                          "i": -0.5
                                      },
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0,
                                          "i": 0.5
                                      },
                                  ])
    assert_url_to_circuit_returns('{"cols":[["(/)","X"]]}',
                                  cirq.Circuit(
                                      cirq.X(a)**+0.5,
                                      cirq.X(b).controlled_by(a),
                                      cirq.X(a)**-0.5,
                                  ),
                                  output_amplitudes_from_quirk=[
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0,
                                          "i": 0.5
                                      },
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0,
                                          "i": -0.5
                                      },
                                  ])

    qs = cirq.LineQubit.range(8)
    assert_url_to_circuit_returns(
        '{"cols":[["X","•","◦","⊕","⊖","⊗","(/)","Z"]]}',
        cirq.Circuit(
            cirq.X(qs[2]),
            cirq.Y(qs[3])**0.5,
            cirq.Y(qs[4])**-0.5,
            cirq.X(qs[5])**-0.5,
            cirq.X(qs[6])**0.5,
            cirq.X(qs[0]).controlled_by(*qs[1:7]),
            cirq.Z(qs[7]).controlled_by(*qs[1:7]),
            cirq.X(qs[6])**-0.5,
            cirq.X(qs[5])**0.5,
            cirq.Y(qs[4])**0.5,
            cirq.Y(qs[3])**-0.5,
            cirq.X(qs[2]),
        ))


def test_parity_controls():
    a, b, c, d, e = cirq.LineQubit.range(5)

    assert_url_to_circuit_returns(
        '{"cols":[["Y","xpar","ypar","zpar","Z"]]}',
        cirq.Circuit(
            cirq.Y(b)**0.5,
            cirq.X(c)**-0.5,
            cirq.CNOT(c, b),
            cirq.CNOT(d, b),
            cirq.Y(a).controlled_by(b),
            cirq.Z(e).controlled_by(b),
            cirq.CNOT(d, b),
            cirq.CNOT(c, b),
            cirq.X(c)**0.5,
            cirq.Y(b)**-0.5,
        ))


def test_control_with_line_qubits_mapped_to():
    a, b = cirq.LineQubit.range(2)
    a2, b2 = cirq.NamedQubit.range(2, prefix='q')
    cell = cirq.interop.quirk.cells.control_cells.ControlCell(
        a, [cirq.Y(b)**0.5])
    mapped_cell = cirq.interop.quirk.cells.control_cells.ControlCell(
        a2, [cirq.Y(b2)**0.5])
    assert cell != mapped_cell
    assert cell.with_line_qubits_mapped_to([a2, b2]) == mapped_cell


def test_parity_control_with_line_qubits_mapped_to():
    a, b, c = cirq.LineQubit.range(3)
    a2, b2, c2 = cirq.NamedQubit.range(3, prefix='q')
    cell = cirq.interop.quirk.cells.control_cells.ParityControlCell(
        [a, b], [cirq.Y(c)**0.5])
    mapped_cell = cirq.interop.quirk.cells.control_cells.ParityControlCell(
        [a2, b2], [cirq.Y(c2)**0.5])
    assert cell != mapped_cell
    assert cell.with_line_qubits_mapped_to([a2, b2, c2]) == mapped_cell


def test_repr():
    a, b, c = cirq.LineQubit.range(3)
    cirq.testing.assert_equivalent_repr(
        cirq.interop.quirk.cells.control_cells.ControlCell(a, [cirq.Y(b)**0.5]))
    cirq.testing.assert_equivalent_repr(
        cirq.interop.quirk.cells.control_cells.ParityControlCell(
            [a, b], [cirq.Y(c)**0.5]))
