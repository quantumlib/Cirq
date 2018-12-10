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

import cirq
import cirq.contrib.qcircuit as ccq


def test_get_qcircuit_diagram_info():
    qubits = cirq.NamedQubit('x'), cirq.NamedQubit('y')


    class FooGate(cirq.Gate, cirq.InterchangeableQubitsGate):
        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                                   ) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=('foo', 'foo'))

        def __str__(self):
            return 'FOO'

    gate = FooGate()
    op = gate(*qubits)
    qubit_map = {q: i for q, i in zip(qubits, (4, 3))}
    args = cirq.CircuitDiagramInfoArgs(
            known_qubits=qubits,
            known_qubit_count=None,
            use_unicode_characters=True,
            precision=3,
            qubit_map=qubit_map)
    actual_info = ccq.get_qcircuit_diagram_info(op, args)
    expected_info = cirq.CircuitDiagramInfo(
            ('\ghost{\\text{FOO}}', '\multigate{1}{\\text{FOO}}'),
            vconnected=False)
    assert actual_info == expected_info

    qubit_map = {q: i for q, i in zip(qubits, (2, 5))}
    args = cirq.CircuitDiagramInfoArgs(
            known_qubits=qubits,
            known_qubit_count=None,
            use_unicode_characters=True,
            precision=3,
            qubit_map=qubit_map)
    actual_info = ccq.get_qcircuit_diagram_info(op, args)
    expected_info = cirq.CircuitDiagramInfo(('\\gate{\\text{foo}}',) * 2)
    assert actual_info == expected_info

    actual_info = ccq.get_qcircuit_diagram_info(op,
            cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT)
    assert actual_info == expected_info

def test_swap_qcircuit_diagram_info():
    qubits = cirq.NamedQubit('x'), cirq.NamedQubit('y')
    gate = cirq.SwapPowGate(exponent=0.5)
    op = gate(*qubits)
    qubit_map = {q: i for i, q in enumerate(qubits)}
    args = cirq.CircuitDiagramInfoArgs(
            known_qubits=qubits,
            known_qubit_count=None,
            use_unicode_characters=True,
            precision=3,
            qubit_map=qubit_map)
    actual_info = ccq.get_qcircuit_diagram_info(op, args)
    name = '{\\text{SWAP}^{0.5}}'
    expected_info = cirq.CircuitDiagramInfo(
            ('\multigate{1}' + name, '\ghost' + name),
            exponent=0.5,
            vconnected=False)
    assert actual_info == expected_info

    gate = cirq.SwapPowGate()
    op = gate(*qubits)
    actual_info = ccq.get_qcircuit_diagram_info(op, args)
    wire_symbols = ('\\ar @{-}[1, 1]', '\\ar @{-}[-1, 1]')
    expected_info = cirq.CircuitDiagramInfo(
            wire_symbols=wire_symbols,
            vconnected=False,
            hconnected=False)
    assert actual_info == expected_info

    args.qubit_map = None
    actual_info = ccq.get_qcircuit_diagram_info(op, args)
    wire_symbols = ('\\gate{\\text{swap}}',) * 2
    expected_info = cirq.CircuitDiagramInfo(
            wire_symbols=wire_symbols)
    assert actual_info == expected_info
