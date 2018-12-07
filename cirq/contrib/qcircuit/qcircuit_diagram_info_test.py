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


"""
def test_get_multigate_parameters():
    qubits = cirq.LineQubit.range(5)
    known_qubits = tuple(qubits[:2])
    qubit_map = {q: i for i, q in enumerate(qubits[:2])}
    args = cirq.CircuitDiagramInfoArgs(
        known_qubits=known_qubits,
        known_qubit_count=None,
        use_unicode_characters=True,
        precision=3,
        qubit_map=qubit_map)
    assert ccq.get_multigate_parameters(cirq.CNOT, args) is None
    args.known_qubits = None
    assert ccq.get_multigate_parameters(cirq.CZ, args) is None
    args.known_qubits = known_qubits
    args.qubit_map = None
    assert ccq.get_multigate_parameters(cirq.CZ, args) is None

    a, b, c = qubits[:3]
    args.known_qubits = (a, b, c)
    args.qubit_map = {a: 2, b: 5, c: 1}
    assert ccq.get_multigate_parameters(cirq.CCZ, args) is None
    args.qubit_map = {a: 4, b: 3, c: 2}
    assert ccq.get_multigate_parameters(cirq.CCZ, args) == (2, 3)
"""

def test_get_qcircuit_diagram_info():
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
            connected=False)
    assert actual_info == expected_info

    gate = cirq.SWAP
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
            ('\ghost{\\text{SWAP}}', '\multigate{1}{\\text{SWAP}}'),
            connected=False)
    assert actual_info == expected_info

    qubit_map = {q: i for q, i in zip(qubits, (2, 5))}
    args = cirq.CircuitDiagramInfoArgs(
            known_qubits=qubits,
            known_qubit_count=None,
            use_unicode_characters=True,
            precision=3,
            qubit_map=qubit_map)
    actual_info = ccq.get_qcircuit_diagram_info(op, args)
    expected_info = cirq.CircuitDiagramInfo(('\\gate{\\text{swap}}',) * 2)
    assert actual_info == expected_info

    actual_info = ccq.get_qcircuit_diagram_info(op,
            cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT)
    assert actual_info == expected_info
