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
    name = r'{\text{SWAP}^{0.5}}'
    expected_info = cirq.CircuitDiagramInfo(
            (r'\multigate{1}' + name, r'\ghost' + name),
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
            (r'\ghost{\text{SWAP}}', r'\multigate{1}{\text{SWAP}}'),
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
    expected_info = cirq.CircuitDiagramInfo((r'\gate{\text{Swap}}',) * 2)
    assert actual_info == expected_info

    actual_info = ccq.get_qcircuit_diagram_info(op,
            cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT)
    assert actual_info == expected_info
