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

import itertools

import cirq
import cirq.contrib.qcircuit as ccq


def assert_qcircuit_diagrams_equal(actual: str, expected: str):
    zipped_lines = itertools.zip_longest(
            actual.split(r'\\'), expected.split(r'\\'), fillvalue='')
    for line_number, (actual_line, expected_line) in enumerate(zipped_lines):
        zipped_entries = itertools.zip_longest(
            actual_line.split(r'&'), expected_line.split(r'&'), fillvalue='')
        for entry_number, (actual_entry, expected_entry) in enumerate(
                zipped_entries):
            assert (actual_entry.strip() == expected_entry.strip()
                ), ('Line {} differs in entry {}:'.format(
                    line_number, entry_number) +
                    '{} vs. {}'.format(actual_entry, expected_entry))


def test_fallback_diagram():
    class MagicGate(cirq.Gate):
        def __str__(self):
            return 'MagicGate'

    class MagicOp(cirq.Operation):
        def __init__(self, *qubits):
            self._qubits = qubits

        def with_qubits(self, *new_qubits):
            return MagicOp(*new_qubits)

        @property
        def qubits(self):
            return self._qubits

        def __str__(self):
            return 'MagicOperate'

    circuit = cirq.Circuit.from_ops(
        MagicOp(cirq.NamedQubit('b')),
        MagicGate().on(cirq.NamedQubit('b'),
                       cirq.NamedQubit('a'),
                       cirq.NamedQubit('c')))
    actual_diagram = ccq.circuit_to_latex_using_qcircuit(circuit)
    expected_diagram = r"""
\Qcircuit @R=1em @C=0.75em {
 \\
 &\lstick{\text{a}}& \qw&                    \qw&\gate{\text{\#2}}       \qw    &\qw\\
 &\lstick{\text{b}}& \qw&\gate{\text{MagicOperate}} \qw&\gate{\text{MagicGate}} \qw\qwx&\qw\\
 &\lstick{\text{c}}& \qw&                    \qw&\gate{\text{\#3}}       \qw\qwx&\qw\\
 \\
}""".strip()
    assert_qcircuit_diagrams_equal(actual_diagram, expected_diagram)


def test_teleportation_diagram():
    ali = cirq.NamedQubit('alice')
    car = cirq.NamedQubit('carrier')
    bob = cirq.NamedQubit('bob')

    circuit = cirq.Circuit.from_ops(
        cirq.H(car),
        cirq.CNOT(car, bob),
        cirq.X(ali)**0.5,
        cirq.CNOT(ali, car),
        cirq.H(ali),
        [cirq.measure(ali), cirq.measure(car)],
        cirq.CNOT(car, bob),
        cirq.CZ(ali, bob))

    actual_diagram = ccq.circuit_to_latex_using_qcircuit(
        circuit,
        qubit_order=cirq.QubitOrder.explicit([ali, car, bob]))
    expected_diagram = r"""
\Qcircuit @R=1em @C=0.75em {
 \\
 &\lstick{\text{alice}}&   \qw&                \qw&\gate{\text{X}^{0.5}} \qw    &\control \qw    &\gate{\text{H}} \qw&\meter \qw&         \qw    &\control \qw    &\qw\\
 &\lstick{\text{carrier}}& \qw&\gate{\text{H}} \qw&\control              \qw    &\targ    \qw\qwx&                \qw&\meter \qw&\control \qw    &         \qw\qwx&\qw\\
 &\lstick{\text{bob}}&     \qw&                \qw&\targ                 \qw\qwx&         \qw    &                \qw&       \qw&\targ    \qw\qwx&\control \qw\qwx&\qw\\
 \\
}""".strip()
    assert_qcircuit_diagrams_equal(actual_diagram, expected_diagram)


def test_other_diagram():
    a, b, c = cirq.LineQubit.range(3)

    circuit = cirq.Circuit.from_ops(
        cirq.X(a),
        cirq.Y(b),
        cirq.Z(c))

    actual_diagram = ccq.circuit_to_latex_using_qcircuit(circuit)
    expected_diagram = r"""
\Qcircuit @R=1em @C=0.75em {
 \\
 &\lstick{\text{0}}& \qw&\targ           \qw&\qw\\
 &\lstick{\text{1}}& \qw&\gate{\text{Y}} \qw&\qw\\
 &\lstick{\text{2}}& \qw&\gate{\text{Z}} \qw&\qw\\
 \\
}""".strip()
    assert_qcircuit_diagrams_equal(actual_diagram, expected_diagram)
