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
import cirq.testing as ct


def assert_has_qcircuit_diagram(actual: cirq.Circuit, desired: str, **kwargs) -> None:
    """Determines if a given circuit has the desired qcircuit diagram.

    Args:
        actual: The circuit that was actually computed by some process.
        desired: The desired qcircuit diagram as a string. Newlines at the
            beginning and whitespace at the end are ignored.
        **kwargs: Keyword arguments to be passed to
            circuit_to_latex_using_qcircuit.
    """
    actual_diagram = ccq.circuit_to_latex_using_qcircuit(actual, **kwargs).lstrip('\n').rstrip()
    desired_diagram = desired.lstrip("\n").rstrip()
    assert actual_diagram == desired_diagram, (
        "Circuit's qcircuit diagram differs from the desired diagram.\n"
        '\n'
        'Diagram of actual circuit:\n'
        '{}\n'
        '\n'
        'Desired qcircuit diagram:\n'
        '{}\n'
        '\n'
        'Highlighted differences:\n'
        '{}\n'.format(
            actual_diagram,
            desired_diagram,
            ct.highlight_text_differences(actual_diagram, desired_diagram),
        )
    )


def test_fallback_diagram():
    class MagicGate(cirq.testing.ThreeQubitGate):
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

    circuit = cirq.Circuit(
        MagicOp(cirq.NamedQubit('b')),
        MagicGate().on(cirq.NamedQubit('b'), cirq.NamedQubit('a'), cirq.NamedQubit('c')),
    )
    expected_diagram = r"""
\Qcircuit @R=1em @C=0.75em {
 \\
 &\lstick{\text{a}}& \qw&                           \qw&\gate{\text{\#2}}       \qw    &\qw\\
 &\lstick{\text{b}}& \qw&\gate{\text{MagicOperate}} \qw&\gate{\text{MagicGate}} \qw\qwx&\qw\\
 &\lstick{\text{c}}& \qw&                           \qw&\gate{\text{\#3}}       \qw\qwx&\qw\\
 \\
}""".strip()
    assert_has_qcircuit_diagram(circuit, expected_diagram)


def test_teleportation_diagram():
    ali = cirq.NamedQubit('alice')
    car = cirq.NamedQubit('carrier')
    bob = cirq.NamedQubit('bob')

    circuit = cirq.Circuit(
        cirq.H(car),
        cirq.CNOT(car, bob),
        cirq.X(ali) ** 0.5,
        cirq.CNOT(ali, car),
        cirq.H(ali),
        [cirq.measure(ali), cirq.measure(car)],
        cirq.CNOT(car, bob),
        cirq.CZ(ali, bob),
    )

    expected_diagram = r"""
\Qcircuit @R=1em @C=0.75em {
 \\
 &\lstick{\text{alice}}&   \qw&\gate{\text{X}^{0.5}} \qw&         \qw    &\control \qw    &\gate{\text{H}} \qw&\meter   \qw    &\control \qw    &\qw\\
 &\lstick{\text{carrier}}& \qw&\gate{\text{H}}       \qw&\control \qw    &\targ    \qw\qwx&\meter          \qw&\control \qw    &         \qw\qwx&\qw\\
 &\lstick{\text{bob}}&     \qw&                      \qw&\targ    \qw\qwx&         \qw    &                \qw&\targ    \qw\qwx&\control \qw\qwx&\qw\\
 \\
}""".strip()
    assert_has_qcircuit_diagram(
        circuit, expected_diagram, qubit_order=cirq.QubitOrder.explicit([ali, car, bob])
    )


def test_other_diagram():
    a, b, c = cirq.LineQubit.range(3)

    circuit = cirq.Circuit(cirq.X(a), cirq.Y(b), cirq.Z(c))

    expected_diagram = r"""
\Qcircuit @R=1em @C=0.75em {
 \\
 &\lstick{\text{0}}& \qw&\targ           \qw&\qw\\
 &\lstick{\text{1}}& \qw&\gate{\text{Y}} \qw&\qw\\
 &\lstick{\text{2}}& \qw&\gate{\text{Z}} \qw&\qw\\
 \\
}""".strip()
    assert_has_qcircuit_diagram(circuit, expected_diagram)


def test_qcircuit_qubit_namer():
    from cirq.contrib.qcircuit import qcircuit_diagram

    assert qcircuit_diagram.qcircuit_qubit_namer(cirq.NamedQubit('q')) == r'\lstick{\text{q}}&'
    assert qcircuit_diagram.qcircuit_qubit_namer(cirq.NamedQubit('q_1')) == r'\lstick{\text{q\_1}}&'
    assert (
        qcircuit_diagram.qcircuit_qubit_namer(cirq.NamedQubit('q^1'))
        == r'\lstick{\text{q\textasciicircum{}1}}&'
    )
    assert (
        qcircuit_diagram.qcircuit_qubit_namer(cirq.NamedQubit('q_{1}'))
        == r'\lstick{\text{q\_\{1\}}}&'
    )


def test_two_cx_diagram():
    # test for no moment indication
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.CX(q0, q2), cirq.CX(q1, q3), cirq.CX(q0, q2), cirq.CX(q1, q3))
    expected_diagram = r"""
\Qcircuit @R=1em @C=0.75em {
 \\
 &\lstick{\text{0}}& \qw&\control \qw    &         \qw    &\control \qw    &         \qw    &\qw\\
 &\lstick{\text{1}}& \qw&         \qw\qwx&\control \qw    &         \qw\qwx&\control \qw    &\qw\\
 &\lstick{\text{2}}& \qw&\targ    \qw\qwx&         \qw\qwx&\targ    \qw\qwx&         \qw\qwx&\qw\\
 &\lstick{\text{3}}& \qw&         \qw    &\targ    \qw\qwx&         \qw    &\targ    \qw\qwx&\qw\\
 \\
}""".strip()
    assert_has_qcircuit_diagram(circuit, expected_diagram)


def test_sqrt_iswap_diagram():
    # test for proper rendering of ISWAP^{0.5}
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.ISWAP(q0, q1) ** 0.5)
    expected_diagram = r"""
\Qcircuit @R=1em @C=0.75em {
 \\
 &\lstick{\text{0}}& \qw&\multigate{1}{\text{ISWAP}^{0.5}} \qw&\qw\\
 &\lstick{\text{1}}& \qw&\ghost{\text{ISWAP}^{0.5}}        \qw&\qw\\
 \\
}""".strip()
    assert_has_qcircuit_diagram(circuit, expected_diagram)
