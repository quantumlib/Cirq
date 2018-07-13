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
from cirq.contrib import circuit_to_latex_using_qcircuit
from cirq.contrib.qcircuit.qcircuit_diagram import _QCircuitQubit
from cirq.contrib.qcircuit.qcircuit_diagrammable import (
    _FallbackQCircuitGate,
)


def test_QCircuitQubit():
    p = cirq.NamedQubit('x')
    q = _QCircuitQubit(p)
    assert repr(q) == '_QCircuitQubit({!r})'.format(p)

    assert q != 0

def test_FallbackQCircuitSymbolsGate():
    class TestGate(cirq.Gate):
        def __str__(self):
            return 'T'

    g = TestGate()
    f = _FallbackQCircuitGate(g)
    assert f.qcircuit_diagram_info(cirq.TextDiagramInfoArgs.UNINFORMED_DEFAULT
                                   ) == ('\\text{T:0}',)
    assert f.qcircuit_diagram_info(cirq.TextDiagramInfoArgs(
        known_qubits=None,
        known_qubit_count=2,
        use_unicode_characters=True,
        precision=None)) == ('\\text{T:0}', '\\text{T:1}')


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

    diagram = circuit_to_latex_using_qcircuit(
        circuit,
        qubit_order=cirq.QubitOrder.explicit([ali, car, bob]))
    assert diagram.strip() == """
\\Qcircuit @R=1em @C=0.75em { \\\\ 
 \\lstick{\\text{alice}}& \\qw &\\qw & \\gate{\\text{X}^{0.5}} \\qw & \\control \\qw & \\gate{\\text{H}} \\qw & \\meter \\qw &\\qw & \\control \\qw &\\qw\\\\
 \\lstick{\\text{carrier}}& \\qw & \\gate{\\text{H}} \\qw & \\control \\qw & \\targ \\qw \\qwx &\\qw & \\meter \\qw & \\control \\qw &\\qw \\qwx &\\qw\\\\
 \\lstick{\\text{bob}}& \\qw &\\qw & \\targ \\qw \\qwx &\\qw &\\qw &\\qw & \\targ \\qw \\qwx & \\control \\qw \\qwx &\\qw \\\\ 
 \\\\ }
        """.strip()
