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

from cirq import ops, Circuit
from cirq.contrib import circuit_to_latex_using_qcircuit
from cirq.contrib.qcircuit.qcircuit_diagram import _QCircuitQubit
from cirq.contrib.qcircuit.qcircuit_diagrammable_gate import (
        _FallbackQCircuitSymbolsGate)


def test_QCircuitQubit():
    p = ops.NamedQubit('x')
    q = _QCircuitQubit(p)
    assert repr(q) == '_QCircuitQubit({!r})'.format(p)

    assert q != 0

def test_FallbackQCircuitSymbolsGate():
    class TestGate(ops.Gate):
        def __str__(self):
            return 'T'

    g = TestGate()
    f = _FallbackQCircuitSymbolsGate(g)
    assert f.qcircuit_wire_symbols() == ('\\text{T:0}',)
    assert f.qcircuit_wire_symbols(2) == ('\\text{T:0}', '\\text{T:1}')


def test_teleportation_diagram():
    ali = ops.NamedQubit('alice')
    car = ops.NamedQubit('carrier')
    bob = ops.NamedQubit('bob')

    circuit = Circuit.from_ops(
        ops.H(car),
        ops.CNOT(car, bob),
        ops.X(ali)**0.5,
        ops.CNOT(ali, car),
        ops.H(ali),
        [ops.measure(ali), ops.measure(car)],
        ops.CNOT(car, bob),
        ops.CZ(ali, bob))

    diagram = circuit_to_latex_using_qcircuit(
        circuit,
        qubit_order=ops.QubitOrder.explicit([ali, car, bob]))
    assert diagram.strip() == """
\\Qcircuit @R=1em @C=0.75em { \\\\ 
 \\lstick{\\text{alice}}& \\qw &\\qw & \\gate{\\text{X}^{0.5}} \\qw & \\control \\qw & \\gate{\\text{H}} \\qw & \\meter \\qw &\\qw & \\control \\qw &\\qw\\\\
 \\lstick{\\text{carrier}}& \\qw & \\gate{\\text{H}} \\qw & \\control \\qw & \\targ \\qw \\qwx &\\qw & \\meter \\qw & \\control \\qw &\\qw \\qwx &\\qw\\\\
 \\lstick{\\text{bob}}& \\qw &\\qw & \\targ \\qw \\qwx &\\qw &\\qw &\\qw & \\targ \\qw \\qwx & \\control \\qw \\qwx &\\qw \\\\ 
 \\\\ }
        """.strip()
