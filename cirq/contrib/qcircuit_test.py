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

from cirq import ops, Circuit
from cirq.contrib import circuit_to_latex_using_qcircuit


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
        [ops.MeasurementGate()(ali), ops.MeasurementGate()(car)],
        ops.CNOT(car, bob),
        ops.CZ(ali, bob))

    diagram = circuit_to_latex_using_qcircuit(
        circuit,
        basis=ops.Basis.explicit([ali, car, bob]))
    assert diagram.strip() == """
\\Qcircuit @R=1em @C=0.75em { \\\\ 
 \\lstick{\\text{alice}}& \\qw &\\qw & \\gate{\\text{X}^{0.5}} \\qw & \\control \\qw & \\gate{\\text{H}} \\qw & \\meter \\qw &\\qw & \\control \\qw &\\qw\\\\
 \\lstick{\\text{carrier}}& \\qw & \\gate{\\text{H}} \\qw & \\control \\qw & \\targ \\qw \\qwx &\\qw & \\meter \\qw & \\control \\qw &\\qw \\qwx &\\qw\\\\
 \\lstick{\\text{bob}}& \\qw &\\qw & \\targ \\qw \\qwx &\\qw &\\qw &\\qw & \\targ \\qw \\qwx & \\control \\qw \\qwx &\\qw \\\\ 
 \\\\ }
        """.strip()
