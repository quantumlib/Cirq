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


@cirq.testing.only_test_in_python3
def test_text_diagram_info_repr():
    info = cirq.CircuitDiagramInfo(('X', 'Y'), 2)
    assert repr(info) == ("cirq.DiagramInfo(wire_symbols=('X', 'Y')"
                          ", exponent=2, connected=True)")


def test_circuit_diagram_info_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.CircuitDiagramInfo(('X',)))
    eq.add_equality_group(cirq.CircuitDiagramInfo(('X', 'Y')),
                          cirq.CircuitDiagramInfo(('X', 'Y'), 1))
    eq.add_equality_group(cirq.CircuitDiagramInfo(('Z',), 2))
    eq.add_equality_group(cirq.CircuitDiagramInfo(('Z', 'Z'), 2))
    eq.add_equality_group(cirq.CircuitDiagramInfo(('Z',), 3))


