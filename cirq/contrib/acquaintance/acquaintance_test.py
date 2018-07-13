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

from cirq import NamedQubit
from cirq.circuits import Circuit, Moment
from cirq.contrib.acquaintance.acquaintance import ACQUAINT
from cirq.ops import TextDiagramInfoArgs

def test_acquaintance_repr():
    assert repr(ACQUAINT) == 'Acq'

def test_text_diagram_info():
    qubits = [NamedQubit(s) for s in 'xyz']
    circuit = Circuit([Moment([ACQUAINT(*qubits)])])
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = """
x: ───█───
      │
y: ───█───
      │
z: ───█───
    """.strip()
    print(actual_text_diagram)
    assert actual_text_diagram == expected_text_diagram

def test_acquaintance_gate_unknown_qubit_count():
    g = ACQUAINT
    args = TextDiagramInfoArgs.UNINFORMED_DEFAULT
    assert g.text_diagram_info(args) == NotImplemented
