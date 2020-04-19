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

def test_inconclusive():
    qubits = [cirq.NamedQubit('a'),cirq.NamedQubit('b'),cirq.NamedQubit('c')]
    q_map = {q: i for i, q in enumerate(qubits)}
    s = cirq.CliffordState(q_map)
    # assert cirq.apply_clifford_tableau(cirq.X, s, (q_map[qubits[0]],))
    assert cirq.apply_clifford_tableau(cirq.X**6, s, (q_map[qubits[0]],))
    # assert cirq.apply_clifford_tableau(cirq.X**1.5, s, (q_map[qubits[0]],))
    assert cirq.apply_clifford_tableau(cirq.X**2.5, s, (q_map[qubits[0]],))
    assert cirq.apply_clifford_tableau(cirq.Z, s, (q_map[qubits[0]],))