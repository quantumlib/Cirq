# Copyright 2026 The Cirq Developers
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


@cirq.value_equality
class MultilevelResetViaResonator(cirq.Gate):
    """Multilevel qubit reset with resonator.

    This is a specialized type of reset that can be used to clear out
    excited levels by utilizing the attached resonator.  Useful in
    performing Quantum Error Correction experiments that require multiple
    rounds of measurement and reset.
    """

    def __init__(self, num_qubits=1, **kwargs):
        self.num_qubits = num_qubits

    def _num_qubits_(self) -> int:
        return self.num_qubits

    def _value_equality_values_(self):
        return (self.num_qubits,)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> list[str]:
        return ["[R (ML)]"]

    def is_reset_gate(self) -> bool:
        return True

    def _decompose_(self, qubits):
        return cirq.reset_each(*qubits)

    def _json_dict_(self):
        return {}

    def __repr__(self) -> str:
        return 'cirq_google.MultilevelResetViaResonator()'
