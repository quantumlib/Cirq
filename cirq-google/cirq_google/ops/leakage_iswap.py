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
from cirq_google.ops.internal_gate import InternalGate


class LeakageISWAP(InternalGate):
    """Leakage iSWAP (LeakageISWAP) Gate.                                                                
                                                                                                         
    A two-qutrit (or multi-level qubit) entangling gate designed for leakage
    removal. It operates by performing a coherent swap interaction in the
    two-excitation subspace, specifically coupling a leakage
    state to a state in the computational subspace.  

    Args:
        phase_matched: Whether to instead pre and post virtual Z operations
            to prevent phase accumulation in the computational subspace.
            Defaults to True.
    """

    def __init__(self, gate_name=None, gate_module=None, num_qubits=1, phase_matched=True,
                 **kwargs):
        self.phase_matched = phase_matched
        if phase_matched:
           super().__init__(gate_name="LeakageISWAPPhaseMatched", num_qubits=2)
        else:
           super().__init__(gate_name="LeakageISWAPUnmatched", num_qubits=2)

    def _num_qubits_(self) -> int:
        return 2

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> list[str]:
        return ["LiS", "LiS"]

    def _decompose_(self, qubits):
        return cirq.I.on_each(qubits)

    def _json_dict_(self):
        return {"phase_matched": self.phase_matched}

    def __repr__(self) -> str:
        return f'cirq_google.LeakageISWAP(phase_matched={self.phase_matched})'
