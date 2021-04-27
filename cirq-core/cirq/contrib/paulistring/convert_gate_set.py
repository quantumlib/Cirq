# Copyright 2018 The ops Developers
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

from cirq import circuits, optimizers

from cirq.contrib.paulistring.convert_to_pauli_string_phasors import ConvertToPauliStringPhasors


def converted_gate_set(
    circuit: circuits.Circuit,
    no_clifford_gates: bool = False,
    atol: float = 1e-8,
) -> circuits.Circuit:
    """Returns a new, equivalent circuit using the gate set
    {SingleQubitCliffordGate,
    CZ/PauliInteractionGate, PauliStringPhasor}.
    """
    conv_circuit = circuits.Circuit(circuit)
    optimizers.ConvertToCzAndSingleGates().optimize_circuit(conv_circuit)
    optimizers.MergeSingleQubitGates().optimize_circuit(conv_circuit)
    ConvertToPauliStringPhasors(
        ignore_failures=True,
        keep_clifford=not no_clifford_gates,
        atol=atol,
    ).optimize_circuit(conv_circuit)
    optimizers.DropEmptyMoments().optimize_circuit(conv_circuit)
    return conv_circuit
