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

from cirq import circuits, transformers, _compat

from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset


@_compat.deprecated(
    deadline='v0.16',
    fix='Use cirq.optimize_for_target_gateset with cirq.contrib.paulistring.CliffordTargetGateset.',
)
def converted_gate_set(
    circuit: circuits.Circuit, no_clifford_gates: bool = False, atol: float = 1e-8
) -> circuits.Circuit:
    """Returns a new, equivalent circuit using the gate set
    {SingleQubitCliffordGate, CZ/PauliInteractionGate, PauliStringPhasor}.
    """
    single_qubit_target = (
        CliffordTargetGateset.SingleQubitTarget.PAULI_STRING_PHASORS
        if no_clifford_gates
        else CliffordTargetGateset.SingleQubitTarget.PAULI_STRING_PHASORS_AND_CLIFFORDS
    )
    gateset = CliffordTargetGateset(single_qubit_target=single_qubit_target, atol=atol)
    return transformers.optimize_for_target_gateset(circuit, gateset=gateset)
