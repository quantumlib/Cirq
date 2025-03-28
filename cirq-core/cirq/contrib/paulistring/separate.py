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

from typing import Iterator, Tuple

from cirq import circuits, ops, transformers
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset


def convert_and_separate_circuit(
    circuit: circuits.Circuit, leave_cliffords: bool = True, atol: float = 1e-8
) -> Tuple[circuits.Circuit, circuits.Circuit]:
    """Converts a circuit into two, one made of PauliStringPhasor and the other Clifford gates.

    Args:
        circuit: Any Circuit that cirq.google.optimized_for_xmon() supports.
            All gates should either provide a decomposition or have a known one
            or two qubit unitary matrix.
        leave_cliffords: If set, single qubit rotations in the Clifford group
                are not converted to SingleQubitCliffordGates.
        atol: The absolute tolerance for the conversion.

    Returns:
        (circuit_left, circuit_right)

        circuit_left contains only PauliStringPhasor operations.

        circuit_right is a Clifford circuit which contains only
        SingleQubitCliffordGate and PauliInteractionGate gates.
        It also contains MeasurementGates if the
        given circuit contains measurements.

    """
    single_qubit_target = (
        CliffordTargetGateset.SingleQubitTarget.PAULI_STRING_PHASORS_AND_CLIFFORDS
        if leave_cliffords
        else CliffordTargetGateset.SingleQubitTarget.PAULI_STRING_PHASORS
    )
    circuit = transformers.optimize_for_target_gateset(
        circuit, gateset=CliffordTargetGateset(atol=atol, single_qubit_target=single_qubit_target)
    )
    return pauli_string_half(circuit), regular_half(circuit)


def regular_half(circuit: circuits.Circuit) -> circuits.Circuit:
    """Return only the Clifford part of a circuit.  See
    convert_and_separate_circuit().

    Args:
        circuit: A Circuit with the gate set {SingleQubitCliffordGate,
            PauliInteractionGate, PauliStringPhasor}.

    Returns:
        A Circuit with SingleQubitCliffordGate and PauliInteractionGate gates.
        It also contains MeasurementGates if the given
        circuit contains measurements.
    """
    return circuits.Circuit(
        circuits.Moment(op for op in moment.operations if not isinstance(op, ops.PauliStringPhasor))
        for moment in circuit
    )


def pauli_string_half(circuit: circuits.Circuit) -> circuits.Circuit:
    """Return only the non-Clifford part of a circuit.  See
    convert_and_separate_circuit().

    Args:
        circuit: A Circuit with the gate set {SingleQubitCliffordGate,
            PauliInteractionGate, PauliStringPhasor}.

    Returns:
        A Circuit with only PauliStringPhasor operations.
    """
    return circuits.Circuit(
        _pull_non_clifford_before(circuit), strategy=circuits.InsertStrategy.EARLIEST
    )


def _pull_non_clifford_before(circuit: circuits.Circuit) -> Iterator[ops.OP_TREE]:
    def _iter_ops_range_reversed(moment_end):
        for i in reversed(range(moment_end)):
            moment = circuit[i]
            for op in moment.operations:
                if not isinstance(op, ops.PauliStringPhasor):
                    yield op

    for i, moment in enumerate(circuit):
        for op in moment.operations:
            if isinstance(op, ops.PauliStringPhasor):
                ops_to_cross = _iter_ops_range_reversed(i)
                yield op.pass_operations_over(ops_to_cross)
