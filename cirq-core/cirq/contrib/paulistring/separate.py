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

from typing import Tuple

from cirq import ops, circuits

from cirq.contrib.paulistring.convert_gate_set import converted_gate_set


# TODO(#3388) Add documentation for Args.
# pylint: disable=missing-param-doc
def convert_and_separate_circuit(
    circuit: circuits.Circuit,
    leave_cliffords: bool = True,
    atol: float = 1e-8,
) -> Tuple[circuits.Circuit, circuits.Circuit]:
    """Converts any circuit into two circuits where (circuit_left+circuit_right)
    is equivalent to the given circuit.

    Args:
        circuit: Any Circuit that cirq.google.optimized_for_xmon() supports.
            All gates should either provide a decomposition or have a known one
            or two qubit unitary matrix.

    Returns:
        (circuit_left, circuit_right)

        circuit_left contains only PauliStringPhasor operations.

        circuit_right is a Clifford circuit which contains only
        SingleQubitCliffordGate and PauliInteractionGate gates.
        It also contains MeasurementGates if the
        given circuit contains measurements.
    """
    circuit = converted_gate_set(circuit, no_clifford_gates=not leave_cliffords, atol=atol)
    return pauli_string_half(circuit), regular_half(circuit)


# pylint: enable=missing-param-doc
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
        ops.Moment(op for op in moment.operations if not isinstance(op, ops.PauliStringPhasor))
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


def _pull_non_clifford_before(circuit: circuits.Circuit) -> ops.OP_TREE:
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
