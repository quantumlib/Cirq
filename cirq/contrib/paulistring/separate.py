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

from cirq.contrib.paulistring.pauli_string_phasor import PauliStringPhasor
from cirq.contrib.paulistring.convert_gate_set import converted_gate_set


def convert_and_separate_circuit(circuit: circuits.Circuit
                                 ) -> Tuple[circuits.Circuit, circuits.Circuit]:
    """Converts any circuit into two circuits where (circuit_left+circuit_right)
    is equivalent to circuit.

    circuit_left contains only PauliStringPhasor operations.

    circuit_right is a Clifford circuit which contains only CliffordGate and
    PauliInteractionGate gates.

    Args:
        circuit: Any Circuit with any kind of gates.

    Returns:
        (circuit_left, circuit_right)

        circuit_left contains only PauliStringPhasor operations.

        circuit_right is a Clifford circuit which contains only CliffordGate and
        PauliInteractionGate gates.
    """
    circuit = converted_gate_set(circuit)
    return separate_circuit(circuit)


def separate_circuit(circuit: circuits.Circuit
                     ) -> Tuple[circuits.Circuit, circuits.Circuit]:
    """Converts a circuit into two circuits where (circuit_left+circuit_right)
    is equivalent to circuit.

    Args:
        circuit: A Circuit with the gate set {CliffordGate,
            PauliInteractionGate, PauliStringPhasor}.

    Returns:
        (circuit_left, circuit_right)

        circuit_left contains only PauliStringPhasor operations.

        circuit_right is a Clifford circuit which contains only CliffordGate and
        PauliInteractionGate gates.
    """
    return non_clifford_half(circuit), clifford_half(circuit)


def clifford_half(circuit: circuits.Circuit) -> circuits.Circuit:
    """Return only circuit_right from separate_circuit().  This is a Clifford
    circuit.

    Args:
        circuit: A Circuit with the gate set {CliffordGate,
            PauliInteractionGate, PauliStringPhasor}.

    Returns:
        A Circuit with CliffordGate and PauliInteractionGate gates.
    """
    return circuits.Circuit(
                circuits.Moment(op for op in moment.operations
                                if not isinstance(op, PauliStringPhasor))
                                for moment in circuit)


def non_clifford_half(circuit: circuits.Circuit) -> circuits.Circuit:
    """Return only circuit_left from separate_circuit().

    Args:
        circuit: A Circuit with the gate set {CliffordGate,
            PauliInteractionGate, PauliStringPhasor}.

    Returns:
        A Circuit with only PauliStringPhasor operations.
    """
    return circuits.Circuit.from_ops(
            _pull_non_clifford_before(circuit),
            strategy=circuits.InsertStrategy.EARLIEST)


def _pull_non_clifford_before(circuit: circuits.Circuit) -> ops.OP_TREE:
    def _iter_ops_range_reversed(moment_start, moment_end):
        for i in reversed(range(moment_start, moment_end)):
            moment = circuit[i]
            for op in moment.operations:
                if not isinstance(op, PauliStringPhasor):
                    yield op

    for i, moment in enumerate(circuit):
        for op in moment.operations:
            if isinstance(op, PauliStringPhasor):
                ops_to_cross = _iter_ops_range_reversed(0, i)
                yield op.pass_operations_over(ops_to_cross)
