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
"""An optimization pass to put as many measurements possible at the end."""

from typing import List, Set, Tuple, cast
from cirq import circuits, ops, protocols


class SynchronizeTerminalMeasurements:
    """Move measurements to the end of the circuit.

    Move all measurements in a circuit to the final moment if it can accommodate
    them (without overlapping with other operations). If
    self._after_other_operations is true then a new moment will be added to the
    end of the circuit containing all the measurements that should be brought
    forward.
    """

    def __init__(self, after_other_operations: bool = True):
        """Inits SynchronizeTerminalMeasurements.

        Args:
            after_other_operations: Set by default. If the circuit's final
                moment contains non-measurement operations and this is set then
                a new empty moment is appended to the circuit before pushing
                measurements to the end.
        """
        self._after_other_operations = after_other_operations

    def __call__(self, circuit: circuits.Circuit):
        self.optimize_circuit(circuit)

    def optimize_circuit(self, circuit: circuits.Circuit) -> None:
        deletions: List[Tuple[int, ops.Operation]] = []
        terminal_measures: Set[ops.Operation] = set()
        qubits = circuit.all_qubits()
        for qubit in qubits:
            moment_index = cast(int, circuit.prev_moment_operating_on((qubit,)))
            op = cast(ops.Operation, circuit.operation_at(qubit, moment_index))
            if protocols.is_measurement(op):
                deletions.append((moment_index, op))
                terminal_measures.add(op)

        if not deletions:
            return

        circuit.batch_remove(deletions)
        if circuit[-1] and self._after_other_operations:
            # Can safely add to the end if
            # self._after_other_operations is false or we happen to get an
            # empty final moment before re-adding all the measurements.
            circuit.append(ops.Moment())

        for op in terminal_measures:
            circuit[-1] = circuit[-1].with_operation(op)
