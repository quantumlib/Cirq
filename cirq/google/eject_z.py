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

"""An optimization pass that pushes Z gates later and later in the circuit."""

from typing import Iterator, Tuple, cast, Optional

from cirq import ops, extension
from cirq.circuits import Circuit, InsertStrategy, OptimizationPass
from cirq.google.decompositions import is_negligible_turn
from cirq.google.xmon_gates import ExpZGate
from cirq.value import Symbol


class EjectZ(OptimizationPass):
    """Removes Z gates by pushing them later and later until they merge.

    As Z gates are removed from the circuit, 'lost phase' builds up. As lost
    phase is pushed rightward, it modifies phaseable operations along the way.
    Eventually the lost phase is discharged into a 'drain'. Only Z gates
    without a parameter dependence are removed.

    There are three kinds of drains:
    - Measurement gates, which absorb phase by discarding it.
    - Parameterized Z gates, which absorb phase into their turns attribute.
    - The end of the circuit, which absorbs phase into a new Z gate.
    """

    def __init__(self,
                 tolerance: float = 0.0,
                 ext: extension.Extensions=None) -> None:
        """
        Args:
            tolerance: Maximum absolute error tolerance. The optimization is
                 permitted to simply drop negligible combinations of Z gates,
                 with a threshold determined by this tolerance.
            ext: Extensions object used for determining if gates are phaseable
                (i.e. if Z gates can pass through them).
        """
        self.tolerance = tolerance
        self.ext = ext or extension.Extensions()

    def optimize_circuit(self, circuit: Circuit):
        for qubit in circuit.all_qubits():
            for start, drain in self._find_optimization_range_drains(circuit,
                                                                     qubit):
                self._optimize_range(circuit, qubit, start, drain)

    def _find_optimization_range_drains(
            self,
            circuit: Circuit,
            qubit: ops.QubitId) -> Iterator[Tuple[int, int]]:
        """Finds ranges where Z gates can be pushed rightward.

        Args:
            circuit: The circuit being optimized.
            qubit: The qubit along which Z operations are being merged.

        Yields:
            (start, drain) tuples. Z gates on the given qubit from moments with
            indices in the range [start, drain) should all be merged into
            whatever is at the drain index.
        """
        start_z = None
        prev_z = None

        for i in range(len(circuit)):
            op = circuit.operation_at(qubit, i)
            if op is None:
                continue

            if start_z is None:
                # Unparameterized Zs start optimization ranges.
                if _try_get_known_z_half_turns(op) is not None:
                    start_z = i
                    prev_z = None

            elif _is_known_measurement(op):
                # Measurement acts like a drain. It destroys phase information.
                yield start_z, i
                start_z = None

            elif _try_get_known_z_half_turns(op) is not None:
                # Could be a drain. Depends if an unphaseable gate follows.
                prev_z = i

            elif not self.ext.can_cast(ops.PhaseableEffect, op):
                # Unphaseable gates force earlier draining.
                if prev_z is not None:
                    yield start_z, prev_z
                start_z = None

        # End of the circuit forces draining.
        if start_z is not None:
            yield start_z, len(circuit)

    def _optimize_range(self, circuit: Circuit, qubit: ops.QubitId,
                        start: int, drain: int):
        """Pushes Z gates from [start, drain) into the drain.

        Assumes no unphaseable gates will be crossed, and that the drain is
        valid.

        Args:
            circuit: The circuit being optimized.
            qubit: The qubit along which Z operations are being merged.
            start: The inclusive start of the range containing Z gates to
                eject.
            drain: The exclusive end of the range containing Z gates to eject.
                Also the index of where the effects of the Z gates should end
                up.
        """
        lost_phase_turns = 0.0

        for i in range(start, drain):
            op = circuit.operation_at(qubit, i)

            if op is None:
                # Empty.
                continue

            known_z_half_turns = _try_get_known_z_half_turns(op)
            if known_z_half_turns is not None:
                # Move Z effects out of the circuit and into lost_phase_turns.
                circuit.clear_operations_touching([qubit], [i])
                lost_phase_turns += known_z_half_turns / 2

            elif self.ext.can_cast(ops.PhaseableEffect, op):
                # Adjust phaseable gates to account for the lost phase.
                phaseable = self.ext.cast(ops.PhaseableEffect, op)
                k = op.qubits.index(qubit)
                circuit.clear_operations_touching(op.qubits, [i])
                circuit.insert(i + 1,
                               cast(ops.Operation,
                                    phaseable.phase_by(-lost_phase_turns, k)),
                               InsertStrategy.INLINE)

        self._drain_into(circuit, qubit, drain, lost_phase_turns)

    def _drain_into(self, circuit: Circuit, qubit: ops.QubitId,
                    drain: int, accumulated_phase: float):
        if is_negligible_turn(accumulated_phase, self.tolerance):
            return

        # Drain type: end of circuit.
        if drain == len(circuit):
            circuit.append(
                ExpZGate(half_turns=2*accumulated_phase).on(qubit),
                InsertStrategy.INLINE)
            return

        # Drain type: another Z gate.
        op = cast(ops.Operation, circuit.operation_at(qubit, drain))
        known_z_half_turns = _try_get_known_z_half_turns(op)
        if known_z_half_turns is not None:
            new_half_turns = known_z_half_turns + accumulated_phase * 2
            circuit.clear_operations_touching([qubit], [drain])
            circuit.insert(
                drain + 1,
                ExpZGate(half_turns=new_half_turns).on(qubit),
                InsertStrategy.INLINE)
            return

            # Drain type: measurement gate.
            # (Don't have to do anything.)


def _is_known_measurement(op: ops.Operation) -> bool:
    return (isinstance(op, ops.GateOperation) and
            isinstance(op.gate, ops.MeasurementGate))


def _try_get_known_z_half_turns(op: ops.Operation) -> Optional[float]:
    if not isinstance(op, ops.GateOperation):
        return None
    if not isinstance(op.gate, (ExpZGate, ops.RotZGate)):
        return None
    h = op.gate.half_turns
    if isinstance(h, Symbol):
        return None
    return h
