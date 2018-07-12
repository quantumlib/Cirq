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
from cirq.circuits import (
    Circuit, InsertStrategy, OptimizationPass, DropNegligible,
)
from cirq.google import xmon_gates
from cirq.google.decompositions import is_negligible_turn
from cirq.google.xmon_gates import ExpZGate
from cirq.value import Symbol


class EjectFullXY(OptimizationPass):
    def __init__(self,
                 tolerance: float = 0.0,
                 ext: extension.Extensions=None) -> None:
        self.tolerance = tolerance
        self.ext = ext or extension.Extensions()

    def _followups(self):
        return [DropNegligible(tolerance=self.tolerance,
                               extensions=self.ext)]

    def optimize_circuit(self, circuit: Circuit):
        qubits = {
            q
            for m in circuit.moments for op in m.operations for q in op.qubits
        }
        for qubit in qubits:
            for start, drain in self._find_optimization_range_drains(circuit,
                                                                     qubit):
                self._optimize_range(circuit, qubit, start, drain)

    def cross(self, q: ops.QubitId, other: ops.Operation):
        gate = xmon_gates.XmonGate.try_get_xmon_gate(other)
        if gate is None:
            return None
        param = self.ext.try_cast(ops.ParameterizableEffect, gate)
        if param is not None and param.is_parameterized():
            return None

        if isinstance(gate, xmon_gates.Exp11Gate):
            if gate.half_turns != 1:
                return None
            other_qubit = list(set(other.qubits) - {q})[0]
            return [gate, xmon_gates.ExpZGate().on(other_qubit)]

        if isinstance(gate, xmon_gates.ExpZGate):
            return gate.inverse().on(other.qubits)
        if isinstance(gate, xmon_gates.ExpWGate):
            if gate.half_turns == 1:
                pass
                # sequence terminates in a cancellation
            return xmon_gates.ExpWGate(axis_half_turns=-gate.axis_half_turns,
                                       half_turns=gate.half_turns)
        if isinstance(gate, xmon_gates.XmonMeasurementGate):
            mask = list(gate.invert_mask or ())
            d = len(other.qubits) - len(gate.invert_mask)
            if d > 0:
                gate.invert_mask += [False,] * d
            mask[other.qubits.index(q)] ^= True
            return xmon_gates.XmonMeasurementGate(gate.key, tuple(mask)
                                                  ).on(*other.qubits)

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

        for i in range(len(circuit.moments)):
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
            yield start_z, len(circuit.moments)

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
        if drain == len(circuit.moments):
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
