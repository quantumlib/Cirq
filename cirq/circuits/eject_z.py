# Copyright 2017 Google LLC
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

from typing import Iterator, Tuple

from cirq import ops
from cirq.circuits import util
from cirq.circuits.circuit import Circuit
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.optimization_pass import OptimizationPass


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

    def __init__(self, tolerance=0):
        self.tolerance = tolerance

    def optimize_circuit(self, circuit: Circuit):
        qubits = {
            q
            for m in circuit.moments for op in m.operations for q in op.qubits
        }
        for qubit in qubits:
            for start, drain in self._find_optimization_range_drains(circuit,
                                                                     qubit):
                self._optimize_range(circuit, qubit, start, drain)

    @staticmethod
    def _find_optimization_range_drains(
            circuit: Circuit,
            qubit: ops.QubitId) -> Iterator[Tuple[int, int]]:
        """Finds ranges where Z gates can be pushed rightward.

    Args:
      circuit: The circuit being optimized.
      qubit: The qubit along which Z operations are being merged.

    Yields:
      (start, drain) tuples. Z gates on the given qubit from moments with
      indices in the range [start, drain) should all be merged into whatever is
      at the drain index.
    """
        start_z = None
        prev_z = None

        for i in range(len(circuit.moments)):
            op = circuit.operation_at(qubit, i)
            if op is None:
                continue

            if start_z is None:
                # Unparameterized Zs start optimization ranges.
                if (isinstance(op.gate, ops.ExpZGate) and
                        not op.gate.turns_param_key):
                    start_z = i
                    prev_z = None

            elif isinstance(op.gate, ops.MeasurementGate):
                # Measurement acts like a drain. It destroys phase information.
                yield start_z, i
                start_z = None

            elif isinstance(op.gate, ops.ExpZGate):
                if op.gate.turns_param_key:
                    # Parameterized Z gates can't move, but can drain.
                    yield start_z, i
                    start_z = None
                else:
                    # Could be a drain. Depends if an unphaseable gate follows.
                    prev_z = i

            elif not isinstance(op.gate, ops.PhaseableGate):
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
        lost_phase_turns = 0

        for i in range(start, drain):
            op = circuit.operation_at(qubit, i)

            if op is None:
                # Empty.
                pass

            elif isinstance(op.gate, ops.ExpZGate):
                # Move Z effects out of the circuit and into lost_phase_turns.
                circuit.clear_operations_touching([qubit], [i])
                lost_phase_turns += op.gate.half_turns / 2

            elif isinstance(op.gate, ops.PhaseableGate):
                # Adjust phaseable gates to account for the lost phase.
                k = op.qubits.index(qubit)
                circuit.clear_operations_touching(op.qubits, [i])
                circuit.insert(i + 1,
                               op.gate.phase_by(-lost_phase_turns, k).on(
                                   *op.qubits),
                               InsertStrategy.INLINE)

        self._drain_into(circuit, qubit, drain, lost_phase_turns)

    def _drain_into(self, circuit: Circuit, qubit: ops.QubitId,
                    drain: int, accumulated_phase: float):
        if util.is_negligible_turn(accumulated_phase, self.tolerance):
            return

        # Drain type: end of circuit.
        if drain == len(circuit.moments):
            circuit.append(
                ops.ZGate(half_turns=2*accumulated_phase).on(qubit),
                InsertStrategy.INLINE)
            return

        # Drain type: another Z gate.
        op = circuit.operation_at(qubit, drain)
        if isinstance(op.gate, ops.ExpZGate):
            circuit.clear_operations_touching([qubit], [drain])
            circuit.insert(
                drain + 1,
                ops.ExpZGate(
                    op.gate.turns_param_key,
                    accumulated_phase + op.gate.half_turns / 2).on(qubit),
                InsertStrategy.INLINE)
            return

            # Drain type: measurement gate.
            # (Don't have to do anything.)
