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

from typing import Optional, cast, TYPE_CHECKING

from collections import defaultdict

from cirq import circuits, ops, extension
from cirq.google.decompositions import is_negligible_turn
from cirq.google.xmon_gates import ExpZGate
from cirq.value import Symbol

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Dict, List, Tuple


class EjectZ(circuits.OptimizationPass):
    """Pushes Z gates towards the end of the circuit.

    As the Z gates get pushed they may absorb other Z gates, get absorbed into
    measurements, cross CZ gates, cross W gates (by phasing them), etc.
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

    def _inplace_followups(self):
        return [circuits.DropNegligible(tolerance=self.tolerance,
                                        extensions=self.ext)]

    def optimize_circuit(self, circuit: circuits.Circuit):
        turns_state = defaultdict(lambda: 0)  # type: Dict[ops.QubitId, float]

        def dump_phases(qubits, index):
            for q in qubits:
                p = turns_state[q]
                if not is_negligible_turn(p, self.tolerance):
                    dump_op = ExpZGate(half_turns=p * 2).on(q)
                    insertions.append((index, dump_op))
                turns_state[q] = 0

        deletions = []  # type: List[Tuple[int, ops.Operation]]
        inline_intos = []  # type: List[Tuple[int, ops.Operation]]
        insertions = []  # type: List[Tuple[int, ops.Operation]]
        for moment_index, moment in enumerate(circuit):
            for op in moment.operations:
                h = _try_get_known_z_half_turns(op)
                if h is not None:
                    q = op.qubits[0]
                    turns_state[q] += h / 2
                    deletions.append((moment_index, op))
                    continue

                if ops.MeasurementGate.is_measurement(op):
                    for q in op.qubits:
                        turns_state[q] = 0

                phases = [turns_state[q] for q in op.qubits]
                if all(is_negligible_turn(p, self.tolerance) for p in phases):
                    continue

                phaseable = self.ext.try_cast(ops.PhaseableEffect, op)
                if phaseable is not None:
                    for i, p in enumerate(phases):
                        if p:
                            phaseable = phaseable.phase_by(-p, i)
                    deletions.append((moment_index, op))
                    inline_intos.append((moment_index,
                                         cast(ops.Operation, phaseable)))
                    continue

                dump_phases(op.qubits, moment_index)

        dump_phases(turns_state.keys(), len(circuit))
        circuit.batch_remove(deletions)
        circuit.batch_insert_into(inline_intos)
        circuit.batch_insert(insertions)


def _try_get_known_z_half_turns(op: ops.Operation) -> Optional[float]:
    if not isinstance(op, ops.GateOperation):
        return None
    if not isinstance(op.gate, (ExpZGate, ops.RotZGate)):
        return None
    h = op.gate.half_turns
    if isinstance(h, Symbol):
        return None
    return h
