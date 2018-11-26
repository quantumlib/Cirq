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

from typing import Optional, cast, TYPE_CHECKING, Iterable

from collections import defaultdict

from cirq import circuits, ops, value, protocols
from cirq.optimizers import decompositions

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Dict, List, Tuple


class EjectZ(circuits.OptimizationPass):
    """Pushes Z gates towards the end of the circuit.

    As the Z gates get pushed they may absorb other Z gates, get absorbed into
    measurements, cross CZ gates, cross W gates (by phasing them), etc.
    """

    def __init__(self, tolerance: float = 0.0) -> None:
        """
        Args:
            tolerance: Maximum absolute error tolerance. The optimization is
                 permitted to simply drop negligible combinations of Z gates,
                 with a threshold determined by this tolerance.
        """
        self.tolerance = tolerance

    def optimize_circuit(self, circuit: circuits.Circuit):
        # Tracks qubit phases (in half turns; multiply by pi to get radians).
        qubit_phase = defaultdict(lambda: 0)  # type: Dict[ops.QubitId, float]

        def dump_tracked_phase(qubits: Iterable[ops.QubitId],
                               index: int) -> None:
            """Zeroes qubit_phase entries by emitting Z gates."""
            for q in qubits:
                p = qubit_phase[q]
                if not decompositions.is_negligible_turn(p, self.tolerance):
                    dump_op = ops.Z(q)**(p * 2)
                    insertions.append((index, dump_op))
                qubit_phase[q] = 0

        deletions = []  # type: List[Tuple[int, ops.Operation]]
        inline_intos = []  # type: List[Tuple[int, ops.Operation]]
        insertions = []  # type: List[Tuple[int, ops.Operation]]
        for moment_index, moment in enumerate(circuit):
            for op in moment.operations:
                # Move Z gates into tracked qubit phases.
                h = _try_get_known_z_half_turns(op)
                if h is not None:
                    q = op.qubits[0]
                    qubit_phase[q] += h / 2
                    deletions.append((moment_index, op))
                    continue

                # Z gate before measurement is a no-op. Drop tracked phase.
                if ops.MeasurementGate.is_measurement(op):
                    for q in op.qubits:
                        qubit_phase[q] = 0

                # If there's no tracked phase, we can move on.
                phases = [qubit_phase[q] for q in op.qubits]
                if all(decompositions.is_negligible_turn(p, self.tolerance)
                       for p in phases):
                    continue

                # Try to move the tracked phasing over the operation.
                phased_op = op
                for i, p in enumerate(phases):
                    if not decompositions.is_negligible_turn(p, self.tolerance):
                        phased_op = protocols.phase_by(phased_op, -p, i,
                                                       default=None)
                if phased_op is not None:
                    deletions.append((moment_index, op))
                    inline_intos.append((moment_index,
                                     cast(ops.Operation, phased_op)))
                else:
                    dump_tracked_phase(op.qubits, moment_index)

        dump_tracked_phase(qubit_phase.keys(), len(circuit))
        circuit.batch_remove(deletions)
        circuit.batch_insert_into(inline_intos)
        circuit.batch_insert(insertions)


def _try_get_known_z_half_turns(op: ops.Operation) -> Optional[float]:
    if not isinstance(op, ops.GateOperation):
        return None
    if not isinstance(op.gate, ops.ZPowGate):
        return None
    h = op.gate.exponent
    if isinstance(h, value.Symbol):
        return None
    return h
