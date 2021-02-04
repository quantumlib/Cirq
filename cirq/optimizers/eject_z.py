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

from typing import cast, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import sympy

from cirq import circuits, ops, protocols
from cirq.optimizers import decompositions


def _is_integer(n):
    return np.isclose(n, np.round(n))


def _is_swaplike(op: ops.Operation):
    if isinstance(op.gate, ops.SwapPowGate):
        return op.gate.exponent == 1

    if isinstance(op.gate, ops.ISwapPowGate):
        return _is_integer((op.gate.exponent - 1) / 2)

    if isinstance(op.gate, ops.FSimGate):
        return _is_integer(op.gate.theta / np.pi - 1 / 2)

    return False


class EjectZ:
    """Pushes Z gates towards the end of the circuit.

    As the Z gates get pushed they may absorb other Z gates, get absorbed into
    measurements, cross CZ gates, cross W gates (by phasing them), etc.
    """

    def __init__(self, tolerance: float = 0.0, eject_parameterized: bool = False) -> None:
        """
        Args:
            tolerance: Maximum absolute error tolerance. The optimization is
                 permitted to simply drop negligible combinations of Z gates,
                 with a threshold determined by this tolerance.
            eject_parameterized: If True, the optimization will attempt to eject
                parameterized Z gates as well.  This may result in other gates
                parameterized by symbolic expressions.
        """
        self.tolerance = tolerance
        self.eject_parameterized = eject_parameterized

    def optimize_circuit(self, circuit: circuits.Circuit):
        # Tracks qubit phases (in half turns; multiply by pi to get radians).
        qubit_phase: Dict[ops.Qid, float] = defaultdict(lambda: 0)
        deletions: List[Tuple[int, ops.Operation]] = []
        replacements: List[Tuple[int, ops.Operation, ops.Operation]] = []
        insertions: List[Tuple[int, ops.Operation]] = []
        phased_xz_replacements: Dict[Tuple[int, ops.Qid], int] = {}

        def dump_tracked_phase(qubits: Iterable[ops.Qid], index: int) -> None:
            """Zeroes qubit_phase entries by emitting Z gates."""
            for q in qubits:
                p = qubit_phase[q]
                qubit_phase[q] = 0
                if decompositions.is_negligible_turn(p, self.tolerance):
                    continue
                dumped = False
                moment_index = circuit.prev_moment_operating_on([q], index)
                if moment_index is not None:
                    op = circuit.moments[moment_index][q]
                    if op and isinstance(op.gate, ops.PhasedXZGate):
                        # Attach z-rotation to replacing PhasedXZ gate.
                        idx = phased_xz_replacements[moment_index, q]
                        _, _, repl_op = replacements[idx]
                        gate = cast(ops.PhasedXZGate, repl_op.gate)
                        repl_op = gate.with_z_exponent(p * 2).on(q)
                        replacements[idx] = (moment_index, op, repl_op)
                        dumped = True
                if not dumped:
                    # Add a new Z gate
                    dump_op = ops.Z(q) ** (p * 2)
                    insertions.append((index, dump_op))

        for moment_index, moment in enumerate(circuit):
            for op in moment.operations:
                # Move Z gates into tracked qubit phases.
                h = _try_get_known_z_half_turns(op, self.eject_parameterized)
                if h is not None:
                    q = op.qubits[0]
                    qubit_phase[q] += h / 2
                    deletions.append((moment_index, op))
                    continue

                # Z gate before measurement is a no-op. Drop tracked phase.
                if isinstance(op.gate, ops.MeasurementGate):
                    for q in op.qubits:
                        qubit_phase[q] = 0

                # If there's no tracked phase, we can move on.
                phases = [qubit_phase[q] for q in op.qubits]
                if not isinstance(op.gate, ops.PhasedXZGate) and all(
                    decompositions.is_negligible_turn(p, self.tolerance) for p in phases
                ):
                    continue

                if _is_swaplike(op):
                    a, b = op.qubits
                    qubit_phase[a], qubit_phase[b] = qubit_phase[b], qubit_phase[a]
                    continue

                # Try to move the tracked phasing over the operation.
                phased_op = op
                for i, p in enumerate(phases):
                    if not decompositions.is_negligible_turn(p, self.tolerance):
                        phased_op = protocols.phase_by(phased_op, -p, i, default=None)
                if phased_op is not None:
                    gate = phased_op.gate
                    if isinstance(gate, ops.PhasedXZGate) and (
                        self.eject_parameterized or not protocols.is_parameterized(gate.z_exponent)
                    ):
                        qubit = phased_op.qubits[0]
                        qubit_phase[qubit] += gate.z_exponent / 2
                        phased_op = gate.with_z_exponent(0).on(qubit)
                        repl_idx = len(replacements)
                        phased_xz_replacements[moment_index, qubit] = repl_idx
                    replacements.append((moment_index, op, phased_op))
                else:
                    dump_tracked_phase(op.qubits, moment_index)

        dump_tracked_phase(qubit_phase.keys(), len(circuit))
        circuit.batch_remove(deletions)
        circuit.batch_replace(replacements)
        circuit.batch_insert(insertions)


def _try_get_known_z_half_turns(op: ops.Operation, eject_parameterized: bool) -> Optional[float]:
    if not isinstance(op, ops.GateOperation):
        return None
    if not isinstance(op.gate, ops.ZPowGate):
        return None
    h = op.gate.exponent
    if not eject_parameterized and isinstance(h, sympy.Basic):
        return None
    return h
