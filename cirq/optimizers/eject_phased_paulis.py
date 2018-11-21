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

"""Pushes 180 degree rotations around axes in the XY plane later in the circuit.
"""

from typing import Optional, cast, TYPE_CHECKING, Iterable, Tuple

from cirq import circuits, ops, value, protocols
from cirq.optimizers import decompositions

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Dict, List


class _OptimizerState:
    def __init__(self):
        # The phases of the W gates currently being pushed along each qubit.
        self.held_w_phases = {}  # type: Dict[ops.QubitId, Optional[float]]

        # Accumulated commands to batch-apply to the circuit later.
        self.deletions = []  # type: List[Tuple[int, ops.Operation]]
        self.inline_intos = []  # type: List[Tuple[int, ops.Operation]]
        self.insertions = []  # type: List[Tuple[int, ops.Operation]]


class EjectPhasedPaulis(circuits.OptimizationPass):
    """Pushes X, Y, and PhasedX gates towards the end of the circuit.

    As the gates get pushed, they may absorb Z gates, cancel against other
    X, Y, or PhasedX gates with exponent=1, get merged into measurements (as
    output bit flips), and cause phase kickback operations across CZs (which can
    then be removed by the EjectZ optimization).
    """

    def __init__(self, tolerance: float = 1e-8) -> None:
        """
        Args:
            tolerance: Maximum absolute error tolerance. The optimization is
                 permitted to simply drop negligible combinations of Z gates,
                 with a threshold determined by this tolerance.
        """
        self.tolerance = tolerance

    def optimize_circuit(self, circuit: circuits.Circuit):
        state = _OptimizerState()

        for moment_index, moment in enumerate(circuit):
            for op in moment.operations:
                affected = [q for q in op.qubits
                            if state.held_w_phases.get(q) is not None]

                # Collect, phase, and merge Ws.
                w = _try_get_known_phased_pauli(op)
                if w is not None:
                    if decompositions.is_negligible_turn(
                            w[0] - 1,
                            self.tolerance):
                        _potential_cross_whole_w(moment_index,
                                                 op,
                                                 self.tolerance,
                                                 state)
                    else:
                        _potential_cross_partial_w(moment_index, op, state)
                    continue

                if not affected:
                    continue

                # Absorb Z rotations.
                t = _try_get_known_z_half_turns(op)
                if t is not None:
                    _absorb_z_into_w(moment_index, op, state)
                    continue

                # Dump coherent flips into measurement bit flips.
                if ops.MeasurementGate.is_measurement(op):
                    _dump_into_measurement(moment_index, op, state)

                # Cross CZs using kickback.
                if _try_get_known_cz_half_turns(op) is not None:
                    if len(affected) == 1:
                        _single_cross_over_cz(moment_index,
                                              op,
                                              affected[0],
                                              state)
                    else:
                        _double_cross_over_cz(op, state)
                    continue

                # Don't know how to handle this situation. Dump the gates.
                _dump_held(op.qubits, moment_index, state)

        # Put anything that's still held at the end of the circuit.
        _dump_held(state.held_w_phases.keys(), len(circuit), state)

        circuit.batch_remove(state.deletions)
        circuit.batch_insert_into(state.inline_intos)
        circuit.batch_insert(state.insertions)


def _absorb_z_into_w(moment_index: int,
                     op: ops.Operation,
                     state: _OptimizerState) -> None:
    """Absorbs a Z^t gate into a W(a) flip.

    [Where W(a) is shorthand for PhasedX(phase_exponent=a).]

    Uses the following identity:
        ───W(a)───Z^t───
        ≡ ───W(a)───────────Z^t/2──────────Z^t/2─── (split Z)
        ≡ ───W(a)───W(a)───Z^-t/2───W(a)───Z^t/2─── (flip Z)
        ≡ ───W(a)───W(a)──────────W(a+t/2)───────── (phase W)
        ≡ ────────────────────────W(a+t/2)───────── (cancel Ws)
        ≡ ───W(a+t/2)───
    """
    t = cast(float, _try_get_known_z_half_turns(op))
    q = op.qubits[0]
    state.held_w_phases[q] = cast(float, state.held_w_phases[q]) + t / 2
    state.deletions.append((moment_index, op))


def _dump_held(qubits: Iterable[ops.QubitId],
               moment_index: int,
               state: _OptimizerState):
    # Note: sorting is to avoid non-determinism in the insertion order.
    for q in sorted(qubits):
        p = state.held_w_phases.get(q)
        if p is not None:
            dump_op = ops.PhasedXPowGate(phase_exponent=p).on(q)
            state.insertions.append((moment_index, dump_op))
        state.held_w_phases[q] = None


def _dump_into_measurement(moment_index: int,
                           op: ops.Operation,
                           state: _OptimizerState) -> None:
    measurement = cast(ops.MeasurementGate, cast(ops.GateOperation, op).gate)
    new_measurement = measurement.with_bits_flipped(
        *[i
          for i, q in enumerate(op.qubits)
          if state.held_w_phases.get(q) is not None]
    ).on(*op.qubits)
    for q in op.qubits:
        state.held_w_phases[q] = None
    state.deletions.append((moment_index, op))
    state.inline_intos.append((moment_index, new_measurement))


def _potential_cross_whole_w(moment_index: int,
                             op: ops.Operation,
                             tolerance: float,
                             state: _OptimizerState) -> None:
    """Grabs or cancels a held W gate against an existing W gate.

    [Where W(a) is shorthand for PhasedX(phase_exponent=a).]

    Uses the following identity:
        ───W(a)───W(b)───
        ≡ ───Z^-a───X───Z^a───Z^-b───X───Z^b───
        ≡ ───Z^-a───Z^-a───Z^b───X───X───Z^b───
        ≡ ───Z^-a───Z^-a───Z^b───Z^b───
        ≡ ───Z^2(b-a)───
    """
    state.deletions.append((moment_index, op))

    _, phase_exponent = cast(Tuple[float, float],
                             _try_get_known_phased_pauli(op))
    q = op.qubits[0]
    a = state.held_w_phases.get(q)
    b = phase_exponent

    if a is None:
        # Collect the gate.
        state.held_w_phases[q] = b
    else:
        # Cancel the gate.
        state.held_w_phases[q] = None
        t = 2*(b - a)
        if not decompositions.is_negligible_turn(t / 2, tolerance):
            leftover_phase = ops.Z(q)**t
            state.inline_intos.append((moment_index, leftover_phase))


def _potential_cross_partial_w(moment_index: int,
                               op: ops.Operation,
                               state: _OptimizerState) -> None:
    """Cross the held W over a partial W gate.

    [Where W(a) is shorthand for PhasedX(phase_exponent=a).]

    Uses the following identity:
        ───W(a)───W(b)^t───
        ≡ ───Z^-a───X───Z^a───W(b)^t────── (expand W(a))
        ≡ ───Z^-a───X───W(b-a)^t───Z^a──── (move Z^a across, phasing axis)
        ≡ ───Z^-a───W(a-b)^t───X───Z^a──── (move X across, negating axis angle)
        ≡ ───W(2a-b)^t───Z^-a───X───Z^a─── (move Z^-a across, phasing axis)
        ≡ ───W(2a-b)^t───W(a)───
    """
    a = state.held_w_phases.get(op.qubits[0])
    if a is None:
        return
    exponent, phase_exponent = cast(Tuple[float, float],
                                    _try_get_known_phased_pauli(op))
    new_op = ops.PhasedXPowGate(
        exponent=exponent,
        phase_exponent=2 * a - phase_exponent).on(op.qubits[0])
    state.deletions.append((moment_index, op))
    state.inline_intos.append((moment_index, new_op))


def _single_cross_over_cz(moment_index: int,
                          op: ops.Operation,
                          qubit_with_w: ops.QubitId,
                          state: _OptimizerState) -> None:
    """Crosses exactly one W flip over a partial CZ.

    [Where W(a) is shorthand for PhasedX(phase_exponent=a).]

    Uses the following identity:

        ──────────@─────
                  │
        ───W(a)───@^t───


        ≡ ───@──────O──────@────────────────────
             |      |      │                      (split into on/off cases)
          ───W(a)───W(a)───@^t──────────────────

        ≡ ───@─────────────@─────────────O──────
             |             │             |        (off doesn't interact with on)
          ───W(a)──────────@^t───────────W(a)───

        ≡ ───────────Z^t───@──────@──────O──────
                           │      |      |        (crossing causes kickback)
          ─────────────────@^-t───W(a)───W(a)───  (X Z^t X Z^-t = exp(pi t) I)

        ≡ ───────────Z^t───@────────────────────
                           │                      (merge on/off cases)
          ─────────────────@^-t───W(a)──────────

        ≡ ───Z^t───@──────────────
                   │
          ─────────@^-t───W(a)────
    """
    t = cast(float, _try_get_known_cz_half_turns(op))
    other_qubit = op.qubits[0] if qubit_with_w == op.qubits[1] else op.qubits[1]
    negated_cz = ops.CZ(*op.qubits)**-t
    kickback = ops.Z(other_qubit)**t

    state.deletions.append((moment_index, op))
    state.inline_intos.append((moment_index, negated_cz))
    state.insertions.append((moment_index, kickback))


def _double_cross_over_cz(op: ops.Operation,
                          state: _OptimizerState) -> None:
    """Crosses two W flips over a partial CZ.

    [Where W(a) is shorthand for PhasedX(phase_exponent=a).]

    Uses the following identity:

        ───W(a)───@─────
                  │
        ───W(b)───@^t───


        ≡ ──────────@────────────W(a)───
                    │                     (single-cross top W over CZ)
          ───W(b)───@^-t─────────Z^t────


        ≡ ──────────@─────Z^-t───W(a)───
                    │                     (single-cross bottom W over CZ)
          ──────────@^t───W(b)───Z^t────


        ≡ ──────────@─────W(a)───Z^t────
                    │                     (flip over Z^-t)
          ──────────@^t───W(b)───Z^t────


        ≡ ──────────@─────W(a+t/2)──────
                    │                     (absorb Zs into Ws)
          ──────────@^t───W(b+t/2)──────

        ≡ ───@─────W(a+t/2)───
             │
          ───@^t───W(b+t/2)───
    """
    t = cast(float, _try_get_known_cz_half_turns(op))
    for q in op.qubits:
        state.held_w_phases[q] = cast(float, state.held_w_phases[q]) + t / 2


def _try_get_known_cz_half_turns(op: ops.Operation) -> Optional[float]:
    if (not isinstance(op, ops.GateOperation) or
            not isinstance(op.gate, ops.CZPowGate)):
        return None
    h = op.gate.exponent
    if isinstance(h, value.Symbol):
        return None
    return h


def _try_get_known_phased_pauli(op: ops.Operation
                                ) -> Optional[Tuple[float, float]]:
    if protocols.is_parameterized(op) or not isinstance(op, ops.GateOperation):
        return None
    gate = op.gate

    if isinstance(gate, ops.PhasedXPowGate):
        e = gate.exponent
        p = gate.phase_exponent
    elif isinstance(gate, ops.YPowGate):
        e = gate.exponent
        p = 0.5
    elif isinstance(gate, ops.XPowGate):
        e = gate.exponent
        p = 0.0
    else:
        return None
    return cast(Tuple[float, float],
                (value.canonicalize_half_turns(e),
                 value.canonicalize_half_turns(p)))


def _try_get_known_z_half_turns(op: ops.Operation) -> Optional[float]:
    if (not isinstance(op, ops.GateOperation) or
            not isinstance(op.gate, ops.ZPowGate)):
        return None
    h = op.gate.exponent
    if isinstance(h, value.Symbol):
        return None
    return h
