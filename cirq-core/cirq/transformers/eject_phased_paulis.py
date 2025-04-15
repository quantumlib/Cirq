# Copyright 2022 The Cirq Developers
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

"""Transformer pass that pushes 180° rotations around axes in the XY plane later in the circuit."""

from typing import cast, Dict, Iterable, Iterator, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import sympy

from cirq import circuits, ops, protocols, value
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions

if TYPE_CHECKING:
    import cirq


@transformer_api.transformer(add_deep_support=True)
def eject_phased_paulis(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    atol: float = 1e-8,
    eject_parameterized: bool = False,
) -> 'cirq.Circuit':
    """Transformer pass to push X, Y, PhasedX & (certain) PhasedXZ gates to the end of the circuit.

    As the gates get pushed, they may absorb Z gates, cancel against other
    X, Y, or PhasedX gates with exponent=1, get merged into measurements (as
    output bit flips), and cause phase kickback operations across CZs (which can
    then be removed by the `cirq.eject_z` transformation).

    `cirq.PhasedXZGate` with `z_exponent=0` (i.e. equivalent to PhasedXPow) or with `x_exponent=0`
    and `axis_phase_exponent=0` (i.e. equivalent to ZPowGate) are also supported.
    To eject `PhasedXZGates` with arbitrary x/z/axis exponents, run
    `cirq.eject_z(cirq.eject_phased_paulis(cirq.eject_z(circuit)))`.

    Args:
        circuit: Input circuit to transform.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        atol: Maximum absolute error tolerance. The optimization is permitted to simply drop
            negligible combinations gates with a threshold determined by this tolerance.
        eject_parameterized: If True, the optimization will attempt to eject parameterized gates
            as well.  This may result in other gates parameterized by symbolic expressions.
    Returns:
          Copy of the transformed input circuit.
    """
    held_w_phases: Dict[ops.Qid, value.TParamVal] = {}
    tags_to_ignore = set(context.tags_to_ignore) if context else set()

    def map_func(op: 'cirq.Operation', _: int) -> 'cirq.OP_TREE':
        # Dump if `op` marked with a no compile tag.
        if set(op.tags) & tags_to_ignore:
            return [_dump_held(op.qubits, held_w_phases, atol), op]

        # Collect, phase, and merge Ws.
        w = _try_get_known_phased_pauli(op, no_symbolic=not eject_parameterized)
        if w is not None:
            return (
                _potential_cross_whole_w(op, atol, held_w_phases)
                if single_qubit_decompositions.is_negligible_turn((w[0] - 1) / 2, atol)
                else _potential_cross_partial_w(op, held_w_phases, atol)
            )

        affected = [q for q in op.qubits if q in held_w_phases]
        if not affected:
            return op

        # Absorb Z rotations.
        t = _try_get_known_z_half_turns(op, no_symbolic=not eject_parameterized)
        if t is not None:
            return _absorb_z_into_w(op, held_w_phases)

        # Dump coherent flips into measurement bit flips.
        if isinstance(op.gate, ops.MeasurementGate):
            return _dump_into_measurement(op, held_w_phases)

        # Cross CZs using kickback.
        if _try_get_known_cz_half_turns(op, no_symbolic=not eject_parameterized) is not None:
            return (
                _single_cross_over_cz(op, affected[0])
                if len(affected) == 1
                else _double_cross_over_cz(op, held_w_phases)
            )

        # Don't know how to handle this situation. Dump the gates.
        return [_dump_held(op.qubits, held_w_phases, atol), op]

    # Map operations and put anything that's still held at the end of the circuit.
    return circuits.Circuit(
        transformer_primitives.map_operations_and_unroll(circuit, map_func),
        _dump_held(held_w_phases.keys(), held_w_phases, atol),
    )


def _absorb_z_into_w(
    op: ops.Operation, held_w_phases: Dict[ops.Qid, value.TParamVal]
) -> 'cirq.OP_TREE':
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
    t = cast(value.TParamVal, _try_get_known_z_half_turns(op))
    q = op.qubits[0]
    held_w_phases[q] += t / 2
    return []


def _dump_held(
    qubits: Iterable[ops.Qid], held_w_phases: Dict[ops.Qid, value.TParamVal], atol: float
) -> Iterator['cirq.OP_TREE']:
    # Note: sorting is to avoid non-determinism in the insertion order.
    for q in sorted(qubits):
        p = held_w_phases.get(q)
        if p is not None:
            gate = _phased_x_or_pauli_gate(exponent=1.0, phase_exponent=p, atol=atol)
            yield gate.on(q)
        held_w_phases.pop(q, None)


def _dump_into_measurement(
    op: ops.Operation, held_w_phases: Dict[ops.Qid, value.TParamVal]
) -> 'cirq.OP_TREE':
    measurement = cast(ops.MeasurementGate, cast(ops.GateOperation, op).gate)
    new_measurement = measurement.with_bits_flipped(
        *[i for i, q in enumerate(op.qubits) if q in held_w_phases]
    ).on(*op.qubits)
    for q in op.qubits:
        held_w_phases.pop(q, None)
    return new_measurement


def _potential_cross_whole_w(
    op: ops.Operation, atol: float, held_w_phases: Dict[ops.Qid, value.TParamVal]
) -> 'cirq.OP_TREE':
    """Grabs or cancels a held W gate against an existing W gate.

    [Where W(a) is shorthand for PhasedX(phase_exponent=a).]

    Uses the following identity:
        ───W(a)───W(b)───
        ≡ ───Z^-a───X───Z^a───Z^-b───X───Z^b───
        ≡ ───Z^-a───Z^-a───Z^b───X───X───Z^b───
        ≡ ───Z^-a───Z^-a───Z^b───Z^b───
        ≡ ───Z^2(b-a)───
    """
    _, phase_exponent = cast(
        Tuple[value.TParamVal, value.TParamVal], _try_get_known_phased_pauli(op)
    )
    q = op.qubits[0]
    a = held_w_phases.get(q, None)
    b = phase_exponent

    if a is None:
        # Collect the gate.
        held_w_phases[q] = b
    else:
        # Cancel the gate.
        del held_w_phases[q]
        t = 2 * (b - a)
        if not single_qubit_decompositions.is_negligible_turn(t / 2, atol):
            return ops.Z(q) ** t
    return []


def _potential_cross_partial_w(
    op: ops.Operation, held_w_phases: Dict[ops.Qid, value.TParamVal], atol: float
) -> 'cirq.OP_TREE':
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
    a = held_w_phases.get(op.qubits[0], None)
    if a is None:
        return op
    exponent, phase_exponent = cast(
        Tuple[value.TParamVal, value.TParamVal], _try_get_known_phased_pauli(op)
    )
    gate = _phased_x_or_pauli_gate(
        exponent=exponent, phase_exponent=2 * a - phase_exponent, atol=atol
    )
    return gate.on(op.qubits[0])


def _single_cross_over_cz(op: ops.Operation, qubit_with_w: 'cirq.Qid') -> 'cirq.OP_TREE':
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
    t = cast(value.TParamVal, _try_get_known_cz_half_turns(op))
    other_qubit = op.qubits[0] if qubit_with_w == op.qubits[1] else op.qubits[1]
    negated_cz = ops.CZ(*op.qubits) ** -t
    kickback = ops.Z(other_qubit) ** t
    return [kickback, negated_cz]


def _double_cross_over_cz(
    op: ops.Operation, held_w_phases: Dict[ops.Qid, value.TParamVal]
) -> 'cirq.OP_TREE':
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
    t = cast(value.TParamVal, _try_get_known_cz_half_turns(op))
    for q in op.qubits:
        held_w_phases[q] += t / 2
    return op


def _try_get_known_cz_half_turns(
    op: ops.Operation, no_symbolic: bool = False
) -> Optional[value.TParamVal]:
    if not isinstance(op.gate, ops.CZPowGate):
        return None
    h = op.gate.exponent
    if no_symbolic and isinstance(h, sympy.Basic):
        return None
    return h


def _try_get_known_phased_pauli(
    op: ops.Operation, no_symbolic: bool = False
) -> Optional[Tuple[value.TParamVal, value.TParamVal]]:
    if no_symbolic and protocols.is_parameterized(op):
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
    elif (
        isinstance(gate, ops.PhasedXZGate)
        and not protocols.is_parameterized(gate.z_exponent)
        and np.isclose(float(gate.z_exponent), 0)
    ):
        e = gate.x_exponent
        p = gate.axis_phase_exponent
    else:
        return None
    return value.canonicalize_half_turns(e), value.canonicalize_half_turns(p)


def _try_get_known_z_half_turns(
    op: ops.Operation, no_symbolic: bool = False
) -> Optional[value.TParamVal]:
    g = op.gate
    if (
        isinstance(g, ops.PhasedXZGate)
        and not protocols.is_parameterized(g.x_exponent)
        and not protocols.is_parameterized(g.axis_phase_exponent)
        and np.isclose(float(g.x_exponent), 0)
        and np.isclose(float(g.axis_phase_exponent), 0)
    ):

        h = g.z_exponent
    elif isinstance(g, ops.ZPowGate):
        h = g.exponent
    else:
        return None
    if no_symbolic and isinstance(h, sympy.Basic):
        return None
    return h


def _phased_x_or_pauli_gate(
    exponent: Union[float, sympy.Expr], phase_exponent: Union[float, sympy.Expr], atol: float
) -> Union['cirq.PhasedXPowGate', 'cirq.XPowGate', 'cirq.YPowGate']:
    """Return PhasedXPowGate or X or Y gate if equivalent within atol in z-axis turns."""
    if not isinstance(phase_exponent, sympy.Expr) or phase_exponent.is_constant():
        half_turns = value.canonicalize_half_turns(float(phase_exponent))
        if abs(half_turns / 2) <= atol:
            return ops.XPowGate(exponent=exponent)
        if abs((half_turns - 0.5) / 2) <= atol:
            return ops.YPowGate(exponent=exponent)
    return ops.PhasedXPowGate(exponent=exponent, phase_exponent=phase_exponent)
