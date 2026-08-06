# Copyright 2026 The Cirq Developers
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

"""A custom transformer for Cirq which uses ZX-Calculus for circuit optimization, implemented
using PyZX."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from fractions import Fraction
from typing import TYPE_CHECKING

import pyzx as zx
from pyzx.circuit import gates as zx_gates
from pyzx.circuit.gates import ConditionalGate, Measurement as PyzxMeasurement, Reset as PyzxReset

from cirq import circuits, ops, protocols, value
from cirq.transformers import transformer_api

if TYPE_CHECKING:
    import cirq


# Forward mapping: Cirq gate singletons to PyZX gate classes (for zero-param gates).
_CIRQ_TO_PYZX: dict[cirq.Gate, type[zx_gates.Gate]] = {
    ops.H: zx_gates.HAD,
    ops.CZ: zx_gates.CZ,
    ops.CNOT: zx_gates.CNOT,
    ops.SWAP: zx_gates.SWAP,
    ops.CCZ: zx_gates.CCZ,
    ops.CCX: zx_gates.Tofolli,
    ops.CSWAP: zx_gates.CSWAP,
}


def _cirq_gate_to_zx_gate(cirq_gate: cirq.Gate | None, qubits: list[int]) -> zx_gates.Gate | None:
    """Convert a Cirq gate to a PyZX gate.

    Returns None for parameterized (symbolic) gates so the caller can pass them through
    as opaque operations; PyZX phases must be concrete rationals.
    """
    if cirq_gate is None:
        return None
    if protocols.is_parameterized(cirq_gate):
        return None

    if isinstance(cirq_gate, ops.XPowGate):
        return zx_gates.XPhase(qubits[0], phase=Fraction(cirq_gate.exponent))
    if isinstance(cirq_gate, ops.YPowGate):
        return zx_gates.YPhase(qubits[0], phase=Fraction(cirq_gate.exponent))
    if isinstance(cirq_gate, ops.ZPowGate):
        return zx_gates.ZPhase(qubits[0], phase=Fraction(cirq_gate.exponent))
    if isinstance(cirq_gate, ops.XXPowGate):
        return zx_gates.RXX(qubits[0], qubits[1], phase=Fraction(cirq_gate.exponent))
    if isinstance(cirq_gate, ops.ZZPowGate):
        return zx_gates.RZZ(qubits[0], qubits[1], phase=Fraction(cirq_gate.exponent))

    pyzx_cls = _CIRQ_TO_PYZX.get(cirq_gate)
    if pyzx_cls is not None:
        return pyzx_cls(*qubits)

    return None


# Reverse mapping: PyZX QASM name to (Cirq gate type, num params, fixed exponent).
# Gates with num_params > 0 use the PyZX phase as the Cirq exponent.
# Gates with num_params == 0 use the fixed exponent.
_PYZX_TO_CIRQ: dict[str, tuple[type[cirq.Gate], int, float | None]] = {
    'x': (ops.XPowGate, 0, 1.0),
    'y': (ops.YPowGate, 0, 1.0),
    'z': (ops.ZPowGate, 0, 1.0),
    's': (ops.ZPowGate, 0, 0.5),
    'sdg': (ops.ZPowGate, 0, -0.5),
    't': (ops.ZPowGate, 0, 0.25),
    'tdg': (ops.ZPowGate, 0, -0.25),
    'sx': (ops.XPowGate, 0, 0.5),
    'sxdg': (ops.XPowGate, 0, -0.5),
    'h': (ops.HPowGate, 0, 1.0),
    'rx': (ops.XPowGate, 1, None),
    'ry': (ops.YPowGate, 1, None),
    'rz': (ops.ZPowGate, 1, None),
    'cx': (ops.CXPowGate, 0, 1.0),
    'cz': (ops.CZPowGate, 0, 1.0),
    'swap': (ops.SwapPowGate, 0, 1.0),
    'rxx': (ops.XXPowGate, 1, None),
    'rzz': (ops.ZZPowGate, 1, None),
    'ccx': (ops.CCXPowGate, 0, 1.0),
    'ccz': (ops.CCZPowGate, 0, 1.0),
    'cswap': (ops.CSwapGate, 0, None),
}


def _make_cirq_gate(gate_name: str, phase: float | None = None) -> cirq.Gate:
    """Create a Cirq gate from a PyZX QASM gate name and optional phase.

    Raises:
        ValueError: If the gate name is not in the reverse mapping table.
    """
    if gate_name not in _PYZX_TO_CIRQ:
        raise ValueError(f"Unsupported gate: {gate_name}.")
    gate_type, num_params, fixed_exp = _PYZX_TO_CIRQ[gate_name]
    if num_params > 0 and phase is not None:
        return gate_type(exponent=phase)  # type: ignore[call-arg]
    if fixed_exp is not None:
        return gate_type(exponent=fixed_exp)  # type: ignore[call-arg]
    return gate_type()


def _pyzx_gate_to_cirq_gate(gate: zx_gates.Gate) -> cirq.Gate:
    """Return the Cirq gate corresponding to a PyZX gate (ignoring its targets)."""
    gate_name = gate.qasm_name_adjoint if getattr(gate, 'adjoint', False) else gate.qasm_name
    phase = float(gate.phase) if hasattr(gate, 'phase') else None
    return _make_cirq_gate(gate_name, phase)


def _is_unitary_gate(gate: zx_gates.Gate) -> bool:
    """Check whether a PyZX gate is unitary (not measurement, reset, or conditional)."""
    return not isinstance(gate, (PyzxMeasurement, PyzxReset, ConditionalGate))


def _optimize_unitary(c: zx.Circuit) -> zx.Circuit:
    """Optimise a purely unitary PyZX circuit using `full_reduce` and extraction."""
    g = c.to_graph()
    zx.simplify.full_reduce(g)
    return zx.extract.extract_circuit(g)


def _optimize(c: zx.Circuit) -> zx.Circuit:
    """Optimise a PyZX circuit, handling hybrid (non-unitary) circuits.

    For purely unitary circuits, uses `full_reduce` + `extract_circuit`.  For hybrid circuits
    containing measurements, resets, or conditional gates, splits the circuit at non-unitary
    boundaries, optimises each unitary segment independently, and reassembles.
    """
    if all(_is_unitary_gate(g) for g in c.gates):
        return _optimize_unitary(c)

    result = zx.Circuit(c.qubits, bit_amount=c.bits or None)
    pending: list[zx_gates.Gate] = []

    def _flush_unitary() -> None:
        if not pending:
            return
        segment = zx.Circuit(c.qubits)
        for g in pending:
            segment.add_gate(g)
        pending.clear()
        for g in _optimize_unitary(segment).gates:
            result.add_gate(g)

    for gate in c.gates:
        if _is_unitary_gate(gate):
            pending.append(gate)
        else:
            _flush_unitary()
            result.add_gate(gate)

    _flush_unitary()
    return result


def _try_convert_conditional(
    op: cirq.ClassicallyControlledOperation, qubit_to_index: dict[cirq.Qid, int]
) -> ConditionalGate | None:
    """Try to convert a `ClassicallyControlledOperation` to a PyZX `ConditionalGate`.

    Returns None if the operation cannot be converted (e.g., multiple conditions,
    multi-qubit gate, or unsupported gate type).  Only single-qubit X and Z rotations
    (subclasses of `XPowGate` or `ZPowGate` in Cirq) are supported by PyZX's
    `ConditionalGate`.
    """
    controls = op.classical_controls
    if len(controls) != 1:
        return None
    cond = next(iter(controls))
    if not isinstance(cond, value.KeyCondition):
        return None
    if len(op.qubits) != 1:
        return None

    inner_gate = op.without_classical_controls().gate
    if inner_gate is None or protocols.is_parameterized(inner_gate):
        return None
    qubit_index = qubit_to_index[op.qubits[0]]

    # Only single-qubit Z or X rotations are supported by PyZX `ConditionalGate`.
    pyzx_inner: zx_gates.Gate
    if isinstance(inner_gate, ops.XPowGate):
        pyzx_inner = zx_gates.XPhase(qubit_index, phase=Fraction(inner_gate.exponent))
    elif isinstance(inner_gate, ops.ZPowGate):
        pyzx_inner = zx_gates.ZPhase(qubit_index, phase=Fraction(inner_gate.exponent))
    else:
        return None

    # `KeyCondition` checks for truthiness (non-zero), which maps to `condition_value=1`.
    return ConditionalGate(cond.key.name, 1, pyzx_inner, 1)


def _cirq_to_circuits_and_ops(
    circuit: circuits.AbstractCircuit,
    qubits: list[cirq.Qid],
    tags_to_ignore: frozenset[Hashable] = frozenset(),
) -> tuple[list[zx.Circuit | cirq.Operation], list[str], list[bool], list[int]]:
    """Convert an `AbstractCircuit` to a list of PyZX Circuits and `cirq.Operation`s.

    As much of the `AbstractCircuit` is converted to PyZX as possible, but some gates are
    not supported by PyZX and are left as `cirq.Operation`s.

    Args:
        circuit: The `AbstractCircuit` to convert.
        qubits: The list of qubits in the circuit.
        tags_to_ignore: Ops carrying any of these tags are emitted opaquely so that
            PyZX cannot rewrite or discard them.

    Returns:
        A tuple of (circuits and ops, measurement keys indexed by `result_bit`,
        measurement invert flags indexed by `result_bit`, source op ids indexed by
        `result_bit`).  The op id is a distinct integer per original Cirq measurement
        op, used to keep bits from separate same-key measurements from being merged
        during recovery.
    """
    circuits_and_ops: list[zx.Circuit | cirq.Operation] = []
    qubit_to_index = {qubit: index for index, qubit in enumerate(qubits)}

    # Pre-scan to count measured bits for PyZX `bit_amount`.  Measurements whose op is
    # tagged for pass-through are excluded, since their bits never enter the PyZX circuit.
    num_measurements = sum(
        len(op.qubits)
        for op in circuit.all_operations()
        if protocols.is_measurement(op) and not tags_to_ignore.intersection(op.tags)
    )
    bit_amount = num_measurements or None

    measurement_keys: list[str] = []
    measurement_invert: list[bool] = []
    measurement_op_ids: list[int] = []
    next_op_id = 0
    current_circuit = zx.Circuit(len(qubits), bit_amount=bit_amount)

    def _emit_opaque(op: cirq.Operation) -> None:
        """Flush the current PyZX segment and append an opaque Cirq operation."""
        nonlocal current_circuit
        if current_circuit.gates:
            circuits_and_ops.append(current_circuit)
            current_circuit = zx.Circuit(len(qubits), bit_amount=bit_amount)
        circuits_and_ops.append(op)

    for op in circuit.all_operations():
        if tags_to_ignore.intersection(op.tags):
            _emit_opaque(op)
            continue

        if isinstance(op.gate, ops.MeasurementGate):
            # A non-empty confusion_map models classical readout error that PyZX has no
            # representation for; passing such measurements through the reverse mapping
            # would silently drop the confusion.  Emit them opaquely instead.
            if op.gate.confusion_map:
                _emit_opaque(op)
                continue
            key = protocols.measurement_key_name(op)
            invert_mask = op.gate.invert_mask or ()
            op_id = next_op_id
            next_op_id += 1
            for i, qubit in enumerate(op.qubits):
                bit_index = len(measurement_keys)
                measurement_keys.append(key)
                measurement_invert.append(invert_mask[i] if i < len(invert_mask) else False)
                measurement_op_ids.append(op_id)
                current_circuit.add_gate(
                    PyzxMeasurement(qubit_to_index[qubit], result_bit=bit_index)
                )
            continue

        if isinstance(op.gate, ops.ResetChannel):
            current_circuit.add_gate(PyzxReset(qubit_to_index[op.qubits[0]]))
            continue

        if isinstance(op, ops.ClassicallyControlledOperation):
            converted = _try_convert_conditional(op, qubit_to_index)
            if converted is not None:
                current_circuit.add_gate(converted)
            else:
                _emit_opaque(op)
            continue

        gate_qubits = [qubit_to_index[qarg] for qarg in op.qubits]
        gate = _cirq_gate_to_zx_gate(op.gate, gate_qubits)
        if gate is None:
            _emit_opaque(op)
        else:
            current_circuit.add_gate(gate)

    if current_circuit.gates:
        circuits_and_ops.append(current_circuit)
    return circuits_and_ops, measurement_keys, measurement_invert, measurement_op_ids


def _flush_pending_measurement(
    cirq_circuit: circuits.Circuit,
    pending_key: str | None,
    pending_qubits: list[cirq.Qid],
    pending_inverts: list[bool],
) -> None:
    """Append a buffered multi-qubit measurement to `cirq_circuit`.

    Callers own the buffer state and are responsible for resetting `pending_key`,
    `pending_qubits`, and `pending_inverts` after this returns.
    """
    if pending_key is not None:
        cirq_circuit.append(
            ops.measure(*pending_qubits, key=pending_key, invert_mask=tuple(pending_inverts))
        )


def _recover_circuit(
    circuits_and_ops: list[zx.Circuit | cirq.Operation],
    qubits: list[cirq.Qid],
    measurement_keys: list[str],
    measurement_invert: list[bool],
    measurement_op_ids: list[int],
) -> circuits.Circuit:
    """Recovers a `cirq.Circuit` from a list of PyZX Circuits and `cirq.Operation`s.

    Args:
        circuits_and_ops: The list of (optimized) PyZX Circuits and `cirq.Operation`s
            from which to recover the `cirq.Circuit`.
        qubits: The list of qubits in the circuit.
        measurement_keys: The list of measurement key names, indexed by `result_bit`.
        measurement_invert: The list of invert flags, indexed by `result_bit`.
        measurement_op_ids: The list of source op ids indexed by `result_bit`.  Bits
            from the same original Cirq measurement share an id; only bits sharing
            both key and id are grouped into a single multi-qubit measurement.

    Returns:
        An optimized version of the original input circuit.

    Raises:
        ValueError: If an unsupported gate has been encountered.
    """
    cirq_circuit = circuits.Circuit()

    # Buffer for grouping multi-qubit measurements into single operations.
    pending_key: str | None = None
    pending_op_id: int | None = None
    pending_qubits: list[cirq.Qid] = []
    pending_inverts: list[bool] = []

    for circuit_or_op in circuits_and_ops:
        _flush_pending_measurement(cirq_circuit, pending_key, pending_qubits, pending_inverts)
        pending_key = None
        pending_op_id = None
        pending_qubits = []
        pending_inverts = []
        if isinstance(circuit_or_op, ops.Operation):
            cirq_circuit.append(circuit_or_op)
            continue
        for gate in circuit_or_op.gates:
            # Measurements are buffered so that adjacent bits from the same original Cirq
            # measurement op group into a single multi-qubit `cirq.measure` operation.
            # Bits from distinct measurement ops are kept separate even if the key matches,
            # since Cirq treats repeated keys as accumulating repeated measurements.
            if isinstance(gate, PyzxMeasurement):
                if gate.result_bit is None:
                    raise ValueError("Invalid measurement data: missing result_bit.")
                if gate.result_bit < 0 or gate.result_bit >= len(measurement_keys):
                    raise ValueError(
                        f"Invalid measurement data: result_bit "
                        f"{gate.result_bit} is out of range."
                    )
                key = measurement_keys[gate.result_bit]
                qubit = qubits[gate.target]
                invert = measurement_invert[gate.result_bit]
                op_id = measurement_op_ids[gate.result_bit]
                if pending_key == key and pending_op_id == op_id:
                    pending_qubits.append(qubit)
                    pending_inverts.append(invert)
                else:
                    _flush_pending_measurement(
                        cirq_circuit, pending_key, pending_qubits, pending_inverts
                    )
                    pending_key = key
                    pending_op_id = op_id
                    pending_qubits = [qubit]
                    pending_inverts = [invert]
                continue

            _flush_pending_measurement(cirq_circuit, pending_key, pending_qubits, pending_inverts)
            pending_key = None
            pending_op_id = None
            pending_qubits = []
            pending_inverts = []

            if isinstance(gate, PyzxReset):
                cirq_circuit.append(ops.ResetChannel()(qubits[gate.target]))
                continue

            if isinstance(gate, ConditionalGate):
                inner = gate.inner_gate
                cirq_gate = _pyzx_gate_to_cirq_gate(inner)
                qubit = qubits[getattr(inner, 'target')]
                cirq_circuit.append(
                    cirq_gate(qubit).with_classical_controls(gate.condition_register)
                )
                continue

            cirq_gate = _pyzx_gate_to_cirq_gate(gate)
            qargs = [
                qubits[getattr(gate, attr)]
                for attr in ('ctrl1', 'ctrl2', 'control', 'target')
                if hasattr(gate, attr)
            ]
            cirq_circuit.append(cirq_gate(*qargs))

    _flush_pending_measurement(cirq_circuit, pending_key, pending_qubits, pending_inverts)
    return cirq_circuit


@transformer_api.transformer
def zx_transformer(
    circuit: circuits.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    optimizer: Callable[[zx.Circuit], zx.Circuit] = _optimize,
) -> circuits.Circuit:
    """Perform circuit optimization using PyZX.

    Args:
        circuit: `cirq.Circuit` input circuit to transform.
        context: `cirq.TransformerContext` storing common configurable
                 options for transformers.
        optimizer: The optimization routine to execute.  Defaults to `_optimize`,
                   which applies `pyzx.simplify.full_reduce` to each unitary segment,
                   splitting at non-unitary boundaries (measurements, resets, conditional
                   gates) when present.

    Returns:
        The modified circuit after optimization.
    """
    # Sort so PyZX line indices are stable across processes; `all_qubits()` returns a
    # `frozenset` whose iteration order depends on hash randomization for `NamedQubit`.
    qubits = sorted(circuit.all_qubits())
    tags_to_ignore = frozenset(context.tags_to_ignore) if context is not None else frozenset()

    circuits_and_ops, measurement_keys, measurement_invert, measurement_op_ids = (
        _cirq_to_circuits_and_ops(circuit, qubits, tags_to_ignore)
    )
    if not circuits_and_ops:
        return circuit.unfreeze(copy=True)

    circuits_and_ops = [optimizer(c) if isinstance(c, zx.Circuit) else c for c in circuits_and_ops]

    return _recover_circuit(
        circuits_and_ops, qubits, measurement_keys, measurement_invert, measurement_op_ids
    )
