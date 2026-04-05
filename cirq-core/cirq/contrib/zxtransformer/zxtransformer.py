# Copyright 2024 The Cirq Developers
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

import functools
from typing import Dict, List, Callable, Optional, Tuple, Type, Union
from fractions import Fraction

import cirq
from cirq import circuits

import pyzx as zx
from pyzx.circuit import gates as zx_gates
from pyzx.circuit.gates import Measurement as PyzxMeasurement
from pyzx.circuit.gates import Reset as PyzxReset
from pyzx.circuit.gates import ConditionalGate


# Forward mapping: Cirq gate instances to pyzx gate classes (for zero-param gates).
@functools.cache
def _cirq_to_pyzx():
    return {
        cirq.H: zx_gates.HAD,
        cirq.CZ: zx_gates.CZ,
        cirq.CNOT: zx_gates.CNOT,
        cirq.SWAP: zx_gates.SWAP,
        cirq.CCZ: zx_gates.CCZ,
        cirq.CCX: zx_gates.Tofolli,
        cirq.CSWAP: zx_gates.CSWAP,
    }


def _cirq_gate_to_zx_gate(
    cirq_gate: Optional[cirq.Gate], qubits: List[int]
) -> Optional[zx_gates.Gate]:
    """Convert a Cirq gate to a PyZX gate."""
    if cirq_gate is None:
        return None

    if isinstance(cirq_gate, cirq.XPowGate):
        return zx_gates.XPhase(qubits[0], phase=Fraction(cirq_gate.exponent))
    if isinstance(cirq_gate, cirq.YPowGate):
        return zx_gates.YPhase(qubits[0], phase=Fraction(cirq_gate.exponent))
    if isinstance(cirq_gate, cirq.ZPowGate):
        return zx_gates.ZPhase(qubits[0], phase=Fraction(cirq_gate.exponent))
    if isinstance(cirq_gate, cirq.XXPowGate):
        return zx_gates.RXX(qubits[0], qubits[1], phase=Fraction(cirq_gate.exponent))
    if isinstance(cirq_gate, cirq.ZZPowGate):
        return zx_gates.RZZ(qubits[0], qubits[1], phase=Fraction(cirq_gate.exponent))

    if (gate := _cirq_to_pyzx().get(cirq_gate, None)) is not None:
        return gate(*qubits)

    return None


# Reverse mapping: pyzx QASM name to (cirq gate type, num params, fixed exponent).
# Gates with num_params > 0 use the pyzx phase as the Cirq exponent.
# Gates with num_params == 0 use the fixed exponent.
_pyzx_to_cirq: Dict[str, Tuple[Type[cirq.Gate], int, Optional[float]]] = {
    'x': (cirq.XPowGate, 0, 1.0),
    'y': (cirq.YPowGate, 0, 1.0),
    'z': (cirq.ZPowGate, 0, 1.0),
    's': (cirq.ZPowGate, 0, 0.5),
    'sdg': (cirq.ZPowGate, 0, -0.5),
    't': (cirq.ZPowGate, 0, 0.25),
    'tdg': (cirq.ZPowGate, 0, -0.25),
    'sx': (cirq.XPowGate, 0, 0.5),
    'sxdg': (cirq.XPowGate, 0, -0.5),
    'h': (cirq.HPowGate, 0, 1.0),
    'rx': (cirq.XPowGate, 1, None),
    'ry': (cirq.YPowGate, 1, None),
    'rz': (cirq.ZPowGate, 1, None),
    'cx': (cirq.CXPowGate, 0, 1.0),
    'cz': (cirq.CZPowGate, 0, 1.0),
    'swap': (cirq.SwapPowGate, 0, 1.0),
    'rxx': (cirq.XXPowGate, 1, None),
    'rzz': (cirq.ZZPowGate, 1, None),
    'ccx': (cirq.CCXPowGate, 0, 1.0),
    'ccz': (cirq.CCZPowGate, 0, 1.0),
    'cswap': (cirq.CSwapGate, 0, None),
}


def _make_cirq_gate(gate_name: str, phase: Optional[float] = None) -> cirq.Gate:
    """Create a Cirq gate from a pyzx QASM gate name and optional phase.

    :raises ValueError: If the gate name is not in the reverse mapping table.
    """
    if gate_name not in _pyzx_to_cirq:
        raise ValueError(f"Unsupported gate: {gate_name}.")
    gate_type, num_params, fixed_exp = _pyzx_to_cirq[gate_name]
    if num_params > 0 and phase is not None:
        return gate_type(exponent=phase)  # type: ignore[call-arg]
    if fixed_exp is not None:
        return gate_type(exponent=fixed_exp)  # type: ignore[call-arg]
    return gate_type()


def _is_unitary_gate(gate: zx_gates.Gate) -> bool:
    """Check whether a PyZX gate is unitary (not measurement, reset, or conditional)."""
    return not isinstance(gate, (PyzxMeasurement, PyzxReset, ConditionalGate))


def _optimize_unitary(c: zx.Circuit) -> zx.Circuit:
    """Optimise a purely unitary PyZX circuit using full_reduce and extraction."""
    g = c.to_graph()
    zx.simplify.full_reduce(g)
    return zx.extract.extract_circuit(g)


def _optimize(c: zx.Circuit) -> zx.Circuit:
    """Optimise a PyZX circuit, handling hybrid (non-unitary) circuits.

    For purely unitary circuits, uses full_reduce + extract_circuit. For hybrid circuits
    containing measurements, resets, or conditional gates, splits the circuit at non-unitary
    boundaries, optimises each unitary segment independently, and reassembles.
    """
    if all(_is_unitary_gate(g) for g in c.gates):
        return _optimize_unitary(c)

    # Split the circuit into unitary segments and non-unitary gates.
    result = zx.Circuit(c.qubits, bit_amount=c.bits or None)
    current_gates: List[zx_gates.Gate] = []

    def _flush_unitary() -> None:
        if not current_gates:
            return
        segment = zx.Circuit(c.qubits)
        for g in current_gates:
            segment.add_gate(g)
        current_gates.clear()
        optimized = _optimize_unitary(segment)
        for g in optimized.gates:
            result.add_gate(g)

    for gate in c.gates:
        if _is_unitary_gate(gate):
            current_gates.append(gate)
        else:
            _flush_unitary()
            result.add_gate(gate)

    _flush_unitary()
    return result


def _try_convert_conditional(
    op: cirq.ClassicallyControlledOperation, qubit_to_index: Dict[cirq.Qid, int]
) -> Optional[ConditionalGate]:
    """Try to convert a ClassicallyControlledOperation to a PyZX ConditionalGate.

    Returns None if the operation cannot be converted (e.g. multiple conditions,
    multi-qubit gate, or unsupported gate type). Only single-qubit X and Z rotations
    (subclasses of XPowGate or ZPowGate in Cirq) are supported by pyzx's ConditionalGate.
    """
    controls = op.classical_controls
    if len(controls) != 1:
        return None
    cond = next(iter(controls))
    if not isinstance(cond, cirq.KeyCondition):
        return None
    if len(op.qubits) != 1:
        return None

    inner_op = op.without_classical_controls()
    inner_gate = inner_op.gate
    qubit_index = qubit_to_index[op.qubits[0]]

    # Only single-qubit Z or X rotations are supported by pyzx ConditionalGate.
    pyzx_inner: zx_gates.Gate
    if isinstance(inner_gate, cirq.XPowGate):
        pyzx_inner = zx_gates.XPhase(qubit_index, phase=Fraction(inner_gate.exponent))
    elif isinstance(inner_gate, cirq.ZPowGate):
        pyzx_inner = zx_gates.ZPhase(qubit_index, phase=Fraction(inner_gate.exponent))
    else:
        return None

    # KeyCondition checks for truthiness (non-zero), which maps to condition_value=1.
    return ConditionalGate(cond.key.name, 1, pyzx_inner, 1)


def _cirq_to_circuits_and_ops(
    circuit: circuits.AbstractCircuit, qubits: List[cirq.Qid]
) -> Tuple[List[Union[zx.Circuit, cirq.Operation]], List[str], List[bool]]:
    """Convert an AbstractCircuit to a list of PyZX Circuits and cirq.Operations.

    As much of the AbstractCircuit is converted to PyZX as possible, but some gates are
    not supported by PyZX and are left as cirq.Operations.

    :param circuit: The AbstractCircuit to convert.
    :param qubits: The list of qubits in the circuit.
    :return: A tuple of (circuits and ops, measurement keys indexed by result_bit,
             measurement invert flags indexed by result_bit).
    """
    circuits_and_ops: List[Union[zx.Circuit, cirq.Operation]] = []
    qubit_to_index = {qubit: index for index, qubit in enumerate(qubits)}

    # Pre-scan to count measured bits for pyzx bit_amount.
    num_measurements = sum(
        len(op.qubits) for moment in circuit for op in moment if cirq.is_measurement(op)
    )

    measurement_keys: List[str] = []
    measurement_invert: List[bool] = []
    current_circuit: Optional[zx.Circuit] = None

    def _ensure_circuit() -> zx.Circuit:
        nonlocal current_circuit
        if current_circuit is None:
            current_circuit = zx.Circuit(
                len(qubits), bit_amount=num_measurements if num_measurements else None
            )
        return current_circuit

    def _flush_circuit() -> None:
        nonlocal current_circuit
        if current_circuit is not None:
            circuits_and_ops.append(current_circuit)
            current_circuit = None

    for moment in circuit:
        for op in moment:
            # Handle measurements.
            if isinstance(op.gate, cirq.MeasurementGate):
                key = cirq.measurement_key_name(op)
                invert_mask = op.gate.invert_mask or ()
                for i, qubit in enumerate(op.qubits):
                    bit_index = len(measurement_keys)
                    measurement_keys.append(key)
                    measurement_invert.append(invert_mask[i] if i < len(invert_mask) else False)
                    _ensure_circuit().add_gate(
                        PyzxMeasurement(qubit_to_index[qubit], result_bit=bit_index)
                    )
                continue

            # Handle resets.
            if isinstance(op.gate, cirq.ResetChannel):
                _ensure_circuit().add_gate(PyzxReset(qubit_to_index[op.qubits[0]]))
                continue

            # Handle classically controlled operations.
            if isinstance(op, cirq.ClassicallyControlledOperation):
                converted = _try_convert_conditional(op, qubit_to_index)
                if converted is not None:
                    _ensure_circuit().add_gate(converted)
                    continue
                _flush_circuit()
                circuits_and_ops.append(op)
                continue

            # Try to convert to a pyzx gate.
            gate_qubits = [qubit_to_index[qarg] for qarg in op.qubits]
            gate = _cirq_gate_to_zx_gate(op.gate, gate_qubits)
            if gate is None:
                _flush_circuit()
                circuits_and_ops.append(op)
                continue

            _ensure_circuit().add_gate(gate)

    _flush_circuit()
    return circuits_and_ops, measurement_keys, measurement_invert


def _recover_circuit(
    circuits_and_ops: List[Union[zx.Circuit, cirq.Operation]],
    qubits: List[cirq.Qid],
    measurement_keys: List[str],
    measurement_invert: List[bool],
) -> circuits.Circuit:
    """Recovers a cirq.Circuit from a list of PyZX Circuits and cirq.Operations.

    :param circuits_and_ops: The list of (optimized) PyZX Circuits and cirq.Operations
                             from which to recover the cirq.Circuit.
    :param qubits: The list of qubits in the circuit.
    :param measurement_keys: The list of measurement key names, indexed by result_bit.
    :param measurement_invert: The list of invert flags, indexed by result_bit.
    :return: An optimized version of the original input circuit.
    :raises ValueError: If an unsupported gate has been encountered.
    """
    cirq_circuit = circuits.Circuit()

    # Buffer for grouping multi-qubit measurements into single operations.
    pending_key: Optional[str] = None
    pending_qubits: List[cirq.Qid] = []
    pending_inverts: List[bool] = []

    def _flush_measurement() -> None:
        nonlocal pending_key, pending_qubits, pending_inverts
        if pending_key is not None:
            invert_mask = tuple(pending_inverts)
            # Trim trailing False to match Cirq's convention.
            while invert_mask and not invert_mask[-1]:
                invert_mask = invert_mask[:-1]
            cirq_circuit.append(
                cirq.measure(*pending_qubits, key=pending_key, invert_mask=invert_mask)
            )
            pending_key = None
            pending_qubits = []
            pending_inverts = []

    for circuit_or_op in circuits_and_ops:
        _flush_measurement()
        if isinstance(circuit_or_op, cirq.Operation):
            cirq_circuit.append(circuit_or_op)
            continue
        for gate in circuit_or_op.gates:
            # Handle measurements (buffered for multi-qubit grouping).
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
                if pending_key == key:
                    pending_qubits.append(qubit)
                    pending_inverts.append(invert)
                else:
                    _flush_measurement()
                    pending_key = key
                    pending_qubits = [qubit]
                    pending_inverts = [invert]
                continue

            _flush_measurement()

            # Handle resets.
            if isinstance(gate, PyzxReset):
                cirq_circuit.append(cirq.ResetChannel()(qubits[gate.target]))
                continue

            # Handle conditional gates.
            if isinstance(gate, ConditionalGate):
                inner = gate.inner_gate
                inner_name = (
                    inner.qasm_name
                    if not (hasattr(inner, 'adjoint') and inner.adjoint)
                    else inner.qasm_name_adjoint
                )
                phase = float(inner.phase) if hasattr(inner, 'phase') else None
                cirq_gate = _make_cirq_gate(inner_name, phase)
                qubit = qubits[getattr(inner, 'target')]
                key_name = gate.condition_register
                cirq_circuit.append(cirq_gate(qubit).with_classical_controls(key_name))
                continue

            # Handle regular gates.
            gate_name = (
                gate.qasm_name
                if not (hasattr(gate, 'adjoint') and gate.adjoint)
                else gate.qasm_name_adjoint
            )
            qargs: List[cirq.Qid] = []
            for attr in ['ctrl1', 'ctrl2', 'control', 'target']:
                if hasattr(gate, attr):
                    qargs.append(qubits[getattr(gate, attr)])
            phase = float(gate.phase) if hasattr(gate, 'phase') else None
            cirq_gate = _make_cirq_gate(gate_name, phase)
            cirq_circuit.append(cirq_gate(*qargs))

    _flush_measurement()
    return cirq_circuit


@cirq.transformer
def zx_transformer(
    circuit: circuits.AbstractCircuit,
    context: Optional[cirq.TransformerContext] = None,
    optimizer: Callable[[zx.Circuit], zx.Circuit] = _optimize,
) -> circuits.Circuit:
    """Perform circuit optimization using pyzx.

    Args:
        circuit: 'cirq.Circuit' input circuit to transform.
        context: `cirq.TransformerContext` storing common configurable
                 options for transformers.
        optimizer: The optimization routine to execute. Defaults to
                   `pyzx.simplify.full_reduce`, splitting at non-unitary boundaries
                   (measurements, resets, conditional gates) when present.

    Returns:
        The modified circuit after optimization.
    """
    qubits: List[cirq.Qid] = [*circuit.all_qubits()]

    circuits_and_ops, measurement_keys, measurement_invert = _cirq_to_circuits_and_ops(
        circuit, qubits
    )
    if not circuits_and_ops:
        copied_circuit = circuit.unfreeze(copy=True)
        return copied_circuit

    circuits_and_ops = [optimizer(c) if isinstance(c, zx.Circuit) else c for c in circuits_and_ops]

    return _recover_circuit(circuits_and_ops, qubits, measurement_keys, measurement_invert)
