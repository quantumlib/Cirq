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

"""ZX-calculus circuit optimization via PyZX.

This module converts supported Cirq circuits to PyZX, applies ZX rewrite
optimization (``full_reduce`` by default), and converts the result back to
Cirq. Unsupported gates are preserved as opaque Cirq operations.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from fractions import Fraction

import cirq
from cirq import circuits, transformers

import pyzx as zx
from pyzx.circuit import gates as zx_gates
from pyzx.circuit.gates import ConditionalGate
from pyzx.circuit.gates import Measurement as PyzxMeasurement
from pyzx.circuit.gates import Reset as PyzxReset


def _to_fraction(exponent: cirq.TParamVal) -> Fraction:
    """Convert a Cirq rotation exponent into an exact PyZX ``Fraction`` phase."""
    if isinstance(exponent, Fraction):
        return exponent
    return Fraction(exponent).limit_denominator()  # type: ignore[arg-type]


@functools.cache
def _cirq_to_pyzx() -> dict[cirq.Gate, type[zx_gates.Gate]]:
    return {
        cirq.H: zx_gates.HAD,
        cirq.CZ: zx_gates.CZ,
        cirq.CNOT: zx_gates.CNOT,
        cirq.SWAP: zx_gates.SWAP,
        cirq.CCZ: zx_gates.CCZ,
        cirq.CCX: zx_gates.Tofolli,
        cirq.CSWAP: zx_gates.CSWAP,
    }


def cirq_gate_to_zx_gate(
    cirq_gate: cirq.Gate | None, qubits: list[int]
) -> zx_gates.Gate | None:
    """Convert a Cirq gate to a PyZX gate, if supported."""
    if cirq_gate is None:
        return None

    if isinstance(cirq_gate, cirq.XPowGate):
        return zx_gates.XPhase(qubits[0], phase=_to_fraction(cirq_gate.exponent))
    if isinstance(cirq_gate, cirq.YPowGate):
        return zx_gates.YPhase(qubits[0], phase=_to_fraction(cirq_gate.exponent))
    if isinstance(cirq_gate, cirq.ZPowGate):
        return zx_gates.ZPhase(qubits[0], phase=_to_fraction(cirq_gate.exponent))
    if isinstance(cirq_gate, cirq.XXPowGate):
        return zx_gates.RXX(
            qubits[0], qubits[1], phase=_to_fraction(cirq_gate.exponent)
        )
    if isinstance(cirq_gate, cirq.ZZPowGate):
        return zx_gates.RZZ(
            qubits[0], qubits[1], phase=_to_fraction(cirq_gate.exponent)
        )

    gate_type = _cirq_to_pyzx().get(cirq_gate)
    if gate_type is not None:
        return gate_type(*qubits)
    return None


_pyzx_to_cirq: dict[str, tuple[type[cirq.Gate], int, float | None]] = {
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


def _make_cirq_gate(gate_name: str, phase: float | None = None) -> cirq.Gate:
    if gate_name not in _pyzx_to_cirq:
        raise ValueError(f'Unsupported gate: {gate_name}.')
    gate_type, num_params, fixed_exp = _pyzx_to_cirq[gate_name]
    if num_params > 0 and phase is not None:
        return gate_type(exponent=phase)  # type: ignore[call-arg,misc]
    if fixed_exp is not None:
        return gate_type(exponent=fixed_exp)  # type: ignore[call-arg,misc]
    return gate_type()  # type: ignore[call-arg,misc]


def _is_unitary_gate(gate: zx_gates.Gate) -> bool:
    return not isinstance(gate, (PyzxMeasurement, PyzxReset, ConditionalGate))


def _optimize_unitary(circuit: zx.Circuit) -> zx.Circuit:
    graph = circuit.to_graph()
    zx.simplify.full_reduce(graph)
    return zx.extract.extract_circuit(graph)


def _optimize(circuit: zx.Circuit) -> zx.Circuit:
    """Run PyZX simplification, splitting at non-unitary boundaries when needed."""
    if all(_is_unitary_gate(gate) for gate in circuit.gates):
        return _optimize_unitary(circuit)

    result = zx.Circuit(circuit.qubits, bit_amount=circuit.bits or None)
    current_gates: list[zx_gates.Gate] = []

    def flush_unitary() -> None:
        if not current_gates:
            return
        segment = zx.Circuit(circuit.qubits)
        for gate in current_gates:
            segment.add_gate(gate)
        current_gates.clear()
        optimized = _optimize_unitary(segment)
        for gate in optimized.gates:
            result.add_gate(gate)

    for gate in circuit.gates:
        if _is_unitary_gate(gate):
            current_gates.append(gate)
        else:
            flush_unitary()
            result.add_gate(gate)

    flush_unitary()
    return result


def _try_convert_conditional(
    op: cirq.ClassicallyControlledOperation, qubit_to_index: dict[cirq.Qid, int]
) -> ConditionalGate | None:
    """Map a single-qubit classically controlled X/Z rotation to PyZX, if possible."""
    controls = op.classical_controls
    if len(controls) != 1:
        return None
    cond = next(iter(controls))
    if not isinstance(cond, cirq.KeyCondition):
        return None
    if len(op.qubits) != 1:
        return None

    inner_gate = op.without_classical_controls().gate
    qubit_index = qubit_to_index[op.qubits[0]]

    if isinstance(inner_gate, cirq.XPowGate):
        pyzx_inner: zx_gates.Gate = zx_gates.XPhase(
            qubit_index, phase=_to_fraction(inner_gate.exponent)
        )
    elif isinstance(inner_gate, cirq.ZPowGate):
        pyzx_inner = zx_gates.ZPhase(
            qubit_index, phase=_to_fraction(inner_gate.exponent)
        )
    else:
        return None

    return ConditionalGate(cond.key.name, 1, pyzx_inner, 1)


class ZXTransformer:
    """Optimize Cirq circuits using PyZX ZX-calculus simplification.

    The transformer splits the input into alternating PyZX circuit segments and
    opaque Cirq operations, optimizes each PyZX segment, then rebuilds a Cirq
    circuit while preserving measurement keys and classical controls.
    """

    def __init__(
        self, optimize: Callable[[zx.Circuit], zx.Circuit] | None = None
    ) -> None:
        """Args:
        optimize: Optional PyZX optimizer. Defaults to ``_optimize``.
        """
        self.qubits: list[cirq.Qid] = []
        self.qubit_to_index: dict[cirq.Qid, int] = {}
        self.measurement_keys: list[str] = []
        self._measurement_invert: list[bool] = []
        self.optimize = optimize or _optimize

    def _cirq_to_circuits_and_ops(
        self, circuit: circuits.AbstractCircuit
    ) -> list[zx.Circuit | cirq.Operation]:
        circuits_and_ops: list[zx.Circuit | cirq.Operation] = []
        self.qubits = [*circuit.all_qubits()]
        self.qubit_to_index = {qubit: index for index, qubit in enumerate(self.qubits)}
        self.measurement_keys = []
        self._measurement_invert = []

        num_measurements = sum(
            len(op.qubits) for moment in circuit for op in moment if cirq.is_measurement(op)
        )

        current_circuit: zx.Circuit | None = None

        def ensure_circuit() -> zx.Circuit:
            nonlocal current_circuit
            if current_circuit is None:
                current_circuit = zx.Circuit(
                    len(self.qubits),
                    bit_amount=num_measurements if num_measurements else None,
                )
            return current_circuit

        def flush_circuit() -> None:
            nonlocal current_circuit
            if current_circuit is not None:
                circuits_and_ops.append(current_circuit)
                current_circuit = None

        for moment in circuit:
            for op in moment:
                if isinstance(op.gate, cirq.MeasurementGate):
                    key = cirq.measurement_key_name(op)
                    invert_mask = op.gate.invert_mask or ()
                    for i, qubit in enumerate(op.qubits):
                        bit_index = len(self.measurement_keys)
                        self.measurement_keys.append(key)
                        self._measurement_invert.append(
                            invert_mask[i] if i < len(invert_mask) else False
                        )
                        ensure_circuit().add_gate(
                            PyzxMeasurement(self.qubit_to_index[qubit], result_bit=bit_index)
                        )
                    continue

                if isinstance(op.gate, cirq.ResetChannel):
                    ensure_circuit().add_gate(PyzxReset(self.qubit_to_index[op.qubits[0]]))
                    continue

                if isinstance(op, cirq.ClassicallyControlledOperation):
                    converted = _try_convert_conditional(op, self.qubit_to_index)
                    if converted is not None:
                        ensure_circuit().add_gate(converted)
                        continue
                    flush_circuit()
                    circuits_and_ops.append(op)
                    continue

                gate_qubits = [self.qubit_to_index[qarg] for qarg in op.qubits]
                gate = cirq_gate_to_zx_gate(op.gate, gate_qubits)
                if gate is None:
                    flush_circuit()
                    circuits_and_ops.append(op)
                    continue

                ensure_circuit().add_gate(gate)

        flush_circuit()
        return circuits_and_ops

    def _recover_circuit(
        self, circuits_and_ops: list[zx.Circuit | cirq.Operation]
    ) -> circuits.Circuit:
        cirq_circuit = circuits.Circuit()

        pending_key: str | None = None
        pending_qubits: list[cirq.Qid] = []
        pending_inverts: list[bool] = []

        def flush_measurement() -> None:
            nonlocal pending_key, pending_qubits, pending_inverts
            if pending_key is not None:
                invert_mask = tuple(pending_inverts)
                while invert_mask and not invert_mask[-1]:
                    invert_mask = invert_mask[:-1]
                cirq_circuit.append(
                    cirq.measure(
                        *pending_qubits, key=pending_key, invert_mask=invert_mask
                    )
                )
                pending_key = None
                pending_qubits = []
                pending_inverts = []

        for circuit_or_op in circuits_and_ops:
            flush_measurement()
            if isinstance(circuit_or_op, cirq.Operation):
                cirq_circuit.append(circuit_or_op)
                continue

            for gate in circuit_or_op.gates:
                if isinstance(gate, PyzxMeasurement):
                    if gate.result_bit is None:
                        raise ValueError('Invalid measurement data: missing result_bit.')
                    if gate.result_bit < 0 or gate.result_bit >= len(self.measurement_keys):
                        raise ValueError(
                            f'Invalid measurement data: result_bit {gate.result_bit} '
                            'is out of range.'
                        )
                    key = self.measurement_keys[gate.result_bit]
                    qubit = self.qubits[gate.target]
                    invert = self._measurement_invert[gate.result_bit]
                    if pending_key == key:
                        pending_qubits.append(qubit)
                        pending_inverts.append(invert)
                    else:
                        flush_measurement()
                        pending_key = key
                        pending_qubits = [qubit]
                        pending_inverts = [invert]
                    continue

                flush_measurement()

                if isinstance(gate, PyzxReset):
                    cirq_circuit.append(cirq.ResetChannel()(self.qubits[gate.target]))
                    continue

                if isinstance(gate, ConditionalGate):
                    inner = gate.inner_gate
                    inner_name = (
                        inner.qasm_name
                        if not (hasattr(inner, 'adjoint') and inner.adjoint)
                        else inner.qasm_name_adjoint
                    )
                    phase = float(inner.phase) if hasattr(inner, 'phase') else None
                    cirq_gate = _make_cirq_gate(inner_name, phase)
                    qubit = self.qubits[inner.target]
                    cirq_circuit.append(
                        cirq_gate(qubit).with_classical_controls(gate.condition_register)
                    )
                    continue

                gate_name = (
                    gate.qasm_name
                    if not (hasattr(gate, 'adjoint') and gate.adjoint)
                    else gate.qasm_name_adjoint
                )
                qargs: list[cirq.Qid] = []
                for attr in ['ctrl1', 'ctrl2', 'control', 'target']:
                    if hasattr(gate, attr):
                        qargs.append(self.qubits[getattr(gate, attr)])
                phase = float(gate.phase) if hasattr(gate, 'phase') else None
                cirq_gate = _make_cirq_gate(gate_name, phase)
                cirq_circuit.append(cirq_gate(*qargs))

        flush_measurement()
        return cirq_circuit

    def __call__(
        self,
        circuit: circuits.AbstractCircuit,
        *,
        context: transformers.TransformerContext | None = None,
    ) -> circuits.Circuit:
        del context
        circuits_and_ops = self._cirq_to_circuits_and_ops(circuit)
        if not circuits_and_ops:
            return circuit.unfreeze(copy=True)

        optimized = [
            self.optimize(segment) if isinstance(segment, zx.Circuit) else segment
            for segment in circuits_and_ops
        ]
        return self._recover_circuit(optimized)


@transformers.transformer
def zx_transformer(
    circuit: circuits.Circuit,
    *,
    context: transformers.TransformerContext | None = None,
    optimizer: Callable[[zx.Circuit], zx.Circuit] | None = None,
) -> circuits.Circuit:
    """Optimize ``circuit`` with PyZX using the default or a custom optimizer."""
    return ZXTransformer(optimize=optimizer)(circuit, context=context)


@transformers.transformer
def full_reduce(
    circuit: circuits.Circuit, *, context: transformers.TransformerContext | None = None
) -> circuits.Circuit:
    """Run PyZX ``full_reduce`` simplification on a circuit."""
    return ZXTransformer()(circuit, context=context)
