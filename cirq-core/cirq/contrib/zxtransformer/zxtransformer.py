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
from typing import List, Callable, Optional, Union
from fractions import Fraction

import cirq
from cirq import circuits

import pyzx as zx
from pyzx.circuit import gates as zx_gates


@functools.cache
def _cirq_to_pyzx():
    return {
        cirq.H: zx_gates.HAD,
        cirq.CZ: zx_gates.CZ,
        cirq.CNOT: zx_gates.CNOT,
        cirq.SWAP: zx_gates.SWAP,
        cirq.CCZ: zx_gates.CCZ,
    }


def cirq_gate_to_zx_gate(
    cirq_gate: Optional[cirq.Gate], qubits: List[int]
) -> Optional[zx_gates.Gate]:
    """Convert a Cirq gate to a PyZX gate."""

    if isinstance(cirq_gate, (cirq.Rx, cirq.XPowGate)):
        return zx_gates.XPhase(*qubits, phase=Fraction(cirq_gate.exponent))
    if isinstance(cirq_gate, (cirq.Ry, cirq.YPowGate)):
        return zx_gates.YPhase(*qubits, phase=Fraction(cirq_gate.exponent))
    if isinstance(cirq_gate, (cirq.Rz, cirq.ZPowGate)):
        return zx_gates.ZPhase(*qubits, phase=Fraction(cirq_gate.exponent))

    # TODO: Deal with exponents other than nice ones.
    if (gate := _cirq_to_pyzx().get(cirq_gate, None)) is not None:
        return gate(*qubits)

    return None


cirq_gate_table = {
    'rx': cirq.XPowGate,
    'ry': cirq.YPowGate,
    'rz': cirq.ZPowGate,
    'h': cirq.HPowGate,
    'cx': cirq.CXPowGate,
    'cz': cirq.CZPowGate,
    'swap': cirq.SwapPowGate,
    'ccz': cirq.CCZPowGate,
}


def _cirq_to_circuits_and_ops(
    circuit: circuits.AbstractCircuit, qubits: List[cirq.Qid]
) -> List[Union[zx.Circuit, cirq.Operation]]:
    """Convert an AbstractCircuit to a list of PyZX Circuits and cirq.Operations. As much of the
    AbstractCircuit is converted to PyZX as possible, but some gates are not supported by PyZX and
    are left as cirq.Operations.

    :param circuit: The AbstractCircuit to convert.
    :return: A list of PyZX Circuits and cirq.Operations corresponding to the AbstractCircuit.
    """
    circuits_and_ops: List[Union[zx.Circuit, cirq.Operation]] = []
    qubit_to_index = {qubit: index for index, qubit in enumerate(qubits)}
    current_circuit: Optional[zx.Circuit] = None
    for moment in circuit:
        for op in moment:
            gate_qubits = [qubit_to_index[qarg] for qarg in op.qubits]
            gate = cirq_gate_to_zx_gate(op.gate, gate_qubits)
            if not gate:
                # Encountered an operation not supported by PyZX, so just store it.
                # Flush the current PyZX Circuit first if there is one.
                if current_circuit is not None:
                    circuits_and_ops.append(current_circuit)
                    current_circuit = None
                circuits_and_ops.append(op)
                continue

            if current_circuit is None:
                current_circuit = zx.Circuit(len(qubits))
            current_circuit.add_gate(gate)

    # Flush any remaining PyZX Circuit.
    if current_circuit is not None:
        circuits_and_ops.append(current_circuit)

    return circuits_and_ops


def _recover_circuit(
    circuits_and_ops: List[Union[zx.Circuit, cirq.Operation]], qubits: List[cirq.Qid]
) -> circuits.Circuit:
    """Recovers a cirq.Circuit from a list of PyZX Circuits and cirq.Operations.

    :param circuits_and_ops: The list of (optimized) PyZX Circuits and cirq.Operations from which to
                             recover the cirq.Circuit.
    :return: An optimized version of the original input circuit to ZXTransformer.
    :raises ValueError: If an unsupported gate has been encountered.
    """
    cirq_circuit = circuits.Circuit()
    for circuit_or_op in circuits_and_ops:
        if isinstance(circuit_or_op, cirq.Operation):
            cirq_circuit.append(circuit_or_op)
            continue
        for gate in circuit_or_op.gates:
            gate_name = (
                gate.qasm_name
                if not (hasattr(gate, 'adjoint') and gate.adjoint)
                else gate.qasm_name_adjoint
            )
            gate_type = cirq_gate_table[gate_name]
            if gate_type is None:
                raise ValueError(f"Unsupported gate: {gate_name}.")
            qargs: List[cirq.Qid] = []
            for attr in ['ctrl1', 'ctrl2', 'control', 'target']:
                if hasattr(gate, attr):
                    qargs.append(qubits[getattr(gate, attr)])
            params: List[float] = []
            if hasattr(gate, 'phase'):
                params = [float(gate.phase)]
            elif hasattr(gate, 'phases'):
                params = [float(phase) for phase in gate.phases]
            elif gate_name in ('h', 'cz', 'cx', 'swap', 'ccz'):
                params = [1.0]
            cirq_circuit.append(gate_type(exponent=params[0])(*qargs))
    return cirq_circuit


def _optimize(c: zx.Circuit) -> zx.Circuit:
    g = c.to_graph()
    zx.simplify.full_reduce(g)
    return zx.extract.extract_circuit(g)


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
        optimizer: The optimization routine to execute. Defaults to `pyzx.simplify.full_reduce` if
                   not specified.

    Returns:
        The modified circuit after optimization.
    """
    qubits: List[cirq.Qid] = [*circuit.all_qubits()]

    circuits_and_ops = _cirq_to_circuits_and_ops(circuit, qubits)
    if not circuits_and_ops:
        copied_circuit = circuit.unfreeze(copy=True)
        return copied_circuit

    circuits_and_ops = [
        optimizer(circuit) if isinstance(circuit, zx.Circuit) else circuit
        for circuit in circuits_and_ops
    ]

    return _recover_circuit(circuits_and_ops, qubits)
