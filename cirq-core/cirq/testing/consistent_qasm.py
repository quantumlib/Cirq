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

import warnings
from typing import Any, List, Sequence, Optional

import numpy as np

from cirq import devices, linalg, ops, protocols
from cirq.testing import lin_alg_utils


def assert_qasm_is_consistent_with_unitary(val: Any):
    """Uses `val._unitary_` to check `val._qasm_`'s behavior."""

    # Only test if qiskit is installed.
    try:
        import qiskit
    except ImportError:  # pragma: no cover
        warnings.warn(
            "Skipped assert_qasm_is_consistent_with_unitary because "
            "qiskit isn't installed to verify against."
        )
        return

    unitary = protocols.unitary(val, None)
    if unitary is None:
        # Vacuous consistency.
        return

    if isinstance(val, ops.Operation):
        qubits: Sequence[ops.Qid] = val.qubits
        op = val
        gate = val.gate
    elif isinstance(val, ops.Gate):
        qid_shape = protocols.qid_shape(val)
        remaining_shape = list(qid_shape)
        controls = getattr(val, 'control_qubits', None)
        if controls is not None:
            for i, q in zip(reversed(range(len(controls))), reversed(controls)):
                if q is not None:
                    remaining_shape.pop(i)
        qubits = devices.LineQid.for_qid_shape(remaining_shape)
        op = val.on(*qubits)
        gate = val
    else:
        raise NotImplementedError(f"Don't know how to test {val!r}")

    if isinstance(gate, ops.GlobalPhaseGate):
        # OpenQASM 2.0 does not support global phase gates.
        return

    args = protocols.QasmArgs(qubit_id_map={q: f'q[{i}]' for i, q in enumerate(qubits)})
    qasm = protocols.qasm(op, args=args, default=None)
    if qasm is None:
        return

    num_qubits = len(qubits)
    header = f"""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
"""
    qasm = header + qasm

    qasm_unitary = None
    try:
        qc = qiskit.QuantumCircuit.from_qasm_str(qasm)
        qc.remove_final_measurements()
        qasm_unitary = qiskit.quantum_info.Operator(qc).data
        qasm_unitary = _reorder_indices_of_matrix(qasm_unitary, list(reversed(range(num_qubits))))

        lin_alg_utils.assert_allclose_up_to_global_phase(
            qasm_unitary, unitary, rtol=1e-8, atol=1e-8
        )
    except Exception as ex:
        p_unitary: Optional[np.ndarray]
        p_qasm_unitary: Optional[np.ndarray]
        if qasm_unitary is not None:
            p_unitary, p_qasm_unitary = linalg.match_global_phase(unitary, qasm_unitary)
        else:
            p_unitary = None
            p_qasm_unitary = None
        raise AssertionError(
            'QASM not consistent with cirq.unitary(op) up to global phase.\n\n'
            f'op:\n{_indent(repr(op))}\n\n'
            f'cirq.unitary(op):\n{_indent(repr(unitary))}\n\n'
            f'Generated QASM:\n\n{_indent(qasm)}\n\n'
            f'Unitary of generated QASM:\n{_indent(repr(qasm_unitary))}\n\n'
            f'Phased matched cirq.unitary(op):\n{_indent(repr(p_unitary))}\n\n'
            f'Phased matched unitary of generated QASM:\n{_indent(repr(p_qasm_unitary))}\n\n'
            f'Underlying error:\n{_indent(str(ex))}'
        )


def assert_qiskit_parsed_qasm_consistent_with_unitary(qasm, unitary):  # pragma: no cover
    try:
        # We don't want to require qiskit as a dependency but
        # if Qiskit is installed, test QASM output against it.
        import qiskit
    except ImportError:
        return

    num_qubits = int(np.log2(len(unitary)))
    qc = qiskit.QuantumCircuit.from_qasm_str(qasm)
    qc.remove_final_measurements()  # no measurements allowed
    qiskit_unitary = qiskit.quantum_info.Operator(qc).data
    qiskit_unitary = _reorder_indices_of_matrix(qiskit_unitary, list(reversed(range(num_qubits))))

    lin_alg_utils.assert_allclose_up_to_global_phase(unitary, qiskit_unitary, rtol=1e-8, atol=1e-8)


def _indent(*content: str) -> str:
    return '    ' + '\n'.join(content).replace('\n', '\n    ')


def _reorder_indices_of_matrix(matrix: np.ndarray, new_order: List[int]):
    num_qubits = matrix.shape[0].bit_length() - 1
    matrix = np.reshape(matrix, (2,) * 2 * num_qubits)
    all_indices = range(2 * num_qubits)
    new_input_indices = new_order
    new_output_indices = [i + num_qubits for i in new_input_indices]
    matrix = np.moveaxis(matrix, all_indices, new_input_indices + new_output_indices)
    matrix = np.reshape(matrix, (2**num_qubits, 2**num_qubits))
    return matrix
