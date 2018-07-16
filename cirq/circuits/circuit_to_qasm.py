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

"""An optimization pass that combines adjacent single-qubit rotations."""

from typing import (
    Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple, Union, cast
)
import pathlib

import string
import numpy as np

from cirq import ops, linalg, extension
from cirq.extension import Extensions


class QasmUGate(ops.Gate, ops.QasmConvertableGate, ops.CompositeGate):
    def __init__(self, theta, phi, lmda):
        """A QASM gate representing any single qubit unitary with a series of
        three rotations, Z, Y, and Z.

        The angles are normalized to the range [0, 2) half_turns.

        Args:
            theta: Half turns to rotate about Y.
            phi: Half turns to rotate about Z (applied last).
            lmda: Half turns to rotate about Z (applied first).
        """
        self.theta = theta % 2
        self.phi = phi % 2
        self.lmda = lmda % 2

    @staticmethod
    def from_matrix(mat: np.array):
        pre_phase, rotation, post_phase = (
            linalg.deconstruct_single_qubit_matrix_into_angles(mat))
        return QasmUGate(rotation/np.pi, post_phase/np.pi, pre_phase/np.pi)

    def qasm_output(self,
                    qubits: Tuple[ops.QubitId, ...],
                    args: ops.QasmOutputArgs) -> Optional[str]:
        args.validate_version('2.0')
        return args.format(
                'u3({:half_turns},{:half_turns},{:half_turns}) {};\n',
                self.theta, self.phi, self.lmda, qubits[0])

    def default_decompose(self, qubits: Tuple[ops.QubitId, ...]) -> ops.OP_TREE:
        q0, = qubits
        yield ops.Z(q0) ** self.lmda
        yield ops.Y(q0) ** self.theta
        yield ops.Z(q0) ** self.phi

    def __repr__(self) -> str:
        return 'QasmUGate({}, {}, {})'.format(self.theta, self.phi, self.lmda)


class QasmTwoQubitGate(ops.Gate, ops.CompositeGate):
    def __init__(self,
                 b00: float, b01: float, b02: float,
                 b10: float, b11: float, b12: float,
                 x: float, y: float, z: float,
                 a00: float, a01: float, a02: float,
                 a10: float, a11: float, a12: float):
        """A two qubit gate represented in QASM by the KAK decomposition.

        All angles are in half turns.  Assumes a canonicalized KAK
        decomposition.

        Args:
            b00: Before Z rotation on qubit 0.
            b01: Y rotation on qubit 0.
            b02: After Z rotation on qubit 0.
            b10: Before Z rotation on qubit 1.
            b11: Y rotation on qubit 1.
            b12: After Z rotation on qubit 1.
            x: XX interaction.
            y: YY interaction.
            z: ZZ interaction.
            a00: Before Z rotation on qubit 0.
            a01: Y rotation on qubit 0.
            a02: After Z rotation on qubit 0.
            a10: Before Z rotation on qubit 1.
            a11: Y rotation on qubit 1.
            a12: After Z rotation on qubit 1.
        """
        self.b00, self.b01, self.b02 = b00, b01, b02
        self.b10, self.b11, self.b12 = b10, b11, b12
        self.x, self.y, self.z = x, y, z
        self.a00, self.a01, self.a02 = a00, a01, a02
        self.a10, self.a11, self.a12 = a10, a11, a12

    @staticmethod
    def from_matrix(mat: np.array, tolerance=1e-8):
        _, (a1, a0), (x, y, z), (b1, b0) = linalg.kak_decomposition(
            mat,
            linalg.Tolerance(atol=tolerance))
        a00, a01, a02 = (
            linalg.deconstruct_single_qubit_matrix_into_angles(a0))
        a10, a11, a12 = (
            linalg.deconstruct_single_qubit_matrix_into_angles(a1))
        b00, b01, b02 = (
            linalg.deconstruct_single_qubit_matrix_into_angles(b0))
        b10, b11, b12 = (
            linalg.deconstruct_single_qubit_matrix_into_angles(b1))
        return QasmTwoQubitGate(b00/np.pi, b01/np.pi, b02/np.pi,
                                b10/np.pi, b11/np.pi, b12/np.pi,
                                x, y, z,
                                a00/np.pi, a01/np.pi, a02/np.pi,
                                a10/np.pi, a11/np.pi, a12/np.pi)

    def default_decompose(self, qubits: Tuple[ops.QubitId, ...]) -> ops.OP_TREE:
        q0, q1 = qubits
        a = self.x * -2 / np.pi + 0.5
        b = self.y * -2 / np.pi + 0.5
        c = self.z * -2 / np.pi + 0.5

        yield QasmUGate(self.b01, self.b02, self.b00)(q0)
        yield QasmUGate(self.b11, self.b12, self.b10)(q1)

        yield ops.X(q0)**0.5
        yield ops.CNOT(q0, q1)
        yield ops.X(q0)**a
        yield ops.Y(q1)**b
        yield ops.CNOT(q1, q0)
        yield ops.X(q1)**-0.5
        yield ops.Z(q1)**c
        yield ops.CNOT(q0, q1)

        yield QasmUGate(self.a01, self.a02, self.a00)(q0)
        yield QasmUGate(self.a11, self.a12, self.a10)(q1)


valid_id_first = set(string.ascii_lowercase)
valid_id_chars = set(string.ascii_letters + string.digits + '_')
def is_valid_qasm_id(id_str: str, args: ops.QasmOutputArgs) -> bool:
    """Test if id_str is a valid id in QASM grammar."""
    args.validate_version('2.0')
    if len(id_str) < 1:
        return False
    if id_str[0] not in valid_id_first:
        return False
    if not all(char in valid_id_chars for char in id_str[1:]):
        return False
    return True


def save_qasm_circuit(operations: ops.OP_TREE,
                      qubits: Tuple[ops.QubitId, ...],
                      path: Union[str, bytes, pathlib.Path],
                      args: ops.QasmOutputArgs,
                      ext: extension.Extensions) -> None:
    with open(path, 'w') as f:
        _write_qasm_circuit(operations, qubits, args, ext,
                            lambda s:f.write(s))


def qasm_circuit_str(operations: ops.OP_TREE,
                     qubits: Tuple[ops.QubitId, ...],
                     args: ops.QasmOutputArgs,
                     ext: extension.Extensions) -> str:
    output = []
    _write_qasm_circuit(operations, qubits, args, ext,
                        lambda s:output.append(s))
    return ''.join(output)


def _write_qasm_circuit(operations: ops.OP_TREE,
                        qubits: Tuple[ops.QubitId, ...],
                        args: ops.QasmOutputArgs,
                        ext: extension.Extensions,
                        output_func: Callable[[str], None]) -> None:
    operation_list = tuple(ops.flatten_op_tree(operations))
    args.qubit_id_map = {qubit: 'q[{}]'.format(i)
                         for i, qubit in enumerate(qubits)}

    measurements = [op.gate for op in operation_list
                            if ops.MeasurementGate.is_measurement(op)]
    # Pick an id for the creg that will store each measurement
    meas_ids = []
    meas_key_id_map = {}
    invalid_ids = {}
    meas_i = 0
    for meas in measurements:
        key = meas.key
        if key in meas_key_id_map:
            continue
        name = 'm_{}'.format(key)
        if not is_valid_qasm_id(name, args):
            name = 'm{}'.format(meas_i)
            meas_i += 1
            invalid_ids[name] = key
        meas_ids.append(name)
        meas_key_id_map[key] = name
    args.meas_key_id_map = meas_key_id_map

    args.validate_version('2.0')

    # Start building the output string
    line_gap = [0]
    def output_line_gap(n):
        if n > line_gap[0]:
            output_func('\n' * (n - line_gap[0]))
        line_gap[0] = n
    def reset_line_gap():
        line_gap[0] = 0
    def output(text):
        output_func(text)
        reset_line_gap()

    # Comment header
    if args.header:
        for line in args.header.split('\n'):
            output(('// ' + line).rstrip() + '\n')
        output('\n')

    # Version
    output('OPENQASM 2.0;\n')
    output('include "qelib1.inc";\n')
    output_line_gap(2)

    # Function definitions
    # None yet

    # Register definitions
    output('// Qubits: [{}]\n'.format(', '.join(map(str, qubits))))
    output('qreg q[{}];\n'.format(len(qubits)))
    for meas_id in meas_ids:
        if meas_id in invalid_ids:
            actual_key = invalid_ids[meas_id]
            actual_key = ' '.join(actual_key.split('\n'))
            output('creg {}[1];  // Measurement: {}\n'.format(
                        meas_id, actual_key))
        else:
            output('creg {}[1];\n'.format(meas_id))
    output_line_gap(2)

    # Operations
    def write_operations(op_tree, top=True):
        for op in ops.flatten_op_tree(op_tree):
            qasm_op = ext.try_cast(ops.QasmConvertableOperation, op)
            if qasm_op is not None:
                out = qasm_op.qasm_output(args)
                if out is not None:
                    output(out)
                    continue

            if isinstance(op, ops.GateOperation):
                comment = 'Gate: {!s}'.format(op.gate)
            else:
                comment = 'Operation: {!s}'.format(op)
            comp_op = ext.try_cast(ops.CompositeOperation, op)
            if comp_op is not None:
                if top:
                    output_line_gap(1)
                    output('// {}\n'.format(comment))
                write_operations(comp_op.default_decompose(), top=False)
                if top:
                    output_line_gap(1)
                continue

            matrix_op = ext.try_cast(ops.KnownMatrix, op)
            if matrix_op is not None and len(op.qubits) == 1:
                u_op = QasmUGate.from_matrix(matrix_op.matrix())(*op.qubits)
                if top:
                    output_line_gap(1)
                    output('// {}\n'.format(comment))
                output(u_op.qasm_output(args))
                if top:
                    output_line_gap(1)
                continue

            if matrix_op is not None and len(op.qubits) == 2:
                u_op = QasmTwoQubitGate.from_matrix(matrix_op.matrix()
                                                    )(*op.qubits)
                if top:
                    output_line_gap(1)
                    output('// {}\n'.format(comment))
                write_operations((u_op,), top=False)
                if top:
                    output_line_gap(1)
                continue

            if isinstance(op, ops.GateOperation):
                raise ValueError('Cannot output gate as QASM: {}'.format(
                                    op.gate))
            else:
                raise ValueError('Cannot output operation as QASM: {}'.format(
                                    op))

    write_operations(operation_list)
