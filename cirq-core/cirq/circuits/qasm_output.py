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

"""Utility classes for representing QASM."""

from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import re
import numpy as np

from cirq import ops, linalg, protocols, value

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class QasmUGate(ops.Gate):
    def __init__(self, theta, phi, lmda) -> None:
        """A QASM gate representing any single qubit unitary with a series of
        three rotations, Z, Y, and Z.

        The angles are normalized to the range [0, 2) half_turns.

        Args:
            theta: Half turns to rotate about Y (applied second).
            phi: Half turns to rotate about Z (applied last).
            lmda: Half turns to rotate about Z (applied first).
        """
        self.lmda = lmda % 2
        self.theta = theta % 2
        self.phi = phi % 2

    def _num_qubits_(self) -> int:
        return 1

    @staticmethod
    def from_matrix(mat: np.ndarray) -> 'QasmUGate':
        pre_phase, rotation, post_phase = linalg.deconstruct_single_qubit_matrix_into_angles(mat)
        return QasmUGate(rotation / np.pi, post_phase / np.pi, pre_phase / np.pi)

    def _has_unitary_(self):
        return True

    def _qasm_(self, qubits: Tuple['cirq.Qid', ...], args: 'cirq.QasmArgs') -> str:
        args.validate_version('2.0', '3.0')
        return args.format(
            'u3({0:half_turns},{1:half_turns},{2:half_turns}) {3};\n',
            self.theta,
            self.phi,
            self.lmda,
            qubits[0],
        )

    def __repr__(self) -> str:
        return (
            f'cirq.circuits.qasm_output.QasmUGate('
            f'theta={self.theta!r}, '
            f'phi={self.phi!r}, '
            f'lmda={self.lmda})'
        )

    def _decompose_(self, qubits):
        q = qubits[0]
        return [
            ops.rz(self.lmda * np.pi).on(q),
            ops.ry(self.theta * np.pi).on(q),
            ops.rz(self.phi * np.pi).on(q),
        ]

    def _value_equality_values_(self):
        return self.lmda, self.theta, self.phi

    def _json_dict_(self) -> Dict[str, float]:
        return {'theta': self.theta, 'phi': self.phi, 'lmda': self.lmda}

    @classmethod
    def _from_json_dict_(cls, theta: float, phi: float, lmda: float, **kwargs) -> 'QasmUGate':
        return cls(theta, phi, lmda)


@value.value_equality
class QasmTwoQubitGate(ops.Gate):
    def __init__(self, kak: 'cirq.KakDecomposition') -> None:
        """A two qubit gate represented in QASM by the KAK decomposition.

        All angles are in half turns.  Assumes a canonicalized KAK
        decomposition.

        Args:
            kak: KAK decomposition of the two-qubit gate.
        """
        self.kak = kak

    def _num_qubits_(self) -> int:
        return 2

    def _value_equality_values_(self):
        return self.kak

    @staticmethod
    def from_matrix(mat: np.ndarray, atol=1e-8) -> 'QasmTwoQubitGate':
        """Creates a QasmTwoQubitGate from the given matrix.

        Args:
            mat: The unitary matrix of the two qubit gate.
            atol: Absolute error tolerance when decomposing.

        Returns:
            A QasmTwoQubitGate implementing the matrix.
        """
        kak = linalg.kak_decomposition(mat, atol=atol)
        return QasmTwoQubitGate(kak)

    def _unitary_(self):
        return protocols.unitary(self.kak)

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> Iterator['cirq.OP_TREE']:
        q0, q1 = qubits
        x, y, z = self.kak.interaction_coefficients
        a = x * -2 / np.pi + 0.5
        b = y * -2 / np.pi + 0.5
        c = z * -2 / np.pi + 0.5

        b0, b1 = self.kak.single_qubit_operations_before
        yield QasmUGate.from_matrix(b0).on(q0)
        yield QasmUGate.from_matrix(b1).on(q1)

        yield ops.X(q0) ** 0.5
        yield ops.CNOT(q0, q1)
        yield ops.X(q0) ** a
        yield ops.Y(q1) ** b
        yield ops.CNOT(q1, q0)
        yield ops.X(q1) ** -0.5
        yield ops.Z(q1) ** c
        yield ops.CNOT(q0, q1)

        a0, a1 = self.kak.single_qubit_operations_after
        yield QasmUGate.from_matrix(a0).on(q0)
        yield QasmUGate.from_matrix(a1).on(q1)

    def __repr__(self) -> str:
        return f'cirq.circuits.qasm_output.QasmTwoQubitGate({self.kak!r})'


class QasmOutput:
    """Representation of a circuit in QASM (quantum assembly) format.

    Please note that the QASM importer is in an experimental state and
    currently only supports a subset of the full OpenQASM spec.
    Amongst others, classical control, arbitrary gate definitions,
    and even some of the gates that don't have a one-to-one representation
    in Cirq, are not yet supported.

    QASM output can be saved to a file using the save method.
    """

    valid_id_re = re.compile(r'[a-z][a-zA-Z0-9_]*\Z')

    def __init__(
        self,
        operations: 'cirq.OP_TREE',
        qubits: Tuple['cirq.Qid', ...],
        header: str = '',
        precision: int = 10,
        version: str = '2.0',
    ) -> None:
        """Representation of a circuit in QASM format.

        Args:
            operations: Tree of operations to insert.
            qubits: The qubits used in the operations.
            header: A multi-line string that is placed in a comment at the top
                of the QASM.
            precision: The number of digits after the decimal to show for
                numbers in the QASM code.
            version: The QASM version to target. Objects may return different
                QASM depending on version.
        """
        self.operations = tuple(ops.flatten_to_ops(operations))
        self.qubits = qubits
        self.header = header
        self.measurements = tuple(
            op for op in self.operations if isinstance(op.gate, ops.MeasurementGate)
        )
        meas_key_id_map, meas_comments = self._generate_measurement_ids()
        self.meas_comments = meas_comments
        qubit_id_map = self._generate_qubit_ids()
        self.args = protocols.QasmArgs(
            precision=precision,
            version=version,
            qubit_id_map=qubit_id_map,
            meas_key_id_map=meas_key_id_map,
        )
        self.cregs = self._generate_cregs()

    def _generate_measurement_ids(self) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
        # Pick an id for the creg that will store each measurement
        meas_key_id_map: Dict[str, str] = {}
        meas_comments: Dict[str, Optional[str]] = {}
        meas_i = 0
        for meas in self.measurements:
            key = protocols.measurement_key_name(meas)
            if key in meas_key_id_map:
                continue
            meas_id = f'm_{key}'
            if self.is_valid_qasm_id(meas_id):
                meas_comments[key] = None
            else:
                meas_id = f'm{meas_i}'
                meas_i += 1
                meas_comments[key] = ' '.join(key.split('\n'))
            meas_key_id_map[key] = meas_id
        return meas_key_id_map, meas_comments

    def _generate_qubit_ids(self) -> Dict['cirq.Qid', str]:
        return {qubit: f'q[{i}]' for i, qubit in enumerate(self.qubits)}

    def _generate_cregs(self) -> Dict[str, tuple[int, str]]:
        """Pick an id for the creg that will store each measurement

        This function finds the largest measurement using each key.
        That is, if multiple measurements are made with the same key,
        it will use the key with the most number of qubits.

        Returns: dictionary with key of measurement id and value of (#qubits, comment).
        """
        cregs: Dict[str, tuple[int, str]] = {}
        for meas in self.measurements:
            key = protocols.measurement_key_name(meas)
            meas_id = self.args.meas_key_id_map[key]

            if self.meas_comments[key] is not None:
                comment = f'  // Measurement: {self.meas_comments[key]}'
            else:
                comment = ''

            if meas_id not in cregs or cregs[meas_id][0] < len(meas.qubits):
                cregs[meas_id] = (len(meas.qubits), comment)

        return cregs

    def is_valid_qasm_id(self, id_str: str) -> bool:
        """Test if id_str is a valid id in QASM grammar."""
        return self.valid_id_re.match(id_str) is not None

    def save(self, path: Union[str, bytes, int]) -> None:
        """Write QASM output to a file specified by path."""
        with open(path, 'w') as f:

            def write(s: str) -> None:
                f.write(s)

            self._write_qasm(write)

    def __str__(self) -> str:
        """Return QASM output as a string."""
        output = []
        self._write_qasm(lambda s: output.append(s))
        return ''.join(output)

    def _write_qasm(self, output_func: Callable[[str], None]) -> None:
        self.args.validate_version('2.0', '3.0')

        # Generate nice line spacing
        line_gap = [0]

        def output_line_gap(n):
            line_gap[0] = max(line_gap[0], n)

        def output(text):
            if line_gap[0] > 0:
                output_func('\n' * line_gap[0])
                line_gap[0] = 0
            output_func(text)

        # Comment header
        if self.header:
            for line in self.header.split('\n'):
                output(('// ' + line).rstrip() + '\n')
            output('\n')

        # Version
        output(f'OPENQASM {self.args.version};\n')
        if self.args.version == '2.0':
            output('include "qelib1.inc";\n')
        else:
            output('include "stdgates.inc";\n')

        output_line_gap(2)

        # Function definitions
        # None yet

        # Register definitions
        # Qubit registers

        output(f"// Qubits: [{', '.join(map(str, self.qubits))}]\n")
        if len(self.qubits) > 0:
            if self.args.version == '2.0':
                output(f'qreg q[{len(self.qubits)}];\n')
            else:
                output(f'qubit[{len(self.qubits)}] q;\n')

        # Classical registers
        for meas_id in self.cregs:
            length, comment = self.cregs[meas_id]
            if self.args.version == '2.0':
                output(f'creg {meas_id}[{length}];{comment}\n')
            else:
                output(f'bit[{length}] {meas_id};{comment}\n')

        # In OpenQASM 2.0, the transformation of global phase gates is ignored.
        # Therefore, no newline is created when the operations contained in
        # a circuit consist only of global phase gates.
        if any(not isinstance(op.gate, ops.GlobalPhaseGate) for op in self.operations):
            output_line_gap(2)

        # Operations
        self._write_operations(self.operations, output, output_line_gap)

    def _write_operations(
        self,
        op_tree: 'cirq.OP_TREE',
        output: Callable[[str], None],
        output_line_gap: Callable[[int], None],
    ) -> None:
        def keep(op: 'cirq.Operation') -> bool:
            return protocols.qasm(op, args=self.args, default=None) is not None

        def fallback(op):
            if len(op.qubits) not in [1, 2]:
                return NotImplemented

            mat = protocols.unitary(op, None)
            if mat is None:
                return NotImplemented

            if len(op.qubits) == 1:
                return QasmUGate.from_matrix(mat).on(*op.qubits)
            return QasmTwoQubitGate.from_matrix(mat).on(*op.qubits)

        def on_stuck(bad_op):
            return ValueError(f'Cannot output operation as QASM: {bad_op!r}')

        for main_op in ops.flatten_op_tree(op_tree):
            decomposed = protocols.decompose(
                main_op, keep=keep, fallback_decomposer=fallback, on_stuck_raise=on_stuck
            )

            qasms = [protocols.qasm(op, args=self.args) for op in decomposed]

            should_annotate = decomposed != [main_op] or qasms[0].count('\n') > 1
            if should_annotate:
                output_line_gap(1)
                if isinstance(main_op, ops.GateOperation):
                    x = str(main_op.gate).replace('\n', '\n //')
                    output(f'// Gate: {x!s}\n')
                else:
                    x = str(main_op).replace('\n', '\n //')
                    output(f'// Operation: {x!s}\n')

            for qasm in qasms:
                output(qasm)

            if should_annotate:
                output_line_gap(1)
