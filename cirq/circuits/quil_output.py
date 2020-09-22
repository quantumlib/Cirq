# Copyright 2020 The Cirq Developers
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

from typing import Callable, Dict, Set, Tuple, Union
import numpy as np
import cirq
from cirq import protocols, value, ops


def to_quil_complex_format(num) -> str:
    """A function for outputting a number to a complex string in QUIL format."""
    cnum = complex(str(num))
    return "{0}+{1}i".format(cnum.real, cnum.imag)


@value.value_equality(approximate=True)
class QuilOneQubitGate(ops.SingleQubitGate):
    """A QUIL gate representing any single qubit unitary with a DEFGATE and
        2x2 matrix in QUIL.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Args:
            matrix: The 2x2 unitary matrix for this gate.
        """
        self.matrix = matrix

    def _quil_(self, qubits: Tuple['cirq.Qid', ...],
               formatter: 'cirq.QuilFormatter') -> str:
        return (f'DEFGATE USERGATE:\n    '
                f'{to_quil_complex_format(self.matrix[0, 0])}, '
                f'{to_quil_complex_format(self.matrix[0, 1])}\n    '
                f'{to_quil_complex_format(self.matrix[1, 0])}, '
                f'{to_quil_complex_format(self.matrix[1, 1])}\n'
                f'{formatter.format("USERGATE {0}", qubits[0])}\n')

    def __repr__(self) -> str:
        return (f'cirq.circuits.quil_output.QuilOneQubitGate(matrix=\n'
                f'{self.matrix}\n)')

    def _value_equality_values_(self):
        return self.matrix


@value.value_equality(approximate=True)
class QuilTwoQubitGate(ops.TwoQubitGate):
    """A two qubit gate represented in QUIL with a DEFGATE and it's 4x4
    unitary matrix.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Args:
            matrix: The 4x4 unitary matrix for this gate.
        """
        self.matrix = matrix

    def _value_equality_values_(self):
        return self.matrix

    def _quil_(self, qubits: Tuple['cirq.Qid', ...],
               formatter: 'cirq.QuilFormatter') -> str:
        return (
            f'DEFGATE USERGATE:\n    '
            f'{to_quil_complex_format(self.matrix[0, 0])}, '
            f'{to_quil_complex_format(self.matrix[0, 1])}, '
            f'{to_quil_complex_format(self.matrix[0, 2])}, '
            f'{to_quil_complex_format(self.matrix[0, 3])}\n    '
            f'{to_quil_complex_format(self.matrix[1, 0])}, '
            f'{to_quil_complex_format(self.matrix[1, 1])}, '
            f'{to_quil_complex_format(self.matrix[1, 2])}, '
            f'{to_quil_complex_format(self.matrix[1, 3])}\n    '
            f'{to_quil_complex_format(self.matrix[2, 0])}, '
            f'{to_quil_complex_format(self.matrix[2, 1])}, '
            f'{to_quil_complex_format(self.matrix[2, 2])}, '
            f'{to_quil_complex_format(self.matrix[2, 3])}\n    '
            f'{to_quil_complex_format(self.matrix[3, 0])}, '
            f'{to_quil_complex_format(self.matrix[3, 1])}, '
            f'{to_quil_complex_format(self.matrix[3, 2])}, '
            f'{to_quil_complex_format(self.matrix[3, 3])}\n'
            f'{formatter.format("USERGATE {0} {1}", qubits[0], qubits[1])}\n')

    def __repr__(self) -> str:
        return (f'cirq.circuits.quil_output.QuilTwoQubitGate(matrix=\n'
                f'{self.matrix}\n)')


class QuilOutput:
    """An object for passing operations and qubits then outputting them to
    QUIL format. The string representation returns the QUIL output for the
    circuit.
    """

    def __init__(self, operations: 'cirq.OP_TREE',
                 qubits: Tuple['cirq.Qid', ...]) -> None:
        """
        Args:
            operations: A list or tuple of `cirq.OP_TREE` arguments.
            qubits: The qubits used in the operations.
        """
        self.qubits = qubits
        self.operations = tuple(cirq.ops.flatten_to_ops(operations))
        self.measurements = tuple(op for op in self.operations
                                  if isinstance(op.gate, ops.MeasurementGate))
        self.qubit_id_map = self._generate_qubit_ids()
        self.measurement_id_map = self._generate_measurement_ids()
        self.formatter = protocols.QuilFormatter(
            qubit_id_map=self.qubit_id_map,
            measurement_id_map=self.measurement_id_map)

    def _generate_qubit_ids(self) -> Dict['cirq.Qid', str]:
        return {qubit: str(i) for i, qubit in enumerate(self.qubits)}

    def _generate_measurement_ids(self) -> Dict[str, str]:
        index = 0
        measurement_id_map: Dict[str, str] = {}
        for op in self.operations:
            if isinstance(op.gate, ops.MeasurementGate):
                key = protocols.measurement_key(op)
                if key in measurement_id_map:
                    continue
                measurement_id_map[key] = 'm{}'.format(index)
                index += 1
        return measurement_id_map

    def save_to_file(self, path: Union[str, bytes, int]) -> None:
        """Write QUIL output to a file specified by path."""
        with open(path, 'w') as f:
            f.write(str(self))

    def __str__(self) -> str:
        output = []
        self._write_quil(lambda s: output.append(s))
        return self.rename_defgates(''.join(output))

    def _write_quil(self, output_func: Callable[[str], None]) -> None:
        output_func('# Created using Cirq.\n\n')
        if len(self.measurements) > 0:
            measurements_declared: Set[str] = set()
            for m in self.measurements:
                key = protocols.measurement_key(m)
                if key in measurements_declared:
                    continue
                measurements_declared.add(key)
                output_func('DECLARE {} BIT[{}]\n'.format(
                    self.measurement_id_map[key], len(m.qubits)))
            output_func('\n')

        def keep(op: 'cirq.Operation') -> bool:
            return protocols.quil(op, formatter=self.formatter) is not None

        def fallback(op):
            if len(op.qubits) not in [1, 2]:
                return NotImplemented

            mat = protocols.unitary(op, None)
            if mat is None:
                return NotImplemented

            # Following code is a safety measure
            # Could not find a gate that doesn't decompose into a gate
            # with a _quil_ implementation
            # coverage: ignore
            if len(op.qubits) == 1:
                return QuilOneQubitGate(mat).on(*op.qubits)
            return QuilTwoQubitGate(mat).on(*op.qubits)

        def on_stuck(bad_op):
            return ValueError(
                'Cannot output operation as QUIL: {!r}'.format(bad_op))

        for main_op in self.operations:
            decomposed = protocols.decompose(main_op,
                                             keep=keep,
                                             fallback_decomposer=fallback,
                                             on_stuck_raise=on_stuck)

            for decomposed_op in decomposed:
                output_func(
                    protocols.quil(decomposed_op, formatter=self.formatter))

    def rename_defgates(self, output: str) -> str:
        """A function for renaming the DEFGATEs within the QUIL output. This
        utilizes a second pass to find each DEFGATE and rename it based on
        a counter.
        """
        result = output
        defString = "DEFGATE"
        nameString = "USERGATE"
        defIdx = 0
        nameIdx = 0
        gateNum = 0
        i = 0
        while i < len(output):
            if result[i] == defString[defIdx]:
                defIdx += 1
            else:
                defIdx = 0
            if result[i] == nameString[nameIdx]:
                nameIdx += 1
            else:
                nameIdx = 0
            if defIdx == len(defString):
                gateNum += 1
                defIdx = 0
            if nameIdx == len(nameString):
                result = result[:i + 1] + str(gateNum) + result[i + 1:]
                nameIdx = 0
                i += 1
            i += 1
        return result
