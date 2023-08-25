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
import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops


def to_quil_complex_format(num) -> str:
    """A function for outputting a number to a complex string in QUIL format."""
    cnum = complex(str(num))
    return f"{cnum.real}+{cnum.imag}i"


class QuilFormatter(string.Formatter):
    """A unique formatter to correctly output values to QUIL."""

    def __init__(
        self, qubit_id_map: Dict['cirq.Qid', str], measurement_id_map: Dict[str, str]
    ) -> None:
        """Inits QuilFormatter.

        Args:
            qubit_id_map: A dictionary {qubit, quil_output_string} for
            the proper QUIL output for each qubit.
            measurement_id_map: A dictionary {measurement_key,
            quil_output_string} for the proper QUIL output for each
            measurement key.
        """
        self.qubit_id_map = {} if qubit_id_map is None else qubit_id_map
        self.measurement_id_map = {} if measurement_id_map is None else measurement_id_map

    def format_field(self, value: Any, spec: str) -> str:
        if isinstance(value, cirq.ops.Qid):
            value = self.qubit_id_map[value]
        if isinstance(value, str) and spec == 'meas':
            value = self.measurement_id_map[value]
            spec = ''
        return super().format_field(value, spec)


@value.value_equality(approximate=True)
class QuilOneQubitGate(ops.Gate):
    """A QUIL gate representing any single qubit unitary with a DEFGATE and
    2x2 matrix in QUIL.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """Inits QuilOneQubitGate.

        Args:
            matrix: The 2x2 unitary matrix for this gate.
        """
        self.matrix = matrix

    def _num_qubits_(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f'cirq.circuits.quil_output.QuilOneQubitGate(matrix=\n{self.matrix}\n)'

    def _value_equality_values_(self):
        return self.matrix


@value.value_equality(approximate=True)
class QuilTwoQubitGate(ops.Gate):
    """A two qubit gate represented in QUIL with a DEFGATE and it's 4x4
    unitary matrix.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """Inits QuilTwoQubitGate.

        Args:
            matrix: The 4x4 unitary matrix for this gate.
        """
        self.matrix = matrix

    def _num_qubits_(self) -> int:
        return 2

    def _value_equality_values_(self):
        return self.matrix

    def __repr__(self) -> str:
        return f'cirq.circuits.quil_output.QuilTwoQubitGate(matrix=\n{self.matrix}\n)'


def _ccnotpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> Optional[str]:
    gate = cast(cirq.CCNotPowGate, op.gate)
    if gate._exponent != 1:
        return None
    return formatter.format('CCNOT {0} {1} {2}\n', op.qubits[0], op.qubits[1], op.qubits[2])


def _cczpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> Optional[str]:
    gate = cast(cirq.CCZPowGate, op.gate)
    if gate._exponent != 1:
        return None
    lines = [
        formatter.format('H {0}\n', op.qubits[2]),
        formatter.format('CCNOT {0} {1} {2}\n', op.qubits[0], op.qubits[1], op.qubits[2]),
        formatter.format('H {0}\n', op.qubits[2]),
    ]
    return ''.join(lines)


def _cnotpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> Optional[str]:
    gate = cast(cirq.CNotPowGate, op.gate)
    if gate._exponent == 1:
        return formatter.format('CNOT {0} {1}\n', op.qubits[0], op.qubits[1])
    return None


def _cswap_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    return formatter.format('CSWAP {0} {1} {2}\n', op.qubits[0], op.qubits[1], op.qubits[2])


def _czpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.CZPowGate, op.gate)
    if gate._exponent == 1:
        return formatter.format('CZ {0} {1}\n', op.qubits[0], op.qubits[1])
    return formatter.format(
        'CPHASE({0}) {1} {2}\n', gate._exponent * np.pi, op.qubits[0], op.qubits[1]
    )


def _hpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.HPowGate, op.gate)
    if gate._exponent == 1:
        return formatter.format('H {0}\n', op.qubits[0])
    return formatter.format(
        'RY({0}) {3}\nRX({1}) {3}\nRY({2}) {3}\n',
        0.25 * np.pi,
        gate._exponent * np.pi,
        -0.25 * np.pi,
        op.qubits[0],
    )


def _identity_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    return ''.join(formatter.format('I {0}\n', qubit) for qubit in op.qubits)


def _iswappow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.ISwapPowGate, op.gate)
    if gate._exponent == 1:
        return formatter.format('ISWAP {0} {1}\n', op.qubits[0], op.qubits[1])
    return formatter.format('XY({0}) {1} {2}\n', gate._exponent * np.pi, op.qubits[0], op.qubits[1])


def _measurement_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.MeasurementGate, op.gate)
    invert_mask = gate.invert_mask
    if len(invert_mask) < len(op.qubits):
        invert_mask = invert_mask + (False,) * (len(op.qubits) - len(invert_mask))
    lines = []
    for i, (qubit, inv) in enumerate(zip(op.qubits, invert_mask)):
        if inv:
            lines.append(formatter.format('X {0} # Inverting for following measurement\n', qubit))
        lines.append(formatter.format('MEASURE {0} {1:meas}[{2}]\n', qubit, gate.key, i))
    return ''.join(lines)


def _quilonequbit_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(QuilOneQubitGate, op.gate)
    return (
        f'DEFGATE USERGATE:\n    '
        f'{to_quil_complex_format(gate.matrix[0, 0])}, '
        f'{to_quil_complex_format(gate.matrix[0, 1])}\n    '
        f'{to_quil_complex_format(gate.matrix[1, 0])}, '
        f'{to_quil_complex_format(gate.matrix[1, 1])}\n'
        f'{formatter.format("USERGATE {0}", op.qubits[0])}\n'
    )


def _quiltwoqubit_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(QuilOneQubitGate, op.gate)
    return (
        f'DEFGATE USERGATE:\n    '
        f'{to_quil_complex_format(gate.matrix[0, 0])}, '
        f'{to_quil_complex_format(gate.matrix[0, 1])}, '
        f'{to_quil_complex_format(gate.matrix[0, 2])}, '
        f'{to_quil_complex_format(gate.matrix[0, 3])}\n    '
        f'{to_quil_complex_format(gate.matrix[1, 0])}, '
        f'{to_quil_complex_format(gate.matrix[1, 1])}, '
        f'{to_quil_complex_format(gate.matrix[1, 2])}, '
        f'{to_quil_complex_format(gate.matrix[1, 3])}\n    '
        f'{to_quil_complex_format(gate.matrix[2, 0])}, '
        f'{to_quil_complex_format(gate.matrix[2, 1])}, '
        f'{to_quil_complex_format(gate.matrix[2, 2])}, '
        f'{to_quil_complex_format(gate.matrix[2, 3])}\n    '
        f'{to_quil_complex_format(gate.matrix[3, 0])}, '
        f'{to_quil_complex_format(gate.matrix[3, 1])}, '
        f'{to_quil_complex_format(gate.matrix[3, 2])}, '
        f'{to_quil_complex_format(gate.matrix[3, 3])}\n'
        f'{formatter.format("USERGATE {0} {1}", op.qubits[0], op.qubits[1])}\n'
    )


def _swappow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.SwapPowGate, op.gate)
    if gate._exponent % 2 == 1:
        return formatter.format('SWAP {0} {1}\n', op.qubits[0], op.qubits[1])
    return formatter.format(
        'PSWAP({0}) {1} {2}\n', gate._exponent * np.pi, op.qubits[0], op.qubits[1]
    )


def _twoqubitdiagonal_gate(op: cirq.Operation, formatter: QuilFormatter) -> Optional[str]:
    gate = cast(cirq.TwoQubitDiagonalGate, op.gate)
    diag_angles_radians = np.asarray(gate._diag_angles_radians)
    if np.count_nonzero(diag_angles_radians) == 1:
        if diag_angles_radians[0] != 0:
            return formatter.format(
                'CPHASE00({0}) {1} {2}\n', diag_angles_radians[0], op.qubits[0], op.qubits[1]
            )
        elif diag_angles_radians[1] != 0:
            return formatter.format(
                'CPHASE01({0}) {1} {2}\n', diag_angles_radians[1], op.qubits[0], op.qubits[1]
            )
        elif diag_angles_radians[2] != 0:
            return formatter.format(
                'CPHASE10({0}) {1} {2}\n', diag_angles_radians[2], op.qubits[0], op.qubits[1]
            )
        elif diag_angles_radians[3] != 0:
            return formatter.format(
                'CPHASE({0}) {1} {2}\n', diag_angles_radians[3], op.qubits[0], op.qubits[1]
            )
    return None


def _wait_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    return 'WAIT\n'


def _xpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.XPowGate, op.gate)
    if gate._exponent == 1 and gate._global_shift != -0.5:
        return formatter.format('X {0}\n', op.qubits[0])
    return formatter.format('RX({0}) {1}\n', gate._exponent * np.pi, op.qubits[0])


def _xxpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.XPowGate, op.gate)
    if gate._exponent == 1:
        return formatter.format('X {0}\nX {1}\n', op.qubits[0], op.qubits[1])
    return formatter.format(
        'RX({0}) {1}\nRX({2}) {3}\n',
        gate._exponent * np.pi,
        op.qubits[0],
        gate._exponent * np.pi,
        op.qubits[1],
    )


def _ypow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.YPowGate, op.gate)
    if gate._exponent == 1 and gate.global_shift != -0.5:
        return formatter.format('Y {0}\n', op.qubits[0])
    return formatter.format('RY({0}) {1}\n', gate._exponent * np.pi, op.qubits[0])


def _yypow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.YYPowGate, op.gate)
    if gate._exponent == 1:
        return formatter.format('Y {0}\nY {1}\n', op.qubits[0], op.qubits[1])

    return formatter.format(
        'RY({0}) {1}\nRY({2}) {3}\n',
        gate._exponent * np.pi,
        op.qubits[0],
        gate._exponent * np.pi,
        op.qubits[1],
    )


def _zpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.ZPowGate, op.gate)
    if gate._exponent == 1 and gate.global_shift != -0.5:
        return formatter.format('Z {0}\n', op.qubits[0])
    return formatter.format('RZ({0}) {1}\n', gate._exponent * np.pi, op.qubits[0])


def _zzpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.ZZPowGate, op.gate)
    if gate._exponent == 1:
        return formatter.format('Z {0}\nZ {1}\n', op.qubits[0], op.qubits[1])

    return formatter.format(
        'RZ({0}) {1}\nRZ({2}) {3}\n',
        gate._exponent * np.pi,
        op.qubits[0],
        gate._exponent * np.pi,
        op.qubits[1],
    )


SUPPORTED_GATES = {
    ops.CCNotPowGate: _ccnotpow_gate,
    ops.CCZPowGate: _cczpow_gate,
    ops.CNotPowGate: _cnotpow_gate,
    ops.CSwapGate: _cswap_gate,
    ops.CZPowGate: _czpow_gate,
    ops.HPowGate: _hpow_gate,
    ops.IdentityGate: _identity_gate,
    ops.ISwapPowGate: _iswappow_gate,
    ops.MeasurementGate: _measurement_gate,
    QuilOneQubitGate: _quilonequbit_gate,
    QuilTwoQubitGate: _quiltwoqubit_gate,
    ops.SwapPowGate: _swappow_gate,
    ops.TwoQubitDiagonalGate: _twoqubitdiagonal_gate,
    ops.WaitGate: _wait_gate,
    ops.XPowGate: _xpow_gate,
    ops.XXPowGate: _xxpow_gate,
    ops.YPowGate: _ypow_gate,
    ops.YYPowGate: _yypow_gate,
    ops.ZPowGate: _zpow_gate,
    ops.ZZPowGate: _zzpow_gate,
}


class QuilOutput:
    """An object for passing operations and qubits then outputting them to
    QUIL format. The string representation returns the QUIL output for the
    circuit.
    """

    def __init__(self, operations: 'cirq.OP_TREE', qubits: Tuple['cirq.Qid', ...]) -> None:
        """Inits QuilOutput.

        Args:
            operations: A list or tuple of `cirq.OP_TREE` arguments.
            qubits: The qubits used in the operations.
        """
        self.qubits = qubits
        self.operations = tuple(cirq.ops.flatten_to_ops(operations))
        self.measurements = tuple(
            op for op in self.operations if isinstance(op.gate, ops.MeasurementGate)
        )
        self.qubit_id_map = self._generate_qubit_ids()
        self.measurement_id_map = self._generate_measurement_ids()
        self.formatter = cirq_rigetti.quil_output.QuilFormatter(
            qubit_id_map=self.qubit_id_map, measurement_id_map=self.measurement_id_map
        )

    def _generate_qubit_ids(self) -> Dict['cirq.Qid', str]:
        return {qubit: str(i) for i, qubit in enumerate(self.qubits)}

    def _generate_measurement_ids(self) -> Dict[str, str]:
        index = 0
        measurement_id_map: Dict[str, str] = {}
        for op in self.operations:
            if isinstance(op.gate, ops.MeasurementGate):
                key = protocols.measurement_key_name(op)
                if key in measurement_id_map:
                    continue
                measurement_id_map[key] = f'm{index}'
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

    def _op_to_maybe_quil(self, op: cirq.Operation) -> Optional[str]:
        for gate_type in SUPPORTED_GATES.keys():
            if isinstance(op.gate, gate_type):
                quil: Callable[[cirq.Operation, QuilFormatter], Optional[str]] = SUPPORTED_GATES[
                    gate_type
                ]
                return quil(op, self.formatter)
        return None

    def _op_to_quil(self, op: cirq.Operation) -> str:
        quil_str = self._op_to_maybe_quil(op)
        if not quil_str:
            raise ValueError("Can't convert Operation to string")
        return quil_str

    def _write_quil(self, output_func: Callable[[str], None]) -> None:
        output_func('# Created using Cirq.\n\n')
        if len(self.measurements) > 0:
            measurements_declared: Set[str] = set()
            for m in self.measurements:
                key = protocols.measurement_key_name(m)
                if key in measurements_declared:
                    continue
                measurements_declared.add(key)
                output_func(f'DECLARE {self.measurement_id_map[key]} BIT[{len(m.qubits)}]\n')
            output_func('\n')

        def keep(op: 'cirq.Operation') -> bool:
            if isinstance(op.gate, tuple(SUPPORTED_GATES.keys())):
                if not self._op_to_maybe_quil(op):
                    return False
                return True
            return False

        def fallback(op):
            if len(op.qubits) not in [1, 2]:
                return NotImplemented

            mat = protocols.unitary(op, None)
            if mat is None:
                return NotImplemented

            # Following code is a safety measure
            # Could not find a gate that doesn't decompose into a gate
            # with a _quil_ implementation
            if len(op.qubits) == 1:  # pragma: no cover
                return QuilOneQubitGate(mat).on(*op.qubits)
            return QuilTwoQubitGate(mat).on(*op.qubits)  # pragma: no cover

        def on_stuck(bad_op):
            return ValueError(f'Cannot output operation as QUIL: {bad_op!r}')

        for main_op in self.operations:
            decomposed = protocols.decompose(
                main_op, keep=keep, fallback_decomposer=fallback, on_stuck_raise=on_stuck
            )

            for decomposed_op in decomposed:
                output_func(self._op_to_quil(decomposed_op))

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
                result = result[: i + 1] + str(gateNum) + result[i + 1 :]
                nameIdx = 0
                i += 1
            i += 1
        return result


class RigettiQCSQuilOutput(QuilOutput):
    """A sub-class of `cirq.circuits.quil_output.QuilOutput` that additionally accepts a
    `qubit_id_map` for explicitly mapping logical qubits to physical qubits.

    Attributes:
        qubit_id_map: A dictionary mapping `cirq.Qid` to strings that
            address physical qubits in the outputted QUIL.
        measurement_id_map: A dictionary mapping a Cirq measurement key to
            the corresponding QUIL memory region.
        formatter: A QUIL formatter that formats QUIL strings account for both
            the `qubit_id_map` and `measurement_id_map`.
    """

    def __init__(
        self,
        *,
        operations: cirq.OP_TREE,
        qubits: Tuple[cirq.Qid, ...],
        decompose_operation: Optional[Callable[[cirq.Operation], List[cirq.Operation]]] = None,
        qubit_id_map: Optional[Dict[cirq.Qid, str]] = None,
    ):
        """Initializes an instance of `RigettiQCSQuilOutput`.

        Args:
            operations: A list or tuple of `cirq.OP_TREE` arguments.
            qubits: The qubits used in the operations.
            decompose_operation: Optional; A callable that decomposes a circuit operation
                into a list of equivalent operations. If None provided, this class
                decomposes operations by invoking `QuilOutput._write_quil`.
            qubit_id_map: Optional; A dictionary mapping `cirq.Qid` to strings that
                address physical qubits in the outputted QUIL.
        """
        super().__init__(operations, qubits)
        self.qubit_id_map = qubit_id_map or self._generate_qubit_ids()
        self.measurement_id_map = self._generate_measurement_ids()

        self.formatter = QuilFormatter(
            qubit_id_map=self.qubit_id_map, measurement_id_map=self.measurement_id_map
        )
        self._decompose_operation = decompose_operation

    def _write_quil(self, output_func: Callable[[str], None]) -> None:
        """Calls `output_func` for successive lines of QUIL output.

        Args:
            output_func: A function that accepts a string of QUIL. This will likely
                write the QUIL to a file.

        Returns:
            None.
        """
        if self._decompose_operation is None:
            return super()._write_quil(output_func)

        output_func("# Created using Cirq.\n\n")

        if len(self.measurements) > 0:
            measurements_declared: Set[str] = set()
            for m in self.measurements:
                key = cirq.measurement_key_name(m)
                if key in measurements_declared:
                    continue
                measurements_declared.add(key)
                output_func(f"DECLARE {self.measurement_id_map[key]} BIT[{len(m.qubits)}]\n")
            output_func("\n")

        for main_op in self.operations:
            decomposed = self._decompose_operation(main_op)
            for decomposed_op in decomposed:
                output_func(self._op_to_quil(decomposed_op))
