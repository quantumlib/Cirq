# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, cast, Dict, Optional, Union

import numpy as np

from cirq import ops, value


class QuirkOp:
    """An operation as understood by Quirk's parser.

    Basically just a series of text identifiers for each qubit, and some rules
    for how things can be combined.
    """

    def __init__(self, *keys: Any, can_merge: bool=True) -> None:
        """
        Args:
            *keys: The JSON object(s) that each qubit is turned into when
                explaining a gate to Quirk. For example, a CNOT is turned into
                the keys ["•", "X"].

                Note that, when keys terminates early, it is implied that later
                qubits should use the same key as the last key.
            can_merge: Whether or not it is safe to merge a column containing
                this operation into a column containing other operations. For
                example, this is not safe if the column contains a control
                because the control would also apply to the other column's
                gates.
        """
        self.keys = keys
        self.can_merge = can_merge


UNKNOWN_GATE = QuirkOp('UNKNOWN', can_merge=False)


def same_half_turns(a1: float, a2: float, atol=0.0001) -> bool:
    d = (a1 - a2 + 1) % 2 - 1
    return abs(d) < atol


def angle_to_exponent_key(t: Union[float, value.Symbol]) -> Optional[str]:
    if isinstance(t, value.Symbol):
        return '^t'

    if same_half_turns(t, 1):
        return ''

    if same_half_turns(t, 0.5):
        return '^½'

    if same_half_turns(t, -0.5):
        return '^-½'

    if same_half_turns(t, 0.25):
        return '^¼'

    if same_half_turns(t, -0.25):
        return '^-¼'

    return None


def single_qubit_matrix_gate(matrix: Optional[np.ndarray]) -> Optional[QuirkOp]:
    if matrix is None or matrix.shape[0] != 2:
        return None

    matrix = matrix.round(6)
    matrix_repr = '{{%s+%si,%s+%si},{%s+%si,%s+%si}}' % (
        np.real(matrix[0, 0]), np.imag(matrix[0, 0]),
        np.real(matrix[1, 0]), np.imag(matrix[1, 0]),
        np.real(matrix[0, 1]), np.imag(matrix[0, 1]),
        np.real(matrix[1, 1]), np.imag(matrix[1, 1]))

    # Clean up.
    matrix_repr = matrix_repr.replace('+-', '-')
    matrix_repr = matrix_repr.replace('+0.0i', '')
    matrix_repr = matrix_repr.replace('.0,', ',')
    matrix_repr = matrix_repr.replace('.0}', '}')
    matrix_repr = matrix_repr.replace('.0+', '+')
    matrix_repr = matrix_repr.replace('.0-', '-')

    return QuirkOp({
        'id': '?',
        'matrix': matrix_repr
    })


def known_quirk_op_for_operation(op: ops.Operation) -> Optional[QuirkOp]:
    if isinstance(op, ops.GateOperation):
        return _gate_to_quirk_op(op.gate)
    return None


def _gate_to_quirk_op(gate: ops.Gate) -> Optional[QuirkOp]:
    for gate_type, func in _known_gate_conversions.items():
        if isinstance(gate, gate_type):
            return func(gate)
    return None


def x_to_known(gate: ops.XPowGate) -> Optional[QuirkOp]:
    e = angle_to_exponent_key(gate.exponent)
    if e is None:
        return None
    return QuirkOp('X' + e)


def y_to_known(gate: ops.YPowGate) -> Optional[QuirkOp]:
    e = angle_to_exponent_key(gate.exponent)
    if e is None:
        return None
    return QuirkOp('Y' + e)


def z_to_known(gate: ops.ZPowGate) -> Optional[QuirkOp]:
    e = angle_to_exponent_key(gate.exponent)
    if e is None:
        return None
    return QuirkOp('Z' + e)


def cz_to_known(gate: ops.CZPowGate) -> Optional[QuirkOp]:
    e = angle_to_exponent_key(gate.exponent)
    if e is None:
        return None
    return QuirkOp('•', 'Z' + e, can_merge=False)


def cnot_to_known(gate: ops.CNotPowGate) -> Optional[QuirkOp]:
    e = angle_to_exponent_key(gate.exponent)
    if e is None:
        return None
    return QuirkOp('•', 'X' + e, can_merge=False)


def h_to_known(gate: ops.HPowGate) -> Optional[QuirkOp]:
    if gate.exponent == 1:
        return QuirkOp('H')
    return None


def swap_to_known(gate: ops.SwapPowGate) -> Optional[QuirkOp]:
    if gate.exponent == 1:
        return QuirkOp('Swap', 'Swap')
    return None


_known_gate_conversions = cast(
    Dict[type, Callable[[ops.Gate], Optional[QuirkOp]]],
    {
        ops.XPowGate: x_to_known,
        ops.YPowGate: y_to_known,
        ops.ZPowGate: z_to_known,
        ops.CZPowGate: cz_to_known,
        ops.CNotPowGate: cnot_to_known,
        ops.SwapPowGate: swap_to_known,
        ops.HPowGate: h_to_known,
        ops.MeasurementGate: lambda _: QuirkOp('Measure')
    }
)
