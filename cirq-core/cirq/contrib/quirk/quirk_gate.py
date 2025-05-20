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

from __future__ import annotations

from typing import Any, Callable, cast, Dict, Optional, Union

import numpy as np
import sympy

from cirq import ops


class QuirkOp:
    """An operation as understood by Quirk's parser.

    Basically just a series of text identifiers for each qubit, and some rules
    for how things can be combined.
    """

    def __init__(self, *keys: Any, can_merge: bool = True) -> None:
        """Inits QuirkOp.

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

    def controlled(self, control_count: int = 1) -> QuirkOp:
        return QuirkOp(*['•'] * control_count, *self.keys, can_merge=False)


UNKNOWN_GATE = QuirkOp('UNKNOWN', can_merge=False)


def same_half_turns(a1: float, a2: float, atol=0.0001) -> bool:
    d = (a1 - a2 + 1) % 2 - 1
    return abs(d) < atol


def _is_supported_formula(formula: sympy.Basic) -> bool:
    if isinstance(
        formula, (sympy.Symbol, sympy.Integer, sympy.Float, sympy.Rational, sympy.NumberSymbol)
    ):
        return True
    if isinstance(formula, (sympy.Add, sympy.Mul)):
        return all(_is_supported_formula(f) for f in formula.args)
    return False


def _val_to_quirk_formula(t: Union[float, sympy.Basic]) -> str:
    if isinstance(t, sympy.Basic):
        if not set(t.free_symbols) <= {sympy.Symbol('t')}:
            raise ValueError(f'Symbol other than "t": {t!r}.')
        if not _is_supported_formula(t):
            raise ValueError(f'Formula uses unsupported operations: {t!r}')
        return str(t)

    return f'{float(t):.4f}'


def angle_to_exponent_key(t: Union[float, sympy.Basic]) -> Optional[str]:
    if isinstance(t, sympy.Basic):
        if t == sympy.Symbol('t'):
            return '^t'

        if t == -sympy.Symbol('t'):
            return '^-t'

        return None

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

    # pylint: disable=consider-using-f-string
    matrix = matrix.round(6)
    matrix_repr = '{{%s+%si,%s+%si},{%s+%si,%s+%si}}' % (
        np.real(matrix[0, 0]),
        np.imag(matrix[0, 0]),
        np.real(matrix[1, 0]),
        np.imag(matrix[1, 0]),
        np.real(matrix[0, 1]),
        np.imag(matrix[0, 1]),
        np.real(matrix[1, 1]),
        np.imag(matrix[1, 1]),
    )

    # Clean up.
    matrix_repr = matrix_repr.replace('+-', '-')
    matrix_repr = matrix_repr.replace('+0.0i', '')
    matrix_repr = matrix_repr.replace('.0,', ',')
    matrix_repr = matrix_repr.replace('.0}', '}')
    matrix_repr = matrix_repr.replace('.0+', '+')
    matrix_repr = matrix_repr.replace('.0-', '-')

    return QuirkOp({'id': '?', 'matrix': matrix_repr})


def known_quirk_op_for_operation(op: ops.Operation) -> Optional[QuirkOp]:
    if isinstance(op, ops.GateOperation):
        return _gate_to_quirk_op(op.gate)
    if isinstance(op, ops.ControlledOperation):
        return controlled_unwrap(op)
    return None


def _gate_to_quirk_op(gate: ops.Gate) -> Optional[QuirkOp]:
    for gate_type, func in _known_gate_conversions.items():
        if isinstance(gate, gate_type):
            return func(gate)
    return None


def xyz_to_quirk_op(axis: str, gate: ops.EigenGate) -> QuirkOp:
    d = axis.lower()
    u = axis.upper()

    if gate.global_shift == -0.5:
        return QuirkOp({'id': f'R{d}ft', 'arg': f'({_val_to_quirk_formula(gate.exponent)}) pi'})

    e = angle_to_exponent_key(gate.exponent)
    if e is not None:
        return QuirkOp(u + e)

    return QuirkOp({'id': f'{u}^ft', 'arg': f'{_val_to_quirk_formula(gate.exponent)}'})


def x_to_quirk_op(gate: ops.XPowGate) -> QuirkOp:
    return xyz_to_quirk_op('x', gate)


def y_to_quirk_op(gate: ops.YPowGate) -> QuirkOp:
    return xyz_to_quirk_op('y', gate)


def z_to_quirk_op(gate: ops.ZPowGate) -> QuirkOp:
    return xyz_to_quirk_op('z', gate)


def cz_to_quirk_op(gate: ops.CZPowGate) -> Optional[QuirkOp]:
    return z_to_quirk_op(ops.Z**gate.exponent).controlled()


def cnot_to_quirk_op(gate: ops.CXPowGate) -> Optional[QuirkOp]:
    return x_to_quirk_op(ops.X**gate.exponent).controlled()


def h_to_quirk_op(gate: ops.HPowGate) -> Optional[QuirkOp]:
    if gate.exponent == 1:
        return QuirkOp('H')
    return None


def swap_to_quirk_op(gate: ops.SwapPowGate) -> Optional[QuirkOp]:
    if gate.exponent == 1:
        return QuirkOp('Swap', 'Swap', can_merge=False)
    return None


def cswap_to_quirk_op(gate: ops.CSwapGate) -> Optional[QuirkOp]:
    return QuirkOp('•', 'Swap', 'Swap', can_merge=False)


def ccx_to_quirk_op(gate: ops.CCXPowGate) -> Optional[QuirkOp]:
    e = angle_to_exponent_key(gate.exponent)
    if e is None:
        return None
    return QuirkOp('•', '•', 'X' + e, can_merge=False)


def ccz_to_quirk_op(gate: ops.CCZPowGate) -> Optional[QuirkOp]:
    e = angle_to_exponent_key(gate.exponent)
    if e is None:
        return None
    return QuirkOp('•', '•', 'Z' + e, can_merge=False)


def controlled_unwrap(op: ops.ControlledOperation) -> Optional[QuirkOp]:
    sub = known_quirk_op_for_operation(op.sub_operation)
    if sub is None:
        return None
    return sub.controlled(len(op.controls))


_known_gate_conversions = cast(
    Dict[type, Callable[[ops.Gate], Optional[QuirkOp]]],
    {
        ops.CCXPowGate: ccx_to_quirk_op,
        ops.CCZPowGate: ccz_to_quirk_op,
        ops.CSwapGate: cswap_to_quirk_op,
        ops.XPowGate: x_to_quirk_op,
        ops.YPowGate: y_to_quirk_op,
        ops.ZPowGate: z_to_quirk_op,
        ops.CXPowGate: cnot_to_quirk_op,
        ops.CZPowGate: cz_to_quirk_op,
        ops.SwapPowGate: swap_to_quirk_op,
        ops.HPowGate: h_to_quirk_op,
        ops.MeasurementGate: lambda _: QuirkOp('Measure'),
    },
)
