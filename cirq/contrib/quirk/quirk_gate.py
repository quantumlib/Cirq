from typing import Optional, Any

import numpy as np

from cirq import ops, Symbol
from cirq.extension import Extensions
from cirq.google.xmon_gates import ExpZGate, Exp11Gate, ExpWGate


class QuirkGate(ops.Gate):
    def __init__(self, *keys: Any, can_merge: bool=True):
        self.keys = keys
        self.can_merge = can_merge


UNKNOWN_GATE = QuirkGate('UNKNOWN', can_merge=False)


def same_half_turns(a1: float, a2: float) -> bool:
    d = (a1 - a2 + 1) % 2 - 1
    return abs(d) < 0.001


def angle_to_exponent(t):
    if isinstance(t, Symbol):
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


def z_to_known(gate: ExpZGate) -> Optional[QuirkGate]:
    e = angle_to_exponent(gate.half_turns)
    if e is None:
        return None
    return QuirkGate('Z' + e)


def x_to_known(gate: ops.RotXGate) -> Optional[QuirkGate]:
    e = angle_to_exponent(gate.half_turns)
    if e is None:
        return None
    return QuirkGate('X' + e)


def y_to_known(gate: ops.RotYGate) -> Optional[QuirkGate]:
    e = angle_to_exponent(gate.half_turns)
    if e is None:
        return None
    return QuirkGate('Y' + e)


def cz_to_known(gate: Exp11Gate) -> Optional[QuirkGate]:
    z = z_to_known(gate)
    if z is not None:
        return QuirkGate('•', z.keys[0], can_merge=False)
    return None


def w_to_known(gate: ExpWGate) -> Optional[QuirkGate]:
    if isinstance(gate.axis_half_turns, Symbol):
        return None
    p = (gate.axis_half_turns + 1) % 2 - 1
    if same_half_turns(p, 0):
        return x_to_known(gate)
    if same_half_turns(p, 0.5):
        return y_to_known(gate)
    return None


def single_qubit_matrix_gate(gate: ops.KnownMatrixGate) -> Optional[QuirkGate]:
    matrix = gate.matrix()
    if matrix.shape[0] != 2:
        return None

    matrix_repr = '{{%s+%si,%s+%si},{%s+%si,%s+%si}}' % (
        np.real(matrix[0, 0]), np.imag(matrix[0, 0]),
        np.real(matrix[1, 0]), np.imag(matrix[1, 0]),
        np.real(matrix[0, 1]), np.imag(matrix[0, 1]),
        np.real(matrix[1, 1]), np.imag(matrix[1, 1]))
    return QuirkGate({
        'id': '?',
        'matrix': matrix_repr
    })


quirk_gate_ext = Extensions()
quirk_gate_ext.add_cast(QuirkGate, ops.RotXGate, x_to_known)
quirk_gate_ext.add_cast(QuirkGate, ops.RotYGate, y_to_known)
quirk_gate_ext.add_cast(QuirkGate, ops.RotZGate, z_to_known)
quirk_gate_ext.add_cast(QuirkGate, ExpZGate, z_to_known)
quirk_gate_ext.add_cast(QuirkGate, ExpWGate, w_to_known)
quirk_gate_ext.add_cast(QuirkGate, ops.Rot11Gate, cz_to_known)
quirk_gate_ext.add_cast(QuirkGate, Exp11Gate, cz_to_known)
quirk_gate_ext.add_cast(QuirkGate,
                        ops.CNotGate,
                        lambda e: QuirkGate('•', 'X',
                                            can_merge=False))
quirk_gate_ext.add_cast(QuirkGate,
                        ops.SwapGate,
                        lambda e: QuirkGate('Swap', 'Swap'))
quirk_gate_ext.add_cast(QuirkGate,
                        ops.HGate,
                        lambda e: QuirkGate('H'))
quirk_gate_ext.add_cast(QuirkGate,
                        ops.MeasurementGate,
                        lambda e: QuirkGate('Measure'))
