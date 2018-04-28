from typing import Optional, Any, Union

import numpy as np

from cirq import ops, Symbol
from cirq.extension import Extensions
from cirq.google.xmon_gates import ExpZGate, Exp11Gate, ExpWGate


class QuirkGate(ops.Gate):
    """A gate as understood by Quirk's parser.

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
                this gate into a column containing other gates. For example,
                this is not safe if the column contains a control because the
                control would also apply to the other column's gates.
        """
        self.keys = keys
        self.can_merge = can_merge


UNKNOWN_GATE = QuirkGate('UNKNOWN', can_merge=False)


def same_half_turns(a1: float, a2: float, atol=0.0001) -> bool:
    d = (a1 - a2 + 1) % 2 - 1
    return abs(d) < atol


def angle_to_exponent_key(t: Union[float, Symbol]) -> Optional[str]:
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


def z_to_known(gate: Union[ExpZGate, ops.RotZGate]) -> Optional[QuirkGate]:
    e = angle_to_exponent_key(gate.half_turns)
    if e is None:
        return None
    return QuirkGate('Z' + e)


def x_to_known(gate: Union[ops.RotXGate, ExpWGate]) -> Optional[QuirkGate]:
    e = angle_to_exponent_key(gate.half_turns)
    if e is None:
        return None
    return QuirkGate('X' + e)


def y_to_known(gate: Union[ops.RotYGate, ExpWGate]) -> Optional[QuirkGate]:
    e = angle_to_exponent_key(gate.half_turns)
    if e is None:
        return None
    return QuirkGate('Y' + e)


def cz_to_known(gate: Union[ops.Rot11Gate, Exp11Gate]) -> Optional[QuirkGate]:
    e = angle_to_exponent_key(gate.half_turns)
    if e is None:
        return None
    return QuirkGate('•', 'Z' + e, can_merge=False)


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
