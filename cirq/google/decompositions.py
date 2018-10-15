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

"""Utility methods related to optimizing quantum circuits."""

import cmath
import math
from typing import List, Tuple

import numpy as np

from cirq import ops, linalg, protocols
from cirq.decompositions import single_qubit_op_to_framed_phase_form
from cirq.google.xmon_gates import ExpWGate


def _signed_mod_1(x: float) -> float:
    return (x + 0.5) % 1 - 0.5


def _deconstruct_single_qubit_matrix_into_gate_turns(
        mat: np.ndarray) -> Tuple[float, float, float]:
    """Breaks down a 2x2 unitary into gate parameters.

    Args:
        mat: The 2x2 unitary matrix to break down.

    Returns:
       A tuple containing the amount to rotate around an XY axis, the phase of
       that axis, and the amount to phase around Z. All results will be in
       fractions of a whole turn, with values canonicalized into the range
       [-0.5, 0.5).
    """
    pre_phase, rotation, post_phase = (
        linalg.deconstruct_single_qubit_matrix_into_angles(mat))

    # Figure out parameters of the actual gates we will do.
    tau = 2 * np.pi
    xy_turn = rotation / tau
    xy_phase_turn = 0.25 - pre_phase / tau
    total_z_turn = (post_phase + pre_phase) / tau

    # Normalize turns into the range [-0.5, 0.5).
    return (_signed_mod_1(xy_turn), _signed_mod_1(xy_phase_turn),
            _signed_mod_1(total_z_turn))


def single_qubit_matrix_to_native_gates(
        mat: np.ndarray, tolerance: float = 0
) -> List[ops.SingleQubitGate]:
    """Implements a single-qubit operation with few native gates.

    Args:
        mat: The 2x2 unitary matrix of the operation to implement.
        tolerance: A limit on the amount of error introduced by the
            construction.

    Returns:
        A list of gates that, when applied in order, perform the desired
            operation.
    """

    xy_turn, xy_phase_turn, total_z_turn = (
        _deconstruct_single_qubit_matrix_into_gate_turns(mat))

    # Build the intended operation out of non-negligible XY and Z rotations.
    result = [
        ExpWGate(half_turns=2*xy_turn, axis_half_turns=2*xy_phase_turn),
        ops.RotZGate(half_turns=2 * total_z_turn)
    ]
    result = [
        g for g in result
        if protocols.trace_distance_bound(g) > tolerance
    ]

    # Special case: XY half-turns can absorb Z rotations.
    if len(result) == 2 and abs(xy_turn) >= 0.5 - tolerance:
        return [
            ExpWGate(axis_half_turns=2*xy_phase_turn + total_z_turn)
        ]

    return result


def controlled_op_to_native_gates(
        control: ops.QubitId,
        target: ops.QubitId,
        operation: np.ndarray,
        tolerance: float = 0.0) -> List[ops.Operation]:
    """Decomposes a controlled single-qubit operation into Z/XY/CZ gates.

    Args:
        control: The control qubit.
        target: The qubit to apply an operation to, when the control is on.
        operation: The single-qubit operation being controlled.
        tolerance: A limit on the amount of error introduced by the
            construction.

    Returns:
        A list of Operations that apply the controlled operation.
    """
    u, z_phase, global_phase = single_qubit_op_to_framed_phase_form(operation)
    if abs(z_phase - 1) <= tolerance:
        return []

    u_gates = single_qubit_matrix_to_native_gates(u, tolerance)
    if u_gates and isinstance(u_gates[-1], ops.RotZGate):
        # Don't keep border operations that commute with CZ.
        del u_gates[-1]

    ops_before = [gate(target) for gate in u_gates]
    ops_after = protocols.inverse(ops_before)
    effect = ops.CZ(control, target) ** (cmath.phase(z_phase) / math.pi)
    kickback = ops.Z(control) ** (cmath.phase(global_phase) / math.pi)

    return list(ops.flatten_op_tree((
        ops_before,
        effect,
        kickback if abs(global_phase - 1) > tolerance else [],
        ops_after)))
