# Copyright 2017 Google LLC
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

import numpy as np
from typing import List, Tuple, Optional

from cirq import linalg
from cirq import ops


def is_negligible_turn(turns: float, tolerance: float) -> bool:
    return abs(_signed_mod_1(turns)) < tolerance


def _phase_matrix(angle: float) -> np.ndarray:
    return np.diag([1, np.exp(1j * angle)])


def _rotation_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _signed_mod_1(x: float) -> float:
    return (x + 0.5) % 1 - 0.5


def _deconstruct_single_qubit_matrix_into_angles(
        mat: np.ndarray) -> Tuple[float, float, float]:
    """Breaks down a 2x2 unitary into more useful ZYZ angle parameters.

    Args:
        mat: The 2x2 unitary matrix to break down.

    Returns:
        A tuple containing the amount to phase around Z, then rotate around Y,
        then phase around Z (all in radians).
    """
    # Anti-cancel left-vs-right phase along top row.
    right_phase = cmath.phase(mat[0, 1] * np.conj(mat[0, 0])) + math.pi
    mat = np.dot(mat, _phase_matrix(-right_phase))

    # Cancel top-vs-bottom phase along left column.
    bottom_phase = cmath.phase(mat[1, 0] * np.conj(mat[0, 0]))
    mat = np.dot(_phase_matrix(-bottom_phase), mat)

    # Lined up for a rotation. Clear the off-diagonal cells with one.
    rotation = math.atan2(abs(mat[1, 0]), abs(mat[0, 0]))
    mat = np.dot(_rotation_matrix(-rotation), mat)

    # Cancel top-left-vs-bottom-right phase.
    diagonal_phase = cmath.phase(mat[1, 1] * np.conj(mat[0, 0]))

    # Note: Ignoring global phase.
    return right_phase + diagonal_phase, rotation, bottom_phase


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
        _deconstruct_single_qubit_matrix_into_angles(mat))

    # Figure out parameters of the actual gates we will do.
    tau = 2 * np.pi
    xy_turn = 2 * rotation / tau
    xy_phase_turn = 0.25 - pre_phase / tau
    total_z_turn = (post_phase + pre_phase) / tau

    # Normalize turns into the range [-0.5, 0.5).
    return (_signed_mod_1(xy_turn), _signed_mod_1(xy_phase_turn),
            _signed_mod_1(total_z_turn))


def _easy_direction_partial_cz(q0: ops.QubitId, q1: ops.QubitId, t: float):
    """The actual hardware can only do CZs that phase counter-clockwise.

    This method replaces clockwise phase(t) to counter-clockwise.

    Args:
      q0: The first qubit being operated on.
      q1: The other qubit being operated on.
      t: The parameter to describe partial-CZ(CZ^t).

    Yields:
      Yields an equivalent circuit for CZ^t with counter-clock phased CZs.
    """
    if t >= 0:
        yield (ops.CZ**t).on(q0, q1)
        return
    yield (ops.Z**t).on(q0)
    yield (ops.X).on(q1)
    yield (ops.CZ**(-t)).on(q0, q1)
    yield (ops.X).on(q1)


def single_qubit_matrix_to_native_gates(
        mat: np.ndarray, tolerance: float = 0
) -> List[ops.ConstantSingleQubitGate]:
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
        ops.XYGate(half_turns=2*xy_turn, axis_half_turns=2*xy_phase_turn),
        ops.ZGate(half_turns=2*total_z_turn)
    ]
    result = [g for g in result if g.trace_distance_bound() > tolerance]

    # Special case: XY half-turns can absorb Z rotations.
    if len(result) == 2 and abs(xy_turn) >= 0.5 - tolerance:
        return [
            ops.XYGate(axis_half_turns=2*xy_phase_turn + total_z_turn)
        ]

    return result


def single_qubit_op_to_framed_phase_form(
        mat: np.ndarray) -> Tuple[np.ndarray, complex, complex]:
    """Decomposes a 2x2 unitary M into U^-1 * diag(1, r) * U * diag(g, g).

    U translates the rotation axis of M to the Z axis.
    g fixes a global phase factor difference caused by the translation.
    r's phase is the amount of rotation around M's rotation axis.

    This decomposition can be used to decompose controlled single-qubit
    rotations into controlled-Z operations bordered by single-qubit operations.

    Args:
      mat:  The qubit operation as a 2x2 unitary matrix.

    Returns:
        A 2x2 unitary U, the complex relative phase factor r, and the complex
        global phase factor g. Applying M is equivalent (up to global phase) to
        applying U, rotating around the Z axis to apply r, then un-applying U.
        When M is controlled, the control must be rotated around the Z axis to
        apply g.
    """
    vals, vecs = np.linalg.eig(mat)
    u = np.conj(vecs).T
    r = vals[1] / vals[0]
    g = vals[0]
    return u, r, g


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
    if u_gates and isinstance(u_gates[-1], ops.ZGate):
        # Don't keep border operations that commute with CZ.
        del u_gates[-1]

    ops_before = [gate(target) for gate in u_gates]
    ops_after = ops.inverse_of_invertable_op_tree(ops_before)
    effect = (ops.CZ**(cmath.phase(z_phase) / math.pi))(control, target)
    kickback = (ops.Z**(cmath.phase(global_phase) / math.pi))(control)

    return list(ops.flatten_op_tree((
        ops_before,
        effect,
        kickback if abs(global_phase - 1) > tolerance else [],
        ops_after)))


def _xx_interaction_via_full_czs(q0: ops.QubitId,
                                 q1: ops.QubitId,
                                 x: float):
    a = x * -2 / np.pi
    yield ops.H.on(q1)
    yield ops.CZ.on(q0, q1)
    yield (ops.X**a).on(q0)
    yield ops.CZ.on(q0, q1)
    yield ops.H.on(q1)


def _xx_yy_interaction_via_full_czs(q0: ops.QubitId,
                                    q1: ops.QubitId,
                                    x: float,
                                    y: float):
    a = x * -2 / np.pi
    b = y * -2 / np.pi
    yield (ops.X**0.5).on(q0)
    yield ops.H.on(q1)
    yield ops.CZ.on(q0, q1)
    yield ops.H.on(q1)
    yield (ops.X**a).on(q0)
    yield (ops.Y**b).on(q1)
    yield ops.H.on(q1)
    yield ops.CZ.on(q0, q1)
    yield ops.H.on(q1)
    yield (ops.X**-0.5).on(q0)


def _xx_yy_zz_interaction_via_full_czs(q0: ops.QubitId,
                                       q1: ops.QubitId,
                                       x: float,
                                       y: float,
                                       z: float):
    a = x * -2 / np.pi + 0.5
    b = y * -2 / np.pi + 0.5
    c = z * -2 / np.pi + 0.5
    yield (ops.X**0.5).on(q0)
    yield ops.H.on(q1)
    yield ops.CZ.on(q0, q1)
    yield ops.H.on(q1)
    yield (ops.X**a).on(q0)
    yield (ops.Y**b).on(q1)
    yield ops.H.on(q0)
    yield (ops.CZ).on(q1, q0)
    yield ops.H.on(q0)
    yield (ops.X**-0.5).on(q1)
    yield (ops.Z**c).on(q1)
    yield ops.H.on(q1)
    yield ops.CZ.on(q0, q1)
    yield ops.H.on(q1)


def two_qubit_matrix_to_native_gates(q0: ops.QubitId,
                                     q1: ops.QubitId,
                                     mat: np.ndarray,
                                     allow_partial_czs: bool,
                                     tolerance: float = 1e-8
                                     ) -> List[ops.Operation]:
    """Decomposes a two-qubit operation into Z/XY/CZ gates.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: Defines the operation to apply to the pair of qubits.
        allow_partial_czs: Enables the use of Partial-CZ gates.
        tolerance: A limit on the amount of error introduced by the
            construction.

    Returns:
        A list of operations implementing the matrix.
    """
    _, (a1, a0), (x, y, z), (b1, b0) = linalg.kak_decomposition(
        mat,
        linalg.Tolerance(atol=tolerance))

    def is_trivial_angle(rad: float) -> bool:
        return abs(rad) < tolerance or abs(rad - np.pi / 4) < tolerance

    def parity_interaction(rads: float,
                           op: Optional[ops.ReversibleGate] = None):
        """Yields a ZZ interaction framed by the given operation."""
        if abs(rads) < tolerance:
            return

        h = rads * -2 / np.pi
        if op is not None:
            yield op.on(q0), op.on(q1)

        # If rads is pi/4 radians within tolerance, single full-CZ suffices.
        if abs(rads - (np.pi / 4)) < tolerance:
            yield ops.CZ.on(q0, q1)
        else:
            yield _easy_direction_partial_cz(q0, q1, -2 * h)

        yield (ops.Z**h).on(q0)
        yield (ops.Z**h).on(q1)
        if op is not None:
            yield op.inverse().on(q0), op.inverse().on(q1)

    def do_single_on(u, q):
        for gate in single_qubit_matrix_to_native_gates(u, tolerance):
            yield gate(q)

    def non_local_part():
        """Yields non-local operation of KAK decomposition."""

        if allow_partial_czs or all(is_trivial_angle(e) for e in [x, y, z]):
            return [
                parity_interaction(x, ops.Y**-0.5),
                parity_interaction(y, ops.X**0.5),
                parity_interaction(z)
            ]

        if abs(z) >= tolerance:
            return _xx_yy_zz_interaction_via_full_czs(q0, q1, x, y, z)

        if y >= tolerance:
            return _xx_yy_interaction_via_full_czs(q0, q1, x, y)

        return _xx_interaction_via_full_czs(q0, q1, x)

    pre = [do_single_on(b1, q1), do_single_on(b0, q0)]
    post = [do_single_on(a1, q1), do_single_on(a0, q0)]

    return list(ops.flatten_op_tree([
        pre,
        non_local_part(),
        post,
    ]))
