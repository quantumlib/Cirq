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

from typing import List, Tuple, Optional, cast

import numpy as np

from cirq import ops, linalg, protocols, circuits
from cirq.optimizers import (
    decompositions,
    eject_z,
    eject_phased_paulis,
    merge_single_qubit_gates,
)


def two_qubit_matrix_to_operations(q0: ops.QubitId,
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
    kak = linalg.kak_decomposition(mat, linalg.Tolerance(atol=tolerance))
    operations = _kak_decomposition_to_operations(
        q0, q1, kak, allow_partial_czs, tolerance)
    return _cleanup_operations(operations)


def _xx_interaction_via_full_czs(q0: ops.QubitId,
                                 q1: ops.QubitId,
                                 x: float):
    a = x * -2 / np.pi
    yield ops.H(q1)
    yield ops.CZ(q0, q1)
    yield ops.X(q0)**a
    yield ops.CZ(q0, q1)
    yield ops.H(q1)


def _xx_yy_interaction_via_full_czs(q0: ops.QubitId,
                                    q1: ops.QubitId,
                                    x: float,
                                    y: float):
    a = x * -2 / np.pi
    b = y * -2 / np.pi
    yield ops.X(q0)**0.5
    yield ops.H(q1)
    yield ops.CZ(q0, q1)
    yield ops.H(q1)
    yield ops.X(q0)**a
    yield ops.Y(q1)**b
    yield ops.H(q1)
    yield ops.CZ(q0, q1)
    yield ops.H(q1)
    yield ops.X(q0)**-0.5


def _xx_yy_zz_interaction_via_full_czs(q0: ops.QubitId,
                                       q1: ops.QubitId,
                                       x: float,
                                       y: float,
                                       z: float):
    a = x * -2 / np.pi + 0.5
    b = y * -2 / np.pi + 0.5
    c = z * -2 / np.pi + 0.5
    yield ops.X(q0)**0.5
    yield ops.H(q1)
    yield ops.CZ(q0, q1)
    yield ops.H(q1)
    yield ops.X(q0)**a
    yield ops.Y(q1)**b
    yield ops.H.on(q0)
    yield ops.CZ(q1, q0)
    yield ops.H(q0)
    yield ops.X(q1)**-0.5
    yield ops.Z(q1)**c
    yield ops.H(q1)
    yield ops.CZ(q0, q1)
    yield ops.H(q1)


def _cleanup_operations(operations: List[ops.Operation]):
    circuit = circuits.Circuit.from_ops(operations)
    merge_single_qubit_gates.merge_single_qubit_gates_into_phased_x_z(circuit)
    eject_phased_paulis.EjectPhasedPaulis().optimize_circuit(circuit)
    eject_z.EjectZ().optimize_circuit(circuit)
    circuit = circuits.Circuit.from_ops(
        circuit.all_operations(),
        strategy=circuits.InsertStrategy.EARLIEST)
    return list(circuit.all_operations())


def _kak_decomposition_to_operations(q0: ops.QubitId,
                                     q1: ops.QubitId,
                                     kak: linalg.KakDecomposition,
                                     allow_partial_czs: bool,
                                     tolerance: float = 1e-8
                                     ) -> List[ops.Operation]:
    """Assumes that the decomposition is canonical."""
    b0, b1 = kak.single_qubit_operations_before
    pre = [_do_single_on(b0, q0, tolerance), _do_single_on(b1, q1, tolerance)]
    a0, a1 = kak.single_qubit_operations_after
    post = [_do_single_on(a0, q0, tolerance), _do_single_on(a1, q1, tolerance)]

    return list(ops.flatten_op_tree([
        pre,
        _non_local_part(q0,
                        q1,
                        kak.interaction_coefficients,
                        allow_partial_czs,
                        tolerance),
        post,
    ]))


def _is_trivial_angle(rad: float, tolerance: float) -> bool:
    """Tests if a circuit for an operator exp(i*rad*XX) (or YY, or ZZ) can
    be performed with a whole CZ.

    Args:
        rad: The angle in radians, assumed to be in the range [-pi/4, pi/4]
    """
    return abs(rad) < tolerance or abs(abs(rad) - np.pi / 4) < tolerance


def _parity_interaction(q0: ops.QubitId,
                        q1: ops.QubitId,
                        rads: float,
                        tolerance: float,
                        gate: Optional[ops.Gate] = None):
    """Yields a ZZ interaction framed by the given operation."""
    if abs(rads) < tolerance:
        return

    h = rads * -2 / np.pi
    if gate is not None:
        g = cast(ops.Gate, gate)
        yield g.on(q0), g.on(q1)

    # If rads is ±pi/4 radians within tolerance, single full-CZ suffices.
    if _is_trivial_angle(rads, tolerance):
        yield ops.CZ.on(q0, q1)
    else:
        yield ops.CZ(q0, q1) ** (-2 * h)

    yield ops.Z(q0)**h
    yield ops.Z(q1)**h
    if gate is not None:
        g = protocols.inverse(gate)
        yield g.on(q0), g.on(q1)


def _do_single_on(u: np.ndarray, q: ops.QubitId, tolerance: float=1e-8):
    for gate in decompositions.single_qubit_matrix_to_gates(u, tolerance):
        yield gate(q)


def _non_local_part(q0: ops.QubitId,
                    q1: ops.QubitId,
                    interaction_coefficients: Tuple[float, float, float],
                    allow_partial_czs: bool,
                    tolerance: float = 1e-8):
    """Yields non-local operation of KAK decomposition."""

    x, y, z = interaction_coefficients

    if (allow_partial_czs or
        all(_is_trivial_angle(e, tolerance) for e in [x, y, z])):
        return [
            _parity_interaction(q0, q1, x, tolerance, ops.Y**-0.5),
            _parity_interaction(q0, q1, y, tolerance, ops.X**0.5),
            _parity_interaction(q0, q1, z, tolerance)
        ]

    if abs(z) >= tolerance:
        return _xx_yy_zz_interaction_via_full_czs(q0, q1, x, y, z)

    if y >= tolerance:
        return _xx_yy_interaction_via_full_czs(q0, q1, x, y)

    return _xx_interaction_via_full_czs(q0, q1, x)
