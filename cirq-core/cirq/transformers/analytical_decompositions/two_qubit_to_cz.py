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

"""Utility methods for decomposing two-qubit unitaries into CZ gates."""

from typing import Iterable, List, Sequence, Tuple, Optional, cast, TYPE_CHECKING

import numpy as np

from cirq.linalg import predicates
from cirq.linalg.decompositions import num_cnots_required, extract_right_diag

from cirq import ops, linalg, protocols, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phased_x_and_z
from cirq.transformers.eject_z import eject_z
from cirq.transformers.eject_phased_paulis import eject_phased_paulis

if TYPE_CHECKING:
    import cirq


def _remove_partial_czs_or_fail(
    operations: Iterable['cirq.Operation'], atol: float
) -> List['cirq.Operation']:
    result = []
    for op in operations:
        if isinstance(op.gate, ops.CZPowGate):
            t = op.gate.exponent % 2  # CZ^t is periodic with period 2.
            if t < atol:
                continue  # Identity.
            elif abs(t - 1) < atol:
                result.append(ops.CZ(*op.qubits))  # Was either CZ or CZ**-1.
            else:
                raise ValueError(f'CZ^t is not allowed for t={t}')
        else:
            result.append(op)
    return result


def two_qubit_matrix_to_cz_operations(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    mat: np.ndarray,
    allow_partial_czs: bool,
    atol: float = 1e-8,
    clean_operations: bool = True,
) -> List[ops.Operation]:
    """Decomposes a two-qubit operation into Z/XY/CZ gates.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: Defines the operation to apply to the pair of qubits.
        allow_partial_czs: Enables the use of Partial-CZ gates.
        atol: A limit on the amount of absolute error introduced by the
            construction.
        clean_operations: Enables optimizing resulting operation list by
            merging operations and ejecting phased Paulis and Z operations.

    Returns:
        A list of operations implementing the matrix.

    Raises:
        ValueError: If allow_partial_czs=False and the matrix requires partial CZs.
    """
    kak = linalg.kak_decomposition(mat, atol=atol)
    operations = _kak_decomposition_to_operations(q0, q1, kak, allow_partial_czs, atol=atol)
    if clean_operations:
        if not allow_partial_czs:
            # CZ^t is not allowed for any $t$ except $t=1$.
            return _remove_partial_czs_or_fail(cleanup_operations(operations), atol=atol)
        return cleanup_operations(operations)
    return operations


def two_qubit_matrix_to_diagonal_and_cz_operations(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    mat: np.ndarray,
    allow_partial_czs: bool = False,
    atol: float = 1e-8,
    clean_operations: bool = True,
) -> Tuple[np.ndarray, List['cirq.Operation']]:
    """Decomposes a 2-qubit unitary to a diagonal and the remaining operations.

    For a 2-qubit unitary V, return ops, a list of operations and
    D diagonal unitary, so that:
        V = cirq.Circuit(ops) @ D

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: the input unitary
        allow_partial_czs: Enables the use of Partial-CZ gates.
        atol: A limit on the amount of absolute error introduced by the
            construction.
        clean_operations: Enables optimizing resulting operation list by
            merging operations and ejecting phased Paulis and Z operations.
    Returns:
        tuple(ops,D): operations `ops`, and the diagonal `D`
    """
    if predicates.is_diagonal(mat, atol=atol):
        return mat, []

    if num_cnots_required(mat) == 3:
        right_diag = extract_right_diag(mat)
        two_cnot_unitary = mat @ right_diag
        # note that this implies that two_cnot_unitary @ d = mat
        return right_diag.conj().T, two_qubit_matrix_to_cz_operations(
            q0,
            q1,
            two_cnot_unitary,
            allow_partial_czs=allow_partial_czs,
            atol=atol,
            clean_operations=clean_operations,
        )

    return np.eye(4), two_qubit_matrix_to_cz_operations(
        q0,
        q1,
        mat,
        allow_partial_czs=allow_partial_czs,
        atol=atol,
        clean_operations=clean_operations,
    )


def _xx_interaction_via_full_czs(q0: 'cirq.Qid', q1: 'cirq.Qid', x: float):
    a = x * -2 / np.pi
    yield ops.H(q1)
    yield ops.CZ(q0, q1)
    yield ops.X(q0) ** a
    yield ops.CZ(q0, q1)
    yield ops.H(q1)


def _xx_yy_interaction_via_full_czs(q0: 'cirq.Qid', q1: 'cirq.Qid', x: float, y: float):
    a = x * -2 / np.pi
    b = y * -2 / np.pi
    yield ops.X(q0) ** 0.5
    yield ops.H(q1)
    yield ops.CZ(q0, q1)
    yield ops.H(q1)
    yield ops.X(q0) ** a
    yield ops.Y(q1) ** b
    yield ops.H(q1)
    yield ops.CZ(q0, q1)
    yield ops.H(q1)
    yield ops.X(q0) ** -0.5


def _xx_yy_zz_interaction_via_full_czs(
    q0: 'cirq.Qid', q1: 'cirq.Qid', x: float, y: float, z: float
):
    a = x * -2 / np.pi + 0.5
    b = y * -2 / np.pi + 0.5
    c = z * -2 / np.pi + 0.5
    yield ops.X(q0) ** 0.5
    yield ops.H(q1)
    yield ops.CZ(q0, q1)
    yield ops.H(q1)
    yield ops.X(q0) ** a
    yield ops.Y(q1) ** b
    yield ops.H.on(q0)
    yield ops.CZ(q1, q0)
    yield ops.H(q0)
    yield ops.X(q1) ** -0.5
    yield ops.Z(q1) ** c
    yield ops.H(q1)
    yield ops.CZ(q0, q1)
    yield ops.H(q1)


def cleanup_operations(operations: Sequence[ops.Operation]):
    circuit = circuits.Circuit(operations)
    circuit = merge_single_qubit_gates_to_phased_x_and_z(circuit)
    circuit = eject_phased_paulis(circuit)
    circuit = eject_z(circuit)
    circuit = circuits.Circuit(circuit.all_operations(), strategy=circuits.InsertStrategy.EARLIEST)
    return list(circuit.all_operations())


def _kak_decomposition_to_operations(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    kak: linalg.KakDecomposition,
    allow_partial_czs: bool,
    atol: float = 1e-8,
) -> List[ops.Operation]:
    """Assumes that the decomposition is canonical."""
    b0, b1 = kak.single_qubit_operations_before
    pre = [_do_single_on(b0, q0, atol=atol), _do_single_on(b1, q1, atol=atol)]
    a0, a1 = kak.single_qubit_operations_after
    post = [_do_single_on(a0, q0, atol=atol), _do_single_on(a1, q1, atol=atol)]

    return list(
        cast(
            Iterable[ops.Operation],
            ops.flatten_op_tree(
                [
                    pre,
                    _non_local_part(
                        q0, q1, kak.interaction_coefficients, allow_partial_czs, atol=atol
                    ),
                    post,
                ]
            ),
        )
    )


def _is_trivial_angle(rad: float, atol: float) -> bool:
    """Tests if a circuit for an operator exp(i*rad*XX) (or YY, or ZZ) can
    be performed with a whole CZ.

    Args:
        rad: The angle in radians, assumed to be in the range [-pi/4, pi/4]
        atol: The absolute tolerance with which to make this judgement.
    """
    return abs(rad) < atol or abs(abs(rad) - np.pi / 4) < atol


def _parity_interaction(
    q0: 'cirq.Qid', q1: 'cirq.Qid', rads: float, atol: float, gate: Optional[ops.Gate] = None
):
    """Yields a ZZ interaction framed by the given operation."""
    if abs(rads) < atol:
        return

    h = rads * -2 / np.pi
    if gate is not None:
        yield gate.on(q0), gate.on(q1)

    # If rads is ±pi/4 radians within tolerance, single full-CZ suffices.
    if _is_trivial_angle(rads, atol):
        yield ops.CZ.on(q0, q1)
    else:
        yield ops.CZ(q0, q1) ** (-2 * h)

    yield ops.Z(q0) ** h
    yield ops.Z(q1) ** h
    if gate is not None:
        g = protocols.inverse(gate)
        yield g.on(q0), g.on(q1)


def _do_single_on(u: np.ndarray, q: 'cirq.Qid', atol: float = 1e-8):
    for gate in single_qubit_decompositions.single_qubit_matrix_to_gates(u, atol):
        yield gate(q)


def _non_local_part(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    interaction_coefficients: Tuple[float, float, float],
    allow_partial_czs: bool,
    atol: float = 1e-8,
):
    """Yields non-local operation of KAK decomposition."""

    x, y, z = interaction_coefficients

    if allow_partial_czs or all(_is_trivial_angle(e, atol) for e in [x, y, z]):
        return [
            _parity_interaction(q0, q1, x, atol, ops.Y**-0.5),
            _parity_interaction(q0, q1, y, atol, ops.X**0.5),
            _parity_interaction(q0, q1, z, atol),
        ]

    if abs(z) >= atol:
        return _xx_yy_zz_interaction_via_full_czs(q0, q1, x, y, z)

    if y >= atol:
        return _xx_yy_interaction_via_full_czs(q0, q1, x, y)

    return _xx_interaction_via_full_czs(q0, q1, x)
