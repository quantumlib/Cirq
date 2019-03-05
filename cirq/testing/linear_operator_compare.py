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

import numpy as np

from cirq.linalg import operator_spaces
from cirq.ops import linear_operator
from cirq.protocols.pauli_expansion import pauli_expansion


def assert_linear_operator_is_consistent(
        op: linear_operator.LinearOperator,
        *,
        atol: float = 1e-9) -> None:
    """Verifies that LinearOperator instance is internally consistent."""
    matrix = op.matrix()
    expansion = pauli_expansion(op, default=None)
    if expansion is None or matrix is None:
        return

    num_qubits = op.num_qubits()
    basis = operator_spaces.kron_bases(
        operator_spaces.PAULI_BASIS, repeat=num_qubits)

    matrix2 = operator_spaces.matrix_from_basis_coefficients(expansion, basis)
    expansion2 = operator_spaces.expand_matrix_in_orthogonal_basis(
        matrix, basis)

    print('matrix\n', matrix)
    print('matrix2\n', matrix2)
    print('expansion', expansion)
    print('expansion2', expansion2)

    assert np.allclose(matrix, matrix2, atol=atol)
    for name in set(expansion.keys()) | set(expansion2.keys()):
        c = expansion.get(name, 0)
        c2 = expansion2.get(name, 0)
        assert abs(c - c2) < atol


def assert_linear_operators_are_equal(
        actual: linear_operator.LinearOperator,
        reference: linear_operator.LinearOperator,
        *,
        atol: float = 1e-9) -> None:
    """Determines whether two linear operators are equal.

    Two linear operators A and B on the same vector space V are equal if for
    every vector v, Av == Bv.

    Alternatively, fix basis of V and denote the matrix of operator D as M(D).
    The above is then equivalent to M(A) == M(B).

    Alternatively, fix basis E1,..., En of the space L(V) of linear operators
    on V and expand A = a1*E1 + ... + an*En and B = b1*E1 + ... bn*En.
    The above is then equivalent to ak == bk for every k=1,...,n.

    We test the two latter conditions since they are more tractable than the
    first. Note that some LinearOperator instances may not have a Pauli basis
    expansion or the matrix, but none should lack both.
    """
    assert_linear_operator_is_consistent(actual, atol=atol)
    assert_linear_operator_is_consistent(reference, atol=atol)

    actual_matrix = actual.matrix()
    reference_matrix = reference.matrix()
    print('actual_matrix\n', actual_matrix)
    print('reference_matrix\n', reference_matrix)
    if actual_matrix is not None and reference_matrix is not None:
        assert actual_matrix.shape == reference_matrix.shape
        assert np.allclose(actual_matrix, reference_matrix, rtol=0, atol=atol)

    actual_expansion = pauli_expansion(actual)
    reference_expansion = pauli_expansion(reference)
    print('actual_expansion', actual_expansion)
    print('reference_expansion', reference_expansion)
    if actual_expansion is not None and reference_expansion is not None:
        for name in (set(actual_expansion.keys()) |
                     set(reference_expansion.keys())):
            actual_coefficient = actual_expansion.get(name, 0)
            reference_coefficient = reference_expansion.get(name, 0)
            assert abs(actual_coefficient - reference_coefficient) < atol

    assert actual_matrix is not None or actual_expansion is not None
    assert reference_matrix is not None or reference_expansion is not None
