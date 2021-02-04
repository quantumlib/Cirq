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

"""Utilities for manipulating linear operators as elements of vector space."""

from typing import Dict, Tuple

import numpy as np

from cirq import value
from cirq._doc import document

PAULI_BASIS = {
    'I': np.eye(2),
    'X': np.array([[0.0, 1.0], [1.0, 0.0]]),
    'Y': np.array([[0.0, -1j], [1j, 0.0]]),
    'Z': np.diag([1.0, -1]),
}
document(PAULI_BASIS, """The four Pauli matrices (including identity) keyed by character.""")


def kron_bases(*bases: Dict[str, np.ndarray], repeat: int = 1) -> Dict[str, np.ndarray]:
    """Creates tensor product of bases."""
    product_basis = {'': 1}
    for basis in bases * repeat:
        product_basis = {
            name1 + name2: np.kron(matrix1, matrix2)
            for name1, matrix1 in product_basis.items()
            for name2, matrix2 in basis.items()
        }
    return product_basis


def hilbert_schmidt_inner_product(m1: np.ndarray, m2: np.ndarray) -> complex:
    """Computes Hilbert-Schmidt inner product of two matrices.

    Linear in second argument.
    """
    return np.einsum('ij,ij', m1.conj(), m2)


def expand_matrix_in_orthogonal_basis(
    m: np.ndarray,
    basis: Dict[str, np.ndarray],
) -> value.LinearDict[str]:
    """Computes coefficients of expansion of m in basis.

    We require that basis be orthogonal w.r.t. the Hilbert-Schmidt inner
    product. We do not require that basis be orthonormal. Note that Pauli
    basis (I, X, Y, Z) is orthogonal, but not orthonormal.
    """
    return value.LinearDict(
        {
            name: (hilbert_schmidt_inner_product(b, m) / hilbert_schmidt_inner_product(b, b))
            for name, b in basis.items()
        }
    )


def matrix_from_basis_coefficients(
    expansion: value.LinearDict[str], basis: Dict[str, np.ndarray]
) -> np.ndarray:
    """Computes linear combination of basis vectors with given coefficients."""
    some_element = next(iter(basis.values()))
    result = np.zeros_like(some_element, dtype=np.complex128)
    for name, coefficient in expansion.items():
        result += coefficient * basis[name]
    return result


def pow_pauli_combination(
    ai: value.Scalar, ax: value.Scalar, ay: value.Scalar, az: value.Scalar, exponent: int
) -> Tuple[value.Scalar, value.Scalar, value.Scalar, value.Scalar]:
    """Computes non-negative integer power of single-qubit Pauli combination.

    Returns scalar coefficients bi, bx, by, bz such that

        bi I + bx X + by Y + bz Z = (ai I + ax X + ay Y + az Z)^exponent

    Correctness of the formulas below follows from the binomial expansion
    and the fact that for any real or complex vector (ax, ay, az) and any
    non-negative integer k:

         [ax X + ay Y + az Z]^(2k) = (ax^2 + ay^2 + az^2)^k I

    """
    if exponent == 0:
        return 1, 0, 0, 0

    v = np.sqrt(ax * ax + ay * ay + az * az)
    s = np.power(ai + v, exponent)
    t = np.power(ai - v, exponent)

    ci = (s + t) / 2
    if s == t:
        # v is near zero, only one term in binomial expansion survives
        cxyz = exponent * np.power(ai, exponent - 1)
    else:
        # v is non-zero, account for all terms of binomial expansion
        cxyz = (s - t) / 2
        cxyz = cxyz / v

    return ci, cxyz * ax, cxyz * ay, cxyz * az
