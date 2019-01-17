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

from typing import Tuple

import numpy as np

PAULI_BASIS = (
    np.eye(2),
    np.array([[0., 1.], [1., 0.]]),
    np.array([[0., -1j], [1j, 0.]]),
    np.diag([1., -1]),
)


def hilbert_schmidt(m1: np.ndarray, m2: np.ndarray) -> complex:
    """Computes Hilbert-Schmidt inner product of two matrices.

    Linear in second argument.
    """
    return np.trace(np.dot(np.conjugate(np.transpose(m1)), m2))


def expand_in_basis(m: np.ndarray, basis: Tuple[np.ndarray, ...]) -> np.ndarray:
    """Computes coefficients of an expansion of m in basis.

    Basis elements are rows in basis.

    We require that basis be orthogonal w.r.t. the Hilbert-Schmidt inner
    product. We do not require that basis be orthonormal. Note that Pauli
    basis (I, X, Y, Z) is orthogonal, but not orthonormal.
    """
    return np.array(
        [hilbert_schmidt(b, m) / hilbert_schmidt(b, b) for b in basis])


def reconstruct_from_expansion(expansion: np.ndarray,
                               basis: Tuple[np.ndarray, ...]) -> np.ndarray:
    return np.sum(c * b for c, b in zip(expansion, basis))


def operator_power(pauli_coefficients: np.ndarray,
                   exponent: int) -> np.ndarray:
    """Computes non-negative integer power of single-qubit linear operator.

    Both input and output operators are represented using their expansion in
    the Pauli basis.

    Correctness of the formulas below follows from the binomial expansion using
    the fact that for any normalized real vector (b, c, d) and any non-negative
    integer k:

        [bX + cY + dZ]^(2k) = I

    """
    if exponent == 0:
        return np.array([1., 0., 0., 0.])

    a, b, c, d = pauli_coefficients

    v = np.sqrt(b*b + c*c + d*d)
    s = np.power(a + v, exponent)
    t = np.power(a - v, exponent)

    ci = (s + t) / 2
    if s == t:
        cxyz = exponent * np.power(a, exponent - 1)
    else:
        cxyz = (s - t) / 2
        cxyz = cxyz / v

    return np.array([ci, cxyz * b, cxyz * c, cxyz * d])
