# Copyright 2026 The Cirq Developers
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

"""Performance benchmarks for Cirq Pauli string and Pauli sum primitives."""

import operator
from collections.abc import Iterable, Sequence

import numpy as np
import pytest

import cirq

# Sizes for `cirq.PauliString` operations, which are linear in the number of qubits.
SPARSE_SIZES = [20, 100, 1000]

# `cirq.DensePauliString` operations are vectorized, so larger sizes are needed before the
# per-qubit cost shows over the fixed numpy call overhead.
DENSE_SIZES = [100, 1000, 10000]

# Clifford conjugation costs hundreds of microseconds per operation.
CLIFFORD_SIZES = [20, 100, 1000]

# `PauliSum` addition is O(num_terms), whereas multiplication is O(num_terms^2).
SUM_ADD_TERMS = [100, 1000, 5000]
SUM_MUL_TERMS = [20, 200, pytest.param(500, marks=pytest.mark.slow)]

MATRIX_SIZES = [10, 16, pytest.param(20, marks=pytest.mark.slow)]

PAULIS: tuple[cirq.Pauli, ...] = (cirq.X, cirq.Y, cirq.Z)


def _pauli_map(qubits: Iterable[cirq.Qid], offset: int = 0) -> dict[cirq.Qid, cirq.Pauli]:
    """Returns a mapping from qubits to Paulis cycling through X, Y, and Z."""
    return {q: PAULIS[(i + offset) % 3] for i, q in enumerate(qubits)}


def _pauli_string(qubits: Iterable[cirq.Qid], offset: int = 0) -> cirq.PauliString:
    """Returns a Pauli string cycling through X, Y, and Z over the given qubits."""
    return cirq.PauliString(qubit_pauli_map=_pauli_map(qubits, offset=offset))


def _pauli_ops(qubits: Iterable[cirq.Qid]) -> list[cirq.Operation]:
    """Returns single-qubit Pauli operations cycling through X, Y, and Z."""
    return [PAULIS[i % 3](q) for i, q in enumerate(qubits)]


def _pauli_sum_terms(qubits: Sequence[cirq.Qid], num_terms: int) -> list[cirq.PauliString]:
    """Returns `num_terms` distinct 2-local and 3-local Pauli strings on neighboring qubits.

    Term `i` starts at qubit `i % len(qubits)`. Each pass over the register uses a different
    Pauli pattern, so no two terms have the same qubit-to-Pauli mapping.
    """
    n = len(qubits)
    if n < 4:
        raise ValueError("At least 4 qubits are needed for distinct terms.")
    if num_terms > 3 * n:
        raise ValueError("Only three passes over the qubits give distinct terms.")
    terms: list[cirq.PauliString] = []
    for i in range(num_terms):
        start, pass_index = i % n, i // n
        pattern = (start + pass_index) % 3
        locality = 3 if (start + pass_index) % 4 == 0 else 2
        pauli_map = {qubits[(start + j) % n]: PAULIS[(pattern + j) % 3] for j in range(locality)}
        coefficient = 0.5 if locality == 3 else 1.25
        terms.append(cirq.PauliString(qubit_pauli_map=pauli_map, coefficient=coefficient))
    return terms


def _clifford_ladder(qubits: Sequence[cirq.Qid]) -> list[cirq.Operation]:
    """Returns a layer of H on every qubit followed by a CNOT ladder."""
    return [cirq.H(q) for q in qubits] + [cirq.CNOT(a, b) for a, b in zip(qubits, qubits[1:])]


def _plus_state(num_qubits: int) -> np.ndarray:
    """Returns the |+>^n state vector, on which any Y or Z Pauli string has expectation 0."""
    dim = 1 << num_qubits
    return np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex64)


class TestPauliStringConstructionAndConversion:
    """Construction and basis conversion on `cirq.PauliString`."""

    group = "pauli_string_construction"

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_from_operations(self, benchmark, num_qubits: int) -> None:
        """Construction from a sequence of single-qubit Pauli operations."""
        ops = _pauli_ops(cirq.LineQubit.range(num_qubits))
        pauli_string = benchmark(cirq.PauliString, ops)
        assert len(pauli_string) == num_qubits

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_to_z_basis_ops(self, benchmark, num_qubits: int) -> None:
        pauli_string = _pauli_string(cirq.LineQubit.range(num_qubits))
        ops = benchmark(lambda: list(pauli_string.to_z_basis_ops()))
        assert len(ops) == num_qubits


class TestPauliStringAlgebra:
    """Multiplication and commutation of `cirq.PauliString` and `cirq.MutablePauliString`."""

    group = "pauli_string_algebra"

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_multiplication(self, benchmark, num_qubits: int) -> None:
        """Multiplying strings with partially overlapping support and identity cancellations.

        `right` covers all qubits of `left` plus `num_qubits // 2` more. It repeats the Paulis
        of `left` on the first half of `left`'s qubits (those products cancel to identity) and
        differs from `left` on the second half.
        """
        half = num_qubits // 2
        qubits = cirq.LineQubit.range(num_qubits + half)
        left = _pauli_string(qubits[:num_qubits])
        right_map = _pauli_map(qubits, offset=1)
        right_map.update(_pauli_map(qubits[:half]))
        right = cirq.PauliString(qubit_pauli_map=right_map)
        product = benchmark(operator.mul, left, right)
        assert len(product) == num_qubits

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_mutable_pauli_string_inplace_multiply(self, benchmark, num_qubits: int) -> None:
        """In-place multiplication on `cirq.MutablePauliString`."""
        qubits = cirq.LineQubit.range(num_qubits)
        left = _pauli_string(qubits, offset=0)
        right = _pauli_string(qubits, offset=1)
        result = benchmark(lambda: left.mutable_copy().inplace_left_multiply_by(right))
        assert len(result) == num_qubits

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_commutes(self, benchmark, num_qubits: int) -> None:
        """Commutation of two strings that anticommute on every qubit."""
        qubits = cirq.LineQubit.range(num_qubits)
        left = _pauli_string(qubits, offset=0)
        right = _pauli_string(qubits, offset=1)
        commutes = benchmark(cirq.commutes, left, right)
        assert commutes == (num_qubits % 2 == 0)


class TestDensePauliString:
    """Vectorized multiplication (including phase calculation) on `cirq.DensePauliString`."""

    group = "dense_pauli_string"

    @pytest.mark.parametrize("num_qubits", DENSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_dense_pauli_string_multiplication(self, benchmark, num_qubits: int) -> None:
        left = cirq.DensePauliString("".join("XYZ"[i % 3] for i in range(num_qubits)))
        right = cirq.DensePauliString("".join("YZX"[i % 3] for i in range(num_qubits)))
        product = benchmark(operator.mul, left, right)
        assert len(product) == num_qubits


class TestCliffordConjugation:
    """Propagating Pauli strings through Clifford operations."""

    group = "pauli_clifford_conjugation"

    @pytest.mark.parametrize("num_qubits", CLIFFORD_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_conjugated_by(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        pauli_string = _pauli_string(qubits)
        clifford_ops = _clifford_ladder(qubits)
        conjugated = benchmark(pauli_string.conjugated_by, clifford_ops)
        assert len(conjugated) > 0

    @pytest.mark.parametrize("num_qubits", CLIFFORD_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_mutable_pauli_string_inplace_before(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        pauli_string = _pauli_string(qubits)
        clifford_ops = _clifford_ladder(qubits)
        conjugated = benchmark(lambda: pauli_string.mutable_copy().inplace_before(clifford_ops))
        assert len(conjugated) > 0


class TestPauliSumAlgebra:
    """Arithmetic on `cirq.PauliSum`."""

    group = "pauli_sum_algebra"

    @pytest.mark.parametrize("num_terms", SUM_ADD_TERMS)
    @pytest.mark.benchmark(group=group)
    def test_pauli_sum_addition(self, benchmark, num_terms: int) -> None:
        """Adding two sums that have half of their terms in common."""
        num_qubits = num_terms // 2
        shared = _pauli_sum_terms(cirq.LineQubit.range(num_qubits), num_terms)
        disjoint = _pauli_sum_terms(cirq.LineQubit.range(num_qubits, 2 * num_qubits), num_terms)
        left = cirq.PauliSum.from_pauli_strings(shared)
        right = cirq.PauliSum.from_pauli_strings(
            shared[: num_terms // 2] + disjoint[num_terms // 2 :]
        )
        combined = benchmark(operator.add, left, right)
        assert len(combined) == num_terms + len(disjoint[num_terms // 2 :])

    @pytest.mark.parametrize("num_terms", SUM_MUL_TERMS)
    @pytest.mark.benchmark(group=group)
    def test_pauli_sum_multiplication(self, benchmark, num_terms: int) -> None:
        """Multiplying two sums on overlapping qubits; every pair of terms is multiplied."""
        qubits = cirq.LineQubit.range(max(8, num_terms // 2))
        left = cirq.PauliSum.from_pauli_strings(_pauli_sum_terms(qubits, num_terms))
        right = cirq.PauliSum.from_pauli_strings(
            _pauli_sum_terms(qubits[1:] + qubits[:1], num_terms)
        )
        product = benchmark(operator.mul, left, right)
        assert 0 < len(product) <= num_terms**2


class TestMatricesAndExpectations:
    """Sparse matrices and state-vector expectation values of `cirq.PauliString`."""

    group = "pauli_matrices_and_expectations"

    @pytest.mark.parametrize("num_qubits", MATRIX_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_sparse_matrix(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        pauli_string = _pauli_string(qubits)
        matrix = benchmark(pauli_string.sparse_matrix, qubits)
        assert matrix.shape == (1 << num_qubits, 1 << num_qubits)

    @pytest.mark.parametrize("num_qubits", MATRIX_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_expectation_from_state_vector(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        state = _plus_state(num_qubits)
        observable = _pauli_string(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        value = benchmark(observable.expectation_from_state_vector, state, qubit_map)
        assert abs(value) < 1e-5
