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

import functools
import operator
from collections.abc import Iterable, Sequence

import numpy as np
import pytest

import cirq

# Sizes for `cirq.PauliString` operations, which are linear in the number of qubits.
SPARSE_SIZES = [10, 100, 500]

# `cirq.DensePauliString` operations are vectorized, so larger sizes are needed before the
# per-qubit cost shows over the fixed numpy call overhead.
DENSE_SIZES = [10, 100, 1000, 10000]

# Clifford conjugation costs hundreds of microseconds per operation.
CLIFFORD_SIZES = [10, 50, 100, pytest.param(200, marks=pytest.mark.slow)]
NUM_TERMS = [20, 100, 500]
MATRIX_SIZES = [8, 12, 16, 20]

# `cirq.PauliSum.sparse_matrix` takes seconds at 20 qubits.
SUM_MATRIX_SIZES = [8, 12, 16, pytest.param(20, marks=pytest.mark.slow)]

# Rounds for `benchmark.pedantic`, used where every round needs a fresh input object.
PEDANTIC_ROUNDS = 100

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


def _commuting_pauli_sum(qubits: Sequence[cirq.Qid], num_terms: int) -> cirq.PauliSum:
    """Returns a sum of `num_terms` distinct 3-local Z-only terms, which commute pairwise."""
    n = len(qubits)
    if n < 4:
        raise ValueError("At least 4 qubits are needed for distinct terms.")
    terms = [
        cirq.PauliString(
            qubit_pauli_map={
                qubits[i % n]: cirq.Z,
                qubits[(i + 1) % n]: cirq.Z,
                qubits[(i + 3) % n]: cirq.Z,
            },
            coefficient=0.25 * ((i % 5) + 1),
        )
        for i in range(num_terms)
    ]
    return cirq.PauliSum.from_pauli_strings(terms)


def _clifford_ladder(qubits: Sequence[cirq.Qid], as_clifford_gates: bool) -> list[cirq.Operation]:
    """Returns a layer of H on every qubit followed by a CNOT ladder.

    With `as_clifford_gates`, the operations use `cirq.CliffordGate` instances, which lets
    the conjugation code skip the per-operation conversion to a Clifford tableau.
    """
    h = cirq.CliffordGate.H if as_clifford_gates else cirq.H
    cnot = cirq.CliffordGate.CNOT if as_clifford_gates else cirq.CNOT
    return [h(q) for q in qubits] + [cnot(a, b) for a, b in zip(qubits, qubits[1:])]


def _plus_state(qubits: Sequence[cirq.Qid]) -> np.ndarray:
    """Returns the |+>^n state vec,; any Pauli string containing Y or Z has expectation 0 on it."""
    return cirq.Circuit(cirq.H.on_each(*qubits)).final_state_vector(dtype=np.complex64)


class TestPauliStringConstructionAndConversion:
    """Construction, qubit remapping, and basis conversion on `cirq.PauliString`."""

    group = "pauli_string_construction"

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_from_qubit_pauli_map(self, benchmark, num_qubits: int) -> None:
        """Construction via the `qubit_pauli_map` keyword argument.

        The mapping is stored without copying, so this measures only the per-Pauli type
        validation done in debug mode (see `cirq.with_debug`), which every internally
        constructed Pauli string pays.
        """
        pauli_map = _pauli_map(cirq.LineQubit.range(num_qubits))
        pauli_string = benchmark(cirq.PauliString, qubit_pauli_map=pauli_map)
        assert len(pauli_string) == num_qubits

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_from_contents_dict(self, benchmark, num_qubits: int) -> None:
        """Construction via a positional dictionary, which goes through the general path."""
        pauli_map = _pauli_map(cirq.LineQubit.range(num_qubits))
        pauli_string = benchmark(cirq.PauliString, pauli_map)
        assert len(pauli_string) == num_qubits

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_from_operations(self, benchmark, num_qubits: int) -> None:
        """Construction from a sequence of single-qubit Pauli operations."""
        ops = _pauli_ops(cirq.LineQubit.range(num_qubits))
        pauli_string = benchmark(cirq.PauliString, ops)
        assert len(pauli_string) == num_qubits

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_from_chained_multiplication(self, benchmark, num_qubits: int) -> None:
        """Construction by repeated `*`, as in `cirq.X(a) * cirq.Y(b) * cirq.Z(c)`."""
        ops = _pauli_ops(cirq.LineQubit.range(num_qubits))
        pauli_string = benchmark(functools.reduce, operator.mul, ops)
        assert len(pauli_string) == num_qubits

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_with_qubits(self, benchmark, num_qubits: int) -> None:
        pauli_string = _pauli_string(cirq.LineQubit.range(num_qubits))
        new_qubits = cirq.LineQubit.range(num_qubits, 2 * num_qubits)
        remapped = benchmark(pauli_string.with_qubits, *new_qubits)
        assert len(remapped) == num_qubits

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_map_qubits(self, benchmark, num_qubits: int) -> None:
        old_qubits = cirq.LineQubit.range(num_qubits)
        new_qubits = cirq.LineQubit.range(num_qubits, 2 * num_qubits)
        pauli_string = _pauli_string(old_qubits)
        qubit_map = dict(zip(old_qubits, new_qubits))
        remapped = benchmark(pauli_string.map_qubits, qubit_map)
        assert len(remapped) == num_qubits

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_to_dense(self, benchmark, num_qubits: int) -> None:
        """Conversion of a string acting on every other qubit to a dense string over all."""
        qubits = cirq.LineQubit.range(num_qubits)
        pauli_string = _pauli_string(qubits[::2])
        dense = benchmark(pauli_string.dense, qubits)
        assert len(dense) == num_qubits

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_to_z_basis_ops(self, benchmark, num_qubits: int) -> None:
        pauli_string = _pauli_string(cirq.LineQubit.range(num_qubits))
        ops = benchmark(lambda: list(pauli_string.to_z_basis_ops()))
        assert len(ops) == num_qubits


class TestPauliStringAlgebra:
    """Multiplication, commutation, equality, and hashing of `cirq.PauliString`."""

    group = "pauli_string_algebra"

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_multiplication_distinct(self, benchmark, num_qubits: int) -> None:
        """Multiplying two strings with identical support and distinct Paulis on every qubit."""
        qubits = cirq.LineQubit.range(num_qubits)
        left = _pauli_string(qubits, offset=0)
        right = _pauli_string(qubits, offset=1)
        product = benchmark(operator.mul, left, right)
        assert len(product) == num_qubits

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_multiplication_partial_cancellation(
        self, benchmark, num_qubits: int
    ) -> None:
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
        """In-place multiplication, with a fresh `cirq.MutablePauliString` for every round."""
        qubits = cirq.LineQubit.range(num_qubits)
        right = _pauli_string(qubits, offset=1)
        result = benchmark.pedantic(
            cirq.MutablePauliString.inplace_left_multiply_by,
            setup=lambda: ((_pauli_string(qubits).mutable_copy(), right), {}),
            rounds=PEDANTIC_ROUNDS,
        )
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

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_equality(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        left = _pauli_string(qubits)
        right = _pauli_string(qubits)
        assert benchmark(operator.eq, left, right)

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_equal_up_to_coefficient(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        left = _pauli_string(qubits)
        right = _pauli_string(qubits).with_coefficient(-1.0)
        assert benchmark(left.equal_up_to_coefficient, right)

    @pytest.mark.parametrize("num_qubits", SPARSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_hash(self, benchmark, num_qubits: int) -> None:
        """First hash of a string; `cirq.PauliString` caches the value for later calls."""
        qubits = cirq.LineQubit.range(num_qubits)
        value = benchmark.pedantic(
            hash, setup=lambda: ((_pauli_string(qubits),), {}), rounds=PEDANTIC_ROUNDS
        )
        assert isinstance(value, int)


class TestDensePauliString:
    """Construction, algebra, commutation, and sparse conversion on `cirq.DensePauliString`."""

    group = "dense_pauli_string"

    @pytest.mark.parametrize("num_qubits", DENSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_dense_pauli_string_from_str(self, benchmark, num_qubits: int) -> None:
        pauli_spec = "".join("IXYZ"[i % 4] for i in range(num_qubits))
        dps = benchmark(cirq.DensePauliString, pauli_spec)
        assert len(dps) == num_qubits

    @pytest.mark.parametrize("num_qubits", DENSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_dense_pauli_string_multiplication(self, benchmark, num_qubits: int) -> None:
        left = cirq.DensePauliString("".join("XYZ"[i % 3] for i in range(num_qubits)))
        right = cirq.DensePauliString("".join("YZX"[i % 3] for i in range(num_qubits)))
        product = benchmark(operator.mul, left, right)
        assert len(product) == num_qubits

    @pytest.mark.parametrize("num_qubits", DENSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_dense_pauli_string_commutes(self, benchmark, num_qubits: int) -> None:
        left = cirq.DensePauliString("".join("XYZ"[i % 3] for i in range(num_qubits)))
        right = cirq.DensePauliString("".join("YZX"[i % 3] for i in range(num_qubits)))
        commutes = benchmark(cirq.commutes, left, right)
        assert commutes == (num_qubits % 2 == 0)

    @pytest.mark.parametrize("num_qubits", DENSE_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_dense_pauli_string_to_sparse(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        dps = cirq.DensePauliString("".join("IXYZ"[i % 4] for i in range(num_qubits)))
        sparse_ps = benchmark(dps.sparse, qubits)
        assert len(sparse_ps) == num_qubits - (num_qubits + 3) // 4


class TestCliffordConjugation:
    """Propagating Pauli strings through Clifford operations."""

    group = "pauli_clifford_conjugation"

    @pytest.mark.parametrize("as_clifford_gates", [False, True], ids=["raw", "clifford_gate"])
    @pytest.mark.parametrize("num_qubits", CLIFFORD_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_conjugated_by(
        self, benchmark, num_qubits: int, as_clifford_gates: bool
    ) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        pauli_string = _pauli_string(qubits)
        clifford_ops = _clifford_ladder(qubits, as_clifford_gates)
        expected = pauli_string.mutable_copy().inplace_before(clifford_ops).frozen()
        conjugated = benchmark(pauli_string.conjugated_by, clifford_ops)
        assert conjugated == expected

    @pytest.mark.parametrize("as_clifford_gates", [False, True], ids=["raw", "clifford_gate"])
    @pytest.mark.parametrize("num_qubits", CLIFFORD_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_mutable_pauli_string_inplace_before(
        self, benchmark, num_qubits: int, as_clifford_gates: bool
    ) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        pauli_string = _pauli_string(qubits)
        clifford_ops = _clifford_ladder(qubits, as_clifford_gates)

        expected = pauli_string.conjugated_by(clifford_ops)

        def _f() -> cirq.MutablePauliString:
            return pauli_string.mutable_copy().inplace_before(clifford_ops)

        assert benchmark(_f).frozen() == expected


class TestPauliSumAlgebra:
    """Construction and arithmetic on `cirq.PauliSum` and `cirq.PauliSumExponential`."""

    group = "pauli_sum_algebra"

    @pytest.mark.parametrize("num_terms", NUM_TERMS)
    @pytest.mark.benchmark(group=group)
    def test_pauli_sum_from_pauli_strings(self, benchmark, num_terms: int) -> None:
        qubits = cirq.LineQubit.range(num_terms // 2)
        terms = _pauli_sum_terms(qubits, num_terms)
        pauli_sum = benchmark(cirq.PauliSum.from_pauli_strings, terms)
        assert len(pauli_sum) == num_terms

    @pytest.mark.parametrize("num_terms", NUM_TERMS)
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

    @pytest.mark.parametrize("num_terms", [10, 40, 80])
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

    @pytest.mark.parametrize("num_terms", [10, 50, 200])
    @pytest.mark.benchmark(group=group)
    def test_pauli_sum_exponential_construction(self, benchmark, num_terms: int) -> None:
        """Construction checks that all pairs of terms commute, which is quadratic in terms."""
        qubits = cirq.LineQubit.range(num_terms)
        pauli_sum = _commuting_pauli_sum(qubits, num_terms)
        exponential = benchmark(cirq.PauliSumExponential, pauli_sum, 0.5)
        assert len(exponential.qubits) == num_terms


class TestMatricesAndExpectations:
    """Sparse matrices and state-vector expectation values of Pauli observables."""

    group = "pauli_matrices_and_expectations"

    @pytest.mark.parametrize("num_qubits", MATRIX_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_sparse_matrix(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        pauli_string = _pauli_string(qubits)
        matrix = benchmark(pauli_string.sparse_matrix, qubits)
        assert matrix.shape == (1 << num_qubits, 1 << num_qubits)

    @pytest.mark.parametrize("num_qubits", SUM_MATRIX_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_sum_sparse_matrix(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        pauli_sum = cirq.PauliSum.from_pauli_strings(_pauli_sum_terms(qubits, 2 * num_qubits))
        matrix = benchmark(pauli_sum.sparse_matrix, qubits)
        assert matrix.shape == (1 << num_qubits, 1 << num_qubits)

    @pytest.mark.parametrize("num_qubits", MATRIX_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_expectation_from_state_vector(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        state = _plus_state(qubits)
        observable = _pauli_string(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        value = benchmark(observable.expectation_from_state_vector, state, qubit_map)
        assert abs(value) < 1e-5

    @pytest.mark.parametrize("num_qubits", MATRIX_SIZES)
    @pytest.mark.benchmark(group=group)
    def test_pauli_sum_expectation_from_state_vector(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        state = _plus_state(qubits)
        observable = cirq.PauliSum.from_pauli_strings(_pauli_sum_terms(qubits, 2 * num_qubits))
        qubit_map = {q: i for i, q in enumerate(qubits)}
        value = benchmark(observable.expectation_from_state_vector, state, qubit_map)
        assert abs(value) < 1e-5
