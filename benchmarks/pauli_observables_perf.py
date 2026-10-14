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

"""Performance tests for Pauli observables and channel representations.

The workloads are taken from `docs/build/pauli_observables.ipynb` and
`docs/build/operators.ipynb`.  They cover Pauli string algebra, observable measurement
and the conversions between Kraus, Choi and superoperator representations.
"""

from collections.abc import Iterable

import numpy as np
import pytest

import cirq


def _mark_slow_sizes(sizes: Iterable[int], slow_sizes: set[int]) -> list:
    """Single-parameter variant of the `_mark_slow` helper used elsewhere."""
    size_list = list(sizes)
    if not slow_sizes.issubset(size_list):
        raise ValueError("Unmatched values in slow_sizes")
    return [(pytest.param(s, marks=pytest.mark.slow) if s in slow_sizes else s) for s in size_list]


def _pauli_string(qubits: list[cirq.Qid], offset: int = 0) -> cirq.PauliString:
    """Returns a dense Pauli string cycling through X, Y and Z over the given qubits."""
    paulis = [cirq.X, cirq.Y, cirq.Z]
    return cirq.PauliString({q: paulis[(i + offset) % 3] for i, q in enumerate(qubits)})


class TestPauliStringAlgebra:
    """Construction and arithmetic on Pauli strings."""

    group = "pauli_observables"
    num_qubits = [10, 50, 100, 200, 300, 500]

    @pytest.mark.parametrize("num_qubits", _mark_slow_sizes(num_qubits, {500}))
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_construction(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        pauli_string = benchmark(_pauli_string, qubits)
        assert len(pauli_string.qubits) == num_qubits

    @pytest.mark.parametrize("num_qubits", _mark_slow_sizes(num_qubits, {500}))
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_multiplication(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        left = _pauli_string(qubits, offset=0)
        right = _pauli_string(qubits, offset=1)
        product = benchmark(lambda: left * right)
        assert len(product.qubits) == num_qubits

    @pytest.mark.parametrize("num_qubits", _mark_slow_sizes(num_qubits, {500}))
    @pytest.mark.benchmark(group=group)
    def test_pauli_string_with_qubits(self, benchmark, num_qubits: int) -> None:
        """Remapping a Pauli string onto a fresh register, as shown in the notebook."""
        pauli_string = _pauli_string(cirq.LineQubit.range(num_qubits))
        new_qubits = cirq.LineQubit.range(num_qubits, 2 * num_qubits)
        remapped = benchmark(pauli_string.with_qubits, *new_qubits)
        assert set(remapped.qubits) == set(new_qubits)

    @pytest.mark.parametrize("num_qubits", _mark_slow_sizes(num_qubits, {500}))
    @pytest.mark.benchmark(group=group)
    def test_pauli_sum_construction(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)

        def _f() -> cirq.PauliSum:
            return cirq.PauliSum.from_pauli_strings([cirq.PauliString({q: cirq.Z}) for q in qubits])

        assert len(benchmark(_f)) == num_qubits


class TestExpectationValues:
    """Expectation values of Pauli observables against a simulated state."""

    group = "pauli_observables"
    num_qubits = [4, 8, 12, 16]
    repetitions = 100

    @pytest.mark.parametrize("num_qubits", _mark_slow_sizes(num_qubits, {12, 16}))
    @pytest.mark.benchmark(group=group, max_time=10)
    def test_expectation_from_state_vector(self, benchmark, num_qubits: int) -> None:
        qubits = cirq.LineQubit.range(num_qubits)
        circuit = cirq.Circuit(cirq.H.on_each(*qubits))
        state = circuit.final_state_vector(dtype=np.complex64)
        observable = _pauli_string(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        value = benchmark(observable.expectation_from_state_vector, state, qubit_map)
        assert np.isfinite(abs(value))

    @pytest.mark.parametrize("num_qubits", _mark_slow_sizes(num_qubits, {12, 16}))
    @pytest.mark.benchmark(group=group, max_time=10)
    def test_measure_single_paulistring(self, benchmark, num_qubits: int) -> None:
        """A `PauliMeasurementGate` measuring the whole observable at once."""
        qubits = cirq.LineQubit.range(num_qubits)
        observable = _pauli_string(qubits)
        circuit = cirq.Circuit(
            cirq.H.on_each(*qubits), cirq.measure_single_paulistring(observable, key='m')
        )
        simulator = cirq.Simulator(seed=1234)
        result = benchmark(simulator.run, circuit, repetitions=self.repetitions)
        assert sum(result.histogram(key='m').values()) == self.repetitions

    @pytest.mark.parametrize("num_qubits", [2, 4, 6])
    @pytest.mark.benchmark(group=group, max_time=10)
    def test_pauli_sum_collector(self, benchmark, num_qubits: int) -> None:
        """Energy estimation by sampling each term, as shown in the notebook."""
        qubits = cirq.LineQubit.range(num_qubits)
        circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]))
        observable = _pauli_string(qubits) + 4 * cirq.X(qubits[0])

        def _f() -> float:
            collector = cirq.PauliSumCollector(
                circuit=circuit, observable=observable, samples_per_term=100
            )
            collector.collect(sampler=cirq.Simulator(seed=1234))
            return collector.estimated_energy()

        assert np.isfinite(abs(benchmark(_f)))


def _commuting_pauli_sum(qubits: list[cirq.Qid]) -> cirq.PauliSum:
    """Returns a sum of single qubit Z terms, which is guaranteed to commute.

    `cirq.PauliSumExponential` rejects non-commuting sums, so the dense Pauli strings
    used elsewhere in this file are not valid input for it.
    """
    return cirq.PauliSum.from_pauli_strings([cirq.PauliString({q: cirq.Z}) for q in qubits])


class TestPauliSumExponential:
    """Exponentials of Pauli sums and their dense matrices."""

    group = "pauli_observables"
    num_qubits = [2, 4, 6, 8, 10]

    @pytest.mark.parametrize("num_qubits", _mark_slow_sizes(num_qubits, {10}))
    @pytest.mark.benchmark(group=group, max_time=10)
    def test_pauli_sum_exponential_construction(self, benchmark, num_qubits: int) -> None:
        pauli_sum = _commuting_pauli_sum(cirq.LineQubit.range(num_qubits))
        exponential = benchmark(cirq.PauliSumExponential, pauli_sum, 0.5)
        assert len(exponential.qubits) == num_qubits

    @pytest.mark.parametrize("num_qubits", _mark_slow_sizes(num_qubits, {10}))
    @pytest.mark.benchmark(group=group, max_time=10)
    def test_pauli_sum_exponential_matrix(self, benchmark, num_qubits: int) -> None:
        pauli_sum = _commuting_pauli_sum(cirq.LineQubit.range(num_qubits))
        exponential = cirq.PauliSumExponential(pauli_sum, 0.5)
        matrix = benchmark(exponential.matrix)
        assert matrix.shape == (2**num_qubits, 2**num_qubits)


class TestChannelRepresentations:
    """Conversions between Kraus, Choi and superoperator forms."""

    group = "pauli_observables"
    probabilities = [0.1, 0.5]

    @pytest.mark.parametrize("p", probabilities)
    @pytest.mark.benchmark(group=group)
    def test_kraus(self, benchmark, p: float) -> None:
        channel = cirq.depolarize(p)
        kraus = benchmark(cirq.kraus, channel)
        assert len(kraus) == 4

    @pytest.mark.parametrize("p", probabilities)
    @pytest.mark.benchmark(group=group)
    def test_kraus_to_choi(self, benchmark, p: float) -> None:
        kraus = cirq.kraus(cirq.depolarize(p))
        choi = benchmark(cirq.kraus_to_choi, kraus)
        assert choi.shape == (4, 4)

    @pytest.mark.parametrize("p", probabilities)
    @pytest.mark.benchmark(group=group)
    def test_kraus_to_superoperator(self, benchmark, p: float) -> None:
        kraus = cirq.kraus(cirq.depolarize(p))
        superoperator = benchmark(cirq.kraus_to_superoperator, kraus)
        assert superoperator.shape == (4, 4)
