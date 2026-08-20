# Copyright 2025 The Cirq Developers
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

import itertools

import pytest

import cirq

QUBITS_PER_LAYER = [
    [cirq.GridQubit(5, 9), cirq.GridQubit(6, 9), cirq.GridQubit(5, 10), cirq.GridQubit(6, 8)],
    [
        cirq.GridQubit(8, 8),
        cirq.GridQubit(6, 9),
        cirq.GridQubit(7, 9),
        cirq.GridQubit(5, 10),
        cirq.GridQubit(6, 10),
        cirq.GridQubit(7, 8),
    ],
    [cirq.GridQubit(6, 9), cirq.GridQubit(7, 9), cirq.GridQubit(6, 10), cirq.GridQubit(7, 8)],
    [cirq.GridQubit(5, 9), cirq.GridQubit(6, 9), cirq.GridQubit(6, 8), cirq.GridQubit(7, 8)],
]

LAYERS_GRID_QUBIT_PAIRS = [
    [(cirq.GridQubit(5, 9), cirq.GridQubit(5, 10)), (cirq.GridQubit(6, 8), cirq.GridQubit(6, 9))],
    [
        (cirq.GridQubit(5, 10), cirq.GridQubit(6, 10)),
        (cirq.GridQubit(6, 9), cirq.GridQubit(7, 9)),
        (cirq.GridQubit(7, 8), cirq.GridQubit(8, 8)),
    ],
    [(cirq.GridQubit(6, 9), cirq.GridQubit(6, 10)), (cirq.GridQubit(7, 8), cirq.GridQubit(7, 9))],
    [(cirq.GridQubit(5, 9), cirq.GridQubit(6, 9)), (cirq.GridQubit(6, 8), cirq.GridQubit(7, 8))],
]


def make_fake_trotter_circuit(num_cycles: int):
    all_qubits = sorted(set(itertools.chain.from_iterable(QUBITS_PER_LAYER)))
    moments = []
    for layer, qubits in zip(LAYERS_GRID_QUBIT_PAIRS, QUBITS_PER_LAYER):
        moments.append(cirq.Moment((cirq.Y**0.3)(qubit) for qubit in qubits))
        moments.append(cirq.Moment(cirq.CZ(*pair) for pair in layer))
    return cirq.Circuit.from_moments(*moments) * num_cycles + cirq.Moment(cirq.M(*all_qubits))


@pytest.mark.parametrize(
    ["num_cycles", "num_circuits"], [(5, 10), pytest.param(50, 100, marks=pytest.mark.slow)]
)
@pytest.mark.benchmark(group="dynamical_decoupling", max_time=10)
def test_dynamical_decoupling_sweep(benchmark, num_cycles: int, num_circuits: int) -> None:
    def _f(num_cycles: int, num_circuits: int) -> cirq.Circuit:
        circuit = make_fake_trotter_circuit(num_cycles)
        circuit1, sweep1 = cirq.transformers.gauge_compiling.CZGaugeTransformer.as_sweep(
            circuit, N=num_circuits
        )
        circuit2, sweep2 = cirq.merge_single_qubit_gates_to_phxz_symbolized(circuit1, sweep=sweep1)
        assert len(sweep2)
        circuit3 = cirq.add_dynamical_decoupling(circuit2)
        return circuit3

    _ = benchmark(_f, num_cycles=num_cycles, num_circuits=num_circuits)


@pytest.mark.parametrize(
    ["num_cycles", "num_circuits"], [(5, 10), pytest.param(50, 100, marks=pytest.mark.slow)]
)
@pytest.mark.benchmark(group="dynamical_decoupling", max_time=10)
def test_dynamical_decoupling_batch(benchmark, num_cycles: int, num_circuits: int) -> None:
    def _f(num_cycles: int, num_circuits: int) -> list[cirq.Circuit]:
        circuit = make_fake_trotter_circuit(num_cycles)
        circuits_batch1 = [
            cirq.transformers.gauge_compiling.cz_gauge.CZGaugeTransformer(circuit)
            for _ in range(num_circuits)
        ]
        circuits_batch2 = [cirq.merge_single_qubit_moments_to_phxz(c) for c in circuits_batch1]
        circuits_batch3 = [cirq.add_dynamical_decoupling(c) for c in circuits_batch2]
        return circuits_batch3

    _ = benchmark(_f, num_cycles=num_cycles, num_circuits=num_circuits)
