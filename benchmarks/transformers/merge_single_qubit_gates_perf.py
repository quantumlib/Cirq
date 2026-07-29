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
    trotter_circuit = cirq.Circuit.from_moments(*moments) * num_cycles + cirq.Moment(
        cirq.M(*all_qubits)
    )
    return cirq.transformers.gauge_compiling.cz_gauge.CZGaugeTransformer(trotter_circuit)


@pytest.mark.parametrize(["num_cycles"], [(500,)])
@pytest.mark.benchmark(group="merge_single_qubit_gates", max_time=10)
def test_merge_single_qubit_gates_to_phxz(benchmark, num_cycles: int) -> None:
    circuit = make_fake_trotter_circuit(num_cycles)
    benchmark(cirq.merge_single_qubit_gates_to_phxz, circuit)


@pytest.mark.parametrize(["num_cycles"], [(1000,)])
@pytest.mark.benchmark(group="merge_single_qubit_gates", max_time=10)
def test_merge_single_qubit_moments_to_phxz_batch(benchmark, num_cycles: int) -> None:
    circuit = make_fake_trotter_circuit(num_cycles)
    benchmark(cirq.merge_single_qubit_moments_to_phxz, circuit)
