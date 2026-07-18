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


import pytest

import cirq

qubits_per_layer = [
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

all_qubits = set()
for qubits in qubits_per_layer:
    all_qubits.update(qubits)


layers_grid_qubit_pairs = [
    [(cirq.GridQubit(5, 9), cirq.GridQubit(5, 10)), (cirq.GridQubit(6, 8), cirq.GridQubit(6, 9))],
    [
        (cirq.GridQubit(5, 10), cirq.GridQubit(6, 10)),
        (cirq.GridQubit(6, 9), cirq.GridQubit(7, 9)),
        (cirq.GridQubit(7, 8), cirq.GridQubit(8, 8)),
    ],
    [(cirq.GridQubit(6, 9), cirq.GridQubit(6, 10)), (cirq.GridQubit(7, 8), cirq.GridQubit(7, 9))],
    [(cirq.GridQubit(5, 9), cirq.GridQubit(6, 9)), (cirq.GridQubit(6, 8), cirq.GridQubit(7, 8))],
]


def _make_fake_trotter_circuit(num_cycles: int) -> cirq.Circuit:
    moments = []
    for layer, qubits in zip(layers_grid_qubit_pairs, qubits_per_layer):
        moments.append(cirq.Moment((cirq.Y**0.3)(qubit)) for qubit in qubits)
        moments.append(cirq.Moment(cirq.CZ(*pair)) for pair in layer)
    return cirq.Circuit.from_moments(*moments) * num_cycles + cirq.Moment(cirq.M(*all_qubits))


PARAMETERS = ((5, 10), (10, 20), (50, 100))


@pytest.mark.parametrize(["num_cycles", "num_circuits"], PARAMETERS)
@pytest.mark.benchmark(group="merge_single_qubit_gates_to_phxz")
def test_merge_single_qubit_gates_to_phxz_symbolized_sweep(
    benchmark, num_cycles: int, num_circuits: int
) -> None:
    # Setup (not benchmarked)
    circuit = _make_fake_trotter_circuit(num_cycles)
    circuit_sweep, sweep_params = cirq.transformers.gauge_compiling.CZGaugeTransformer.as_sweep(
        circuit, N=num_circuits
    )

    # Benchmark only the merge part
    benchmark(cirq.merge_single_qubit_gates_to_phxz_symbolized, circuit_sweep, sweep=sweep_params)


@pytest.mark.parametrize(["num_cycles", "num_circuits"], PARAMETERS)
@pytest.mark.benchmark(group="merge_single_qubit_gates_to_phxz")
def test_merge_single_qubit_moments_to_phxz_batch(
    benchmark, num_cycles: int, num_circuits: int
) -> None:
    # Setup (not benchmarked)
    circuit = _make_fake_trotter_circuit(num_cycles)
    circuits_batch = [
        cirq.transformers.gauge_compiling.cz_gauge.CZGaugeTransformer(circuit)
        for _ in range(num_circuits)
    ]

    # Benchmark only the merge part on the batch
    def run_merge_batch():
        for c in circuits_batch:
            _ = cirq.merge_single_qubit_moments_to_phxz(c)

    benchmark(run_merge_batch)
