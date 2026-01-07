# Copyright 2022 The Cirq Developers
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

"""Performance tests for circuit qubit routing."""

import itertools
from typing import Final

import pytest

import cirq


@pytest.mark.parametrize(
    ["qubits", "depth"], itertools.product([10, 25, 50, 75, 100], [10, 50, 100, 250, 500, 1000])
)
@pytest.mark.benchmark(group="circuit_routing", max_time=10)
def test_circuit_routing(benchmark, qubits: int, depth: int) -> None:
    """Benchmark circuit construction for Rotated Bottom-Z Surface code."""
    op_density: Final = 0.5
    grid_device_size: Final = 10
    gate_domain: Final[dict[cirq.Gate, int]] = {cirq.CNOT: 2, cirq.X: 1}

    circuit = cirq.testing.random_circuit(
        qubits, n_moments=depth, op_density=op_density, gate_domain=gate_domain, random_state=12345
    )
    device = cirq.testing.construct_grid_device(grid_device_size, grid_device_size)
    router = cirq.RouteCQC(device.metadata.nx_graph)
    routed_circuit = benchmark(router, circuit)
    device.validate_circuit(routed_circuit)
