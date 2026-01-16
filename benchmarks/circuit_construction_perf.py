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

"""Performance tests for circuit construction."""

import itertools
from collections.abc import Sequence

import pandas
import pytest

import cirq


def rotated_surface_code_memory_z_cycle(
    data_qubits: set[cirq.GridQubit],
    z_measure_qubits: set[cirq.GridQubit],
    x_measure_qubits: set[cirq.GridQubit],
    z_order: Sequence[tuple[int, int]],
    x_order: Sequence[tuple[int, int]],
) -> cirq.Circuit:
    """Constructs a circuit for a single round of rotated memory Z surface code.

    Args:
        data_qubits: data qubits for the surface code patch.
        z_measure_qubits: measure qubits to measure Z stabilizers for surface code patch.
        x_measure_qubits: measure qubits to measure X stabilizers for surface code patch.
        z_order: Specifies the order in which the 2/4 data qubit neighbours of a Z measure qubit
            should be processed.
        x_order: Specifies the order in which the 2/4 data qubit neighbours of a X measure qubit
            should be processed.

    Returns:
        A `cirq.Circuit` for a single round of rotated memory Z surface code cycle.
    """

    circuit = cirq.Circuit()
    circuit += cirq.Moment([cirq.H(q) for q in x_measure_qubits])
    for k in range(4):
        op_list = []
        for measure_qubits, add, is_x in [
            (x_measure_qubits, x_order[k], True),
            (z_measure_qubits, z_order[k], False),
        ]:
            for q_meas in measure_qubits:
                q_data = q_meas + add
                if q_data in data_qubits:
                    op_list.append(cirq.CNOT(q_meas, q_data) if is_x else cirq.CNOT(q_data, q_meas))
        circuit += cirq.Moment(op_list)
    circuit += cirq.Moment([cirq.H(q) for q in x_measure_qubits])
    circuit += cirq.Moment(cirq.measure_each(*x_measure_qubits, *z_measure_qubits))
    return circuit


def surface_code_circuit(
    distance: int, num_rounds: int, moment_by_moment: bool = True
) -> cirq.Circuit:
    """Constructs a rotated memory Z surface code circuit with `distance` and `num_rounds`.

    The circuit has `dxd` data qubits and `d ** 2 - 1` measure qubits, where `d` is the distance
    of surface code. For more details on rotated surface codes and qubit indexing, see figure 13
    https://arxiv.org/abs/1111.4022.

    Args:
        distance: Distance of the surface code.
        num_rounds: Number of error correction rounds for memory Z experiment.
        moment_by_moment: If True, the circuit is constructed moment-by-moment instead of
            operation-by-operation. This is useful to benchmark different circuit construction
            patterns for the same circuit.

    Returns:
        A `cirq.Circuit` for surface code memory Z experiment for `distance` and `num_rounds`.
    """

    def ndrange(*ranges: tuple[int, ...]):
        return itertools.product(*[range(*r) for r in ranges])

    data_qubits = {cirq.q(2 * x + 1, 2 * y + 1) for x, y in ndrange((distance,), (distance,))}
    z_measure_qubits = {
        cirq.q(2 * x, 2 * y) for x, y in ndrange((1, distance), (distance + 1,)) if x % 2 != y % 2
    }
    x_measure_qubits = {
        cirq.q(2 * x, 2 * y) for x, y in ndrange((distance + 1,), (1, distance)) if x % 2 == y % 2
    }
    x_order = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    z_order = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    surface_code_cycle = rotated_surface_code_memory_z_cycle(
        data_qubits, x_measure_qubits, z_measure_qubits, x_order, z_order
    )
    if moment_by_moment:
        return cirq.Circuit(
            surface_code_cycle * num_rounds, cirq.Moment(cirq.measure_each(*data_qubits))
        )
    else:
        return cirq.Circuit(
            [*surface_code_cycle.all_operations()] * num_rounds, cirq.measure_each(*data_qubits)
        )


class TestSurfaceCodeRotatedMemoryZ:
    """Surface Code Rotated Memory-Z Benchmarks."""

    group = "circuit_construction"
    expected = pandas.DataFrame.from_dict(
        {
            3: [64, 369],
            5: [176, 3225],
            7: [344, 12985],
            9: [568, 36369],
            11: [848, 82401],
            13: [1184, 162409],
            15: [1576, 290025],
            17: [2024, 481185],
            19: [2528, 754129],
            21: [3088, 1129401],
            23: [3704, 1629849],
            25: [4376, 2280625],
        },
        columns=["depth", "operation_count"],
        orient="index",
    )

    @pytest.mark.parametrize("distance", expected.index)
    @pytest.mark.benchmark(group=group)
    def test_circuit_construction_moment_by_moment(self, benchmark, distance: int) -> None:
        """Benchmark circuit construction for Rotated Bottom-Z Surface code."""
        circuit = benchmark(
            surface_code_circuit, distance, num_rounds=distance * distance, moment_by_moment=True
        )
        assert len(circuit) == self.expected.depth[distance]

    @pytest.mark.parametrize("distance", expected.index)
    @pytest.mark.benchmark(group=group)
    def test_circuit_construction_operations_by_operation(self, benchmark, distance: int) -> None:
        """Benchmark circuit construction for Rotated Bottom-Z Surface code."""
        circuit = benchmark(
            surface_code_circuit, distance, num_rounds=distance * distance, moment_by_moment=False
        )
        assert sum(1 for _ in circuit.all_operations()) == self.expected.operation_count[distance]


class TestXOnAllQubitsCircuit:
    """N * D times X gate on all qubits."""

    group = "circuit_operations"

    @pytest.mark.parametrize(
        ["qubit_count", "depth"], itertools.product([1, 10, 100, 1000], [1, 10, 100, 1000])
    )
    @pytest.mark.benchmark(group=group)
    def test_circuit_construction(self, benchmark, qubit_count: int, depth: int) -> None:
        q = cirq.LineQubit.range(qubit_count)
        f = lambda: cirq.Circuit(cirq.Moment(cirq.X.on_each(*q)) for _ in range(depth))
        circuit = benchmark(f)
        assert len(circuit) == depth
