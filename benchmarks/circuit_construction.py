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
from typing import Set
import numpy as np

import cirq


def get_qubits(distance: int):
    data_qubits: Set[cirq.Qid] = set()
    z_measure_qubits: Set[cirq.Qid] = set()
    x_measure_qubits: Set[cirq.Qid] = set()
    for x in range(distance + 1):
        for y in range(distance + 1):
            if x < distance and y < distance:
                data_qubits.add(cirq.GridQubit(2 * x + 1, 2 * y + 1))
            on_boundary_x = (x == 0) or (x == distance)
            on_boundary_y = (y == 0) or (y == distance)
            parity = x % 2 != y % 2
            if (on_boundary_x and parity) or (on_boundary_y and not parity):
                continue
            if parity:
                z_measure_qubits.add(cirq.GridQubit(2 * x, 2 * y))
            else:
                x_measure_qubits.add(cirq.GridQubit(2 * x, 2 * y))
    return data_qubits, z_measure_qubits, x_measure_qubits


def rotated_surface_code_memory_z_cycle(
    data_qubits, z_measure_qubits, x_measure_qubits, z_order, x_order
) -> cirq.OP_TREE:
    yield cirq.Moment([cirq.H(q) for q in x_measure_qubits])
    for k in range(4):
        op_list = []
        for measure_qubits, add, is_x in [
            [x_measure_qubits, x_order[k], True],
            [z_measure_qubits, z_order[k], False],
        ]:
            for q_meas in measure_qubits:
                q_data = q_meas + add
                if q_data in data_qubits:
                    op_list.append(cirq.CNOT(q_meas, q_data) if is_x else cirq.CNOT(q_data, q_meas))
        yield cirq.Moment(op_list)
    yield cirq.Moment([cirq.H(q) for q in x_measure_qubits])
    yield cirq.Moment(cirq.measure_each(*x_measure_qubits, *z_measure_qubits))


def surface_code_circuit(distance: int, num_rounds: int) -> cirq.Circuit:
    data_qubits, z_measure_qubits, x_measure_qubits = get_qubits(distance)
    x_order = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    z_order = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    surface_code_cycle = cirq.Circuit(
        rotated_surface_code_memory_z_cycle(
            data_qubits, x_measure_qubits, z_measure_qubits, x_order, z_order
        )
    )
    return cirq.Circuit(
        surface_code_cycle * num_rounds, cirq.Moment(cirq.measure_each(*data_qubits))
    )


class SurfaceCode_Rotated_Memory_Z:
    pretty_name = "Surface Code Rotated Memory-Z Benchmarks."
    params = [*range(3, 26, 2)]
    param_names = ["distance"]

    def time_surface_code_circuit_construction(self, distance: int) -> None:
        """Benchmark circuit construction for Rotated Bottom-Z Surface code."""
        _ = surface_code_circuit(distance, distance * distance)

    def track_surface_code_circuit_operation_count(self, distance: int) -> int:
        """Benchmark operation count for Rotated Bottom-Z Surface code."""
        circuit = surface_code_circuit(distance, distance * distance)
        return sum(1 for _ in circuit.all_operations())


SurfaceCode_Rotated_Memory_Z.track_surface_code_circuit_operation_count.unit = "Operation count"


class NDCircuit:
    pretty_name = "N * D times X gate on all qubits."
    params = [[1, 10, 100, 1000], [1, 10, 100, 1000]]
    param_names = ["Number of Qubits(N)", "Depth(D)"]

    def time_circuit_construction(self, N: int, D: int):
        q = cirq.LineQubit.range(N)
        return cirq.Circuit(cirq.Moment(cirq.X.on_each(*q)) for _ in range(D))
