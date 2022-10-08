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

from typing import List, Sequence
import functools
import numpy as np
import cirq
from cirq.experiments.qubit_characterizations import _single_qubit_cliffords, _find_inv_matrix


def dot(args: Sequence[np.ndarray]) -> np.ndarray:
    return functools.reduce(np.dot, args)


class SingleQubitRandomizedBenchmarking:
    """Benchmarks circuit construction time for single qubit randomized benchmarking circuits.

    Given a combination of `depth`, `num_qubits` and `num_circuits`, the benchmark constructs
    `num_circuits` different circuits, each spanning `num_qubits` and containing `depth` moments.
    Each moment of the circuit contains a single qubit clifford operation for each qubit.

    Thus, the generated circuits have `depth * num_qubits` single qubit clifford operations.
    """

    params = [[1, 10, 50, 100, 250, 500, 1000], [100], [20]]
    param_names = ["depth", "num_qubits", "num_circuits"]
    timeout = 600  # Change timeout to 10 minutes instead of default 60 seconds.

    def setup(self, *_):
        self.sq_xz_matrices = np.array(
            [
                dot([cirq.unitary(c) for c in reversed(group)])
                for group in _single_qubit_cliffords().c1_in_xz
            ]
        )
        self.sq_xz_cliffords: List[cirq.Gate] = [
            cirq.PhasedXZGate.from_matrix(mat) for mat in self.sq_xz_matrices
        ]

    def _get_op_grid(self, qubits: List[cirq.Qid], depth: int) -> List[List[cirq.Operation]]:
        op_grid: List[List[cirq.Operation]] = []
        for q in qubits:
            gate_ids = np.random.choice(len(self.sq_xz_cliffords), depth)
            idx = _find_inv_matrix(dot(self.sq_xz_matrices[gate_ids][::-1]), self.sq_xz_matrices)
            op_sequence = [self.sq_xz_cliffords[id].on(q) for id in gate_ids]
            op_sequence.append(self.sq_xz_cliffords[idx].on(q))
            op_grid.append(op_sequence)
        return op_grid

    def time_rb_op_grid_generation(self, depth: int, num_qubits: int, num_circuits: int):
        qubits = cirq.GridQubit.rect(1, num_qubits)
        for _ in range(num_circuits):
            self._get_op_grid(qubits, depth)

    def time_rb_circuit_construction(self, depth: int, num_qubits: int, num_circuits: int):
        qubits = cirq.GridQubit.rect(1, num_qubits)
        for _ in range(num_circuits):
            op_grid = self._get_op_grid(qubits, depth)
            circuit = cirq.Circuit(
                [cirq.Moment(ops[d] for ops in op_grid) for d in range(depth + 1)],
                cirq.Moment(cirq.measure(*qubits)),
            )
