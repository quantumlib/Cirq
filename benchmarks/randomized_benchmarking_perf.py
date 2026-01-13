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

import functools
import itertools
from collections.abc import Iterable, Sequence

import numpy as np
import pytest

import cirq
from cirq.experiments.qubit_characterizations import _find_inv_matrix, _single_qubit_cliffords


def dot(args: Iterable[np.ndarray]) -> np.ndarray:
    return functools.reduce(np.dot, args)


class TestSingleQubitRandomizedBenchmarking:
    """Benchmarks circuit construction time for single qubit randomized benchmarking circuits.

    Given a combination of `depth`, `num_qubits` and `num_circuits`, the benchmark constructs
    `num_circuits` different circuits, each spanning `num_qubits` and containing `depth` moments.
    Each moment of the circuit contains a single qubit clifford operation for each qubit.

    Thus, the generated circuits have `depth * num_qubits` single qubit clifford operations.
    """

    group = "randomized_benchmarking"
    depth = [1, 10, 50, 100, 250, 500, 1000]
    num_qubits = [100]
    num_circuits = [20]

    # assigned in setup_class
    sq_xz_matrices: np.ndarray
    sq_xz_cliffords: Sequence[cirq.Gate]

    @classmethod
    def setup_class(cls):
        cls.sq_xz_matrices = np.array(
            [
                dot([cirq.unitary(c) for c in reversed(group)])
                for group in _single_qubit_cliffords().c1_in_xz
            ]
        )
        cls.sq_xz_cliffords = [cirq.PhasedXZGate.from_matrix(mat) for mat in cls.sq_xz_matrices]

    def _get_op_grid(self, qubits: Sequence[cirq.Qid], depth: int) -> list[list[cirq.Operation]]:
        op_grid: list[list[cirq.Operation]] = []
        for q in qubits:
            gate_ids = np.random.choice(len(self.sq_xz_cliffords), depth)
            idx = _find_inv_matrix(dot(self.sq_xz_matrices[gate_ids][::-1]), self.sq_xz_matrices)
            op_sequence = [self.sq_xz_cliffords[gate_id].on(q) for gate_id in gate_ids]
            op_sequence.append(self.sq_xz_cliffords[idx].on(q))
            op_grid.append(op_sequence)
        return op_grid

    @pytest.mark.parametrize(
        ["depth", "num_qubits", "num_circuits"], itertools.product(depth, num_qubits, num_circuits)
    )
    @pytest.mark.benchmark(group=group)
    def test_rb_op_grid_generation(
        self, benchmark, depth: int, num_qubits: int, num_circuits: int
    ) -> None:
        qubits = cirq.GridQubit.rect(1, num_qubits)

        def _f() -> None:
            for _ in range(num_circuits):
                self._get_op_grid(qubits, depth)

        benchmark(_f)

    @pytest.mark.parametrize(
        ["depth", "num_qubits", "num_circuits"], itertools.product(depth, num_qubits, num_circuits)
    )
    @pytest.mark.benchmark(group=group)
    def test_rb_circuit_construction(
        self, benchmark, depth: int, num_qubits: int, num_circuits: int
    ) -> None:
        qubits = cirq.GridQubit.rect(1, num_qubits)

        def _f() -> None:
            for _ in range(num_circuits):
                op_grid = self._get_op_grid(qubits, depth)
                cirq.Circuit(
                    [cirq.Moment(ops[d] for ops in op_grid) for d in range(depth + 1)],
                    cirq.Moment(cirq.measure(*qubits)),
                )

        benchmark(_f)
