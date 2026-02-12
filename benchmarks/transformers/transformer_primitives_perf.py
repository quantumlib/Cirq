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

PARAMETERS = (
    # num_qubits num_moments
    (100, 100),
    (100, 1000),
    (100, 4000),
    (500, 100),
    (500, 1000),
    (500, 4000),
    (1000, 100),
    (1000, 1000),
    (1000, 4000),
)


def _make_circuit(num_qubits: int, num_moments: int) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(num_qubits)
    one_q_x_moment = cirq.Moment(cirq.X(q) for q in qubits[::2])
    one_q_y_moment = cirq.Moment(cirq.Y(q) for q in qubits[1::2])
    two_q_cx_moment = cirq.Moment(cirq.CNOT(q1, q2) for q1, q2 in zip(qubits[::4], qubits[1::4]))
    two_q_cz_moment = cirq.Moment(cirq.CZ(q1, q2) for q1, q2 in zip(qubits[::4], qubits[1::4]))
    circuit = cirq.Circuit(
        [one_q_x_moment, two_q_cx_moment, one_q_y_moment, two_q_cz_moment] * (num_moments // 4)
    )
    return circuit


@pytest.mark.parametrize(["num_qubits", "num_moments"], PARAMETERS)
@pytest.mark.benchmark(group="transformer_primitives")
def test_map_moments(benchmark, num_qubits: int, num_moments: int) -> None:
    circuit = _make_circuit(num_qubits, num_moments)
    all_qubits = cirq.LineQubit.range(num_qubits)

    def map_func(m: cirq.Moment, _index: int) -> cirq.Moment:
        new_ops = [op.with_tags("old op") for op in m.operations]
        new_ops += [
            cirq.Z(q).with_tags("new op") for q in all_qubits if not m.operates_on_single_qubit(q)
        ]
        return cirq.Moment(new_ops)

    benchmark(cirq.map_moments, circuit=circuit, map_func=map_func)


@pytest.mark.parametrize(["num_qubits", "num_moments"], PARAMETERS)
@pytest.mark.benchmark(group="transformer_primitives")
def test_map_operations_apply_tag(benchmark, num_qubits: int, num_moments: int) -> None:
    circuit = _make_circuit(num_qubits, num_moments)

    def map_func(op: cirq.Operation, _index: int) -> cirq.Operation:
        return op.with_tags("old op")

    benchmark(cirq.map_operations, circuit=circuit, map_func=map_func)


@pytest.mark.parametrize(["num_qubits", "num_moments"], PARAMETERS)
@pytest.mark.benchmark(group="transformer_primitives")
def test_map_operations_to_optree(benchmark, num_qubits: int, num_moments: int) -> None:
    circuit = _make_circuit(num_qubits, num_moments)

    def map_func(op: cirq.Operation, _index: int) -> cirq.OP_TREE:
        return [op, op]

    benchmark(cirq.map_operations, circuit=circuit, map_func=map_func)


@pytest.mark.parametrize(["num_qubits", "num_moments"], PARAMETERS)
@pytest.mark.benchmark(group="transformer_primitives")
def test_map_operations_to_optree_and_unroll(benchmark, num_qubits: int, num_moments: int) -> None:
    circuit = _make_circuit(num_qubits, num_moments)

    def map_func(op: cirq.Operation, _index: int) -> cirq.OP_TREE:
        return [op, op]

    benchmark(cirq.map_operations_and_unroll, circuit=circuit, map_func=map_func)
