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

import pytest

import cirq

PARAMETERS = (
    # num_qubits num_moments expected_gzip_size
    (100, 100, 31874),
    (100, 1000, 316622),
    (100, 4000, 1265955),
    (500, 100, 147916),
    (500, 1000, 1479772),
    (500, 4000, 5918482),
    (1000, 100, 285386),
    (1000, 1000, 2853094),
    (1000, 4000, 11412197),
)


def _make_circuit(num_qubits: int, num_moments: int) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(num_qubits)
    one_q_x_moment = cirq.Moment(cirq.X(q) for q in qubits[::2])
    one_q_y_moment = cirq.Moment(cirq.Y(q) for q in qubits[1::2])
    two_q_cx_moment = cirq.Moment(cirq.CNOT(q1, q2) for q1, q2 in zip(qubits[::4], qubits[1::4]))
    two_q_cz_moment = cirq.Moment(cirq.CZ(q1, q2) for q1, q2 in zip(qubits[::4], qubits[1::4]))
    measurement_moment = cirq.Moment(cirq.measure_each(*qubits))
    circuit = cirq.Circuit(
        [one_q_x_moment, two_q_cx_moment, one_q_y_moment, two_q_cz_moment, measurement_moment]
        * (num_moments // 5)
    )
    return circuit


@pytest.mark.parametrize(
    ["num_qubits", "num_moments", "_expected_gzip_size"],
    PARAMETERS,
    ids=[f"{nq}-{nm}" for nq, nm, _ in PARAMETERS],
)
@pytest.mark.benchmark(group="serialization")
def test_json_serialization(
    benchmark, num_qubits: int, num_moments: int, _expected_gzip_size: int
) -> None:
    """Benchmark cirq.to_json."""
    circuit = _make_circuit(num_qubits, num_moments)
    benchmark(cirq.to_json, circuit)


@pytest.mark.parametrize(
    ["num_qubits", "num_moments", "expected_gzip_size"],
    PARAMETERS,
    ids=[f"{nq}-{nm}" for nq, nm, _ in PARAMETERS],
)
@pytest.mark.benchmark(group="serialization")
def test_json_gzip_serialization(
    benchmark, num_qubits: int, num_moments: int, expected_gzip_size: int
) -> None:
    """Benchmark cirq.to_json_gzip and check its output size."""
    circuit = _make_circuit(num_qubits, num_moments)
    gzip_data = benchmark(cirq.to_json_gzip, circuit)
    # tolerate absolute increase by 1KB or a relative increase by 1 per mille
    allowed_size = expected_gzip_size + max(expected_gzip_size // 1000, 1024)
    assert len(gzip_data) < allowed_size
