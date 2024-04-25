# Copyright 2021 The Cirq Developers
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
import cirq_google.engine.engine_validator as engine_validator

SERIALIZABLE_GATE_DOMAIN = {cirq.X: 1, cirq.Y: 1, cirq.Z: 1, cirq.S: 1, cirq.T: 1, cirq.CZ: 2}


def _big_circuit(num_cycles: int) -> cirq.Circuit:
    qubits = cirq.GridQubit.rect(6, 6)
    moment_1q = cirq.Moment([cirq.X(q) for q in qubits])
    moment_2q = cirq.Moment(
        [
            cirq.CZ(cirq.GridQubit(row, col), cirq.GridQubit(row, col + 1))
            for row in range(6)
            for col in [0, 2, 4]
        ]
    )
    c = cirq.Circuit()
    for _ in range(num_cycles):
        c.append(moment_1q)
        c.append(moment_2q)
    return c


def test_validate_for_engine():
    circuit = _big_circuit(4)
    long_circuit = cirq.Circuit([cirq.X(cirq.GridQubit(0, 0))] * 10001)

    with pytest.raises(RuntimeError, match='Provided circuit exceeds the limit'):
        engine_validator.validate_for_engine([long_circuit], [{}], 1000)

    with pytest.raises(RuntimeError, match='the number of requested total repetitions'):
        engine_validator.validate_for_engine([circuit], [{}], 10_000_000)

    with pytest.raises(RuntimeError, match='the number of requested total repetitions'):
        engine_validator.validate_for_engine([circuit] * 6, [{}] * 6, 1_000_000)

    with pytest.raises(RuntimeError, match='the number of requested total repetitions'):
        engine_validator.validate_for_engine(
            [circuit] * 6, [{}] * 6, [4_000_000, 2_000_000, 1, 1, 1, 1]
        )


def test_validate_for_engine_no_meas():
    circuit = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0)))
    with pytest.raises(RuntimeError, match='Code must measure at least one qubit.'):
        engine_validator.validate_for_engine([circuit] * 6, [{}] * 6, 1_000)
