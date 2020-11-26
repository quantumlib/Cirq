# Copyright 2020 The Cirq Developers
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

import cirq.ionq as ionq
import cirq.testing


def test_qpu_result_fields():
    result = ionq.QPUResult({0: 10, 1: 10}, num_qubits=1)
    assert result.counts() == {0: 10, 1: 10}
    assert result.repetitions() == 20
    assert result.num_qubits() == 1


def test_qpu_result_str():
    result = ionq.QPUResult({0: 10, 1: 10}, num_qubits=2)
    assert str(result) == '00: 10\n01: 10'


def test_qpu_result_eq():
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(
        ionq.QPUResult({
            0: 10,
            1: 10
        }, num_qubits=1), ionq.QPUResult({
            0: 10,
            1: 10
        }, num_qubits=1))
    equals_tester.add_equality_group(
        ionq.QPUResult({
            0: 10,
            1: 20
        }, num_qubits=1))
    equals_tester.add_equality_group(
        ionq.QPUResult({
            0: 10,
            1: 20
        }, num_qubits=2))


def test_simulator_result_fields():
    result = ionq.SimulatorResult({0: 0.4, 1: 0.6}, num_qubits=1)
    assert result.probabilities() == {0: 0.4, 1: 0.6}
    assert result.num_qubits() == 1


def test_simulator_result_str():
    result = ionq.SimulatorResult({0: 0.4, 1: 0.6}, num_qubits=2)
    assert str(result) == '00: 0.4\n01: 0.6'


def test_simulator_result_eq():
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(
        ionq.SimulatorResult({
            0: 0.5,
            1: 0.5
        }, num_qubits=1), ionq.SimulatorResult({
            0: 0.5,
            1: 0.5
        }, num_qubits=1))
    equals_tester.add_equality_group(
        ionq.SimulatorResult({
            0: 0.4,
            1: 0.6
        }, num_qubits=1))
    equals_tester.add_equality_group(
        ionq.SimulatorResult({
            0: 0.4,
            1: 0.6
        }, num_qubits=2))
