# Copyright 2019 The Cirq Developers
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

import duet

import cirq


@duet.sync
async def test_pauli_string_sample_collector() -> None:
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliSumCollector(
        circuit=cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.X(a), cirq.Z(b)),
        observable=(1 + 0j) * cirq.X(a) * cirq.X(b)
        - 16 * cirq.Y(a) * cirq.Y(b)
        + 4 * cirq.Z(a) * cirq.Z(b)
        + (1 - 0j),
        samples_per_term=100,
    )
    await p.collect_async(sampler=cirq.Simulator())
    energy = p.estimated_energy()
    assert isinstance(energy, float) and energy == 12


@duet.sync
async def test_pauli_string_sample_single() -> None:
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliSumCollector(
        circuit=cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.X(a), cirq.Z(b)),
        observable=cirq.X(a) * cirq.X(b),
        samples_per_term=100,
    )
    await p.collect_async(sampler=cirq.Simulator())
    assert p.estimated_energy() == -1


def test_pauli_string_sample_collector_identity() -> None:
    p = cirq.PauliSumCollector(
        circuit=cirq.Circuit(), observable=cirq.PauliSum() + 2j, samples_per_term=100
    )
    p.collect(sampler=cirq.Simulator())
    assert p.estimated_energy() == 2j


def test_pauli_string_sample_collector_extra_qubit_z() -> None:
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliSumCollector(
        circuit=cirq.Circuit(cirq.H(a)), observable=3 * cirq.Z(b), samples_per_term=100
    )
    p.collect(sampler=cirq.Simulator())
    assert p.estimated_energy() == 3


def test_pauli_string_sample_collector_extra_qubit_x() -> None:
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliSumCollector(
        circuit=cirq.Circuit(cirq.H(a)), observable=3 * cirq.X(b), samples_per_term=10000
    )
    p.collect(sampler=cirq.Simulator())
    assert abs(p.estimated_energy()) < 0.5
