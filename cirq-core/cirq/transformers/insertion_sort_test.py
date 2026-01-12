# Copyright 2024 The Cirq Developers
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

from __future__ import annotations

import cirq
import cirq.transformers


def test_insertion_sort() -> None:
    c = cirq.Circuit(
        cirq.CZ(cirq.q(2), cirq.q(1)),
        cirq.CZ(cirq.q(2), cirq.q(4)),
        cirq.CZ(cirq.q(0), cirq.q(1)),
        cirq.CZ(cirq.q(2), cirq.q(1)),
        cirq.GlobalPhaseGate(1j).on(),
    )
    sorted_circuit = cirq.transformers.insertion_sort_transformer(c)
    cirq.testing.assert_same_circuits(
        sorted_circuit,
        cirq.Circuit(
            cirq.GlobalPhaseGate(1j).on(),
            cirq.CZ(cirq.q(0), cirq.q(1)),
            cirq.CZ(cirq.q(2), cirq.q(1)),
            cirq.CZ(cirq.q(2), cirq.q(1)),
            cirq.CZ(cirq.q(2), cirq.q(4)),
        ),
    )


def test_insertion_sort_same_measurement_key() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.measure(q1, key='k'), cirq.measure(q0, key='k'))
    cirq.testing.assert_same_circuits(cirq.transformers.insertion_sort_transformer(c), c)


def test_insertion_sort_measurement_and_control_key_conflict() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.measure(q1, key='k'), cirq.X(q0).with_classical_controls('k'))
    # Second operation depends on the first so they don't commute.
    cirq.testing.assert_same_circuits(cirq.transformers.insertion_sort_transformer(c), c)


def test_insertion_sort_measurement_and_control_key_conflict_other_way_around() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.measure(q0, key='k'),
        cirq.X(q1).with_classical_controls('k'),
        cirq.measure(q0, key='k'),
    )
    cirq.testing.assert_same_circuits(cirq.transformers.insertion_sort_transformer(c), c)


def test_insertion_sort_distinct_measurement_keys() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.measure(q1, key='k1'), cirq.measure(q0, key='k0'))
    # Measurement keys are distinct, so the measurements commute.
    expected = cirq.Circuit(cirq.measure(q0, key='k0'), cirq.measure(q1, key='k1'))
    assert cirq.transformers.insertion_sort_transformer(c)[0].operations == expected[0].operations


def test_insertion_sort_shared_control_key() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.X(q1).with_classical_controls('k'), cirq.X(q0).with_classical_controls('k')
    )
    expected = cirq.Circuit(
        cirq.X(q0).with_classical_controls('k'), cirq.X(q1).with_classical_controls('k')
    )
    assert cirq.transformers.insertion_sort_transformer(c)[0].operations == expected[0].operations
