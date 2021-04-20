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

import numpy as np
import pytest

import cirq


def test_measure_qubits():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Empty application.
    with pytest.raises(ValueError, match='empty set of qubits'):
        _ = cirq.measure()

    assert cirq.measure(a) == cirq.MeasurementGate(num_qubits=1, key='a').on(a)
    assert cirq.measure(a, b) == cirq.MeasurementGate(num_qubits=2, key='a,b').on(a, b)
    assert cirq.measure(b, a) == cirq.MeasurementGate(num_qubits=2, key='b,a').on(b, a)
    assert cirq.measure(a, key='b') == cirq.MeasurementGate(num_qubits=1, key='b').on(a)
    assert cirq.measure(a, invert_mask=(True,)) == cirq.MeasurementGate(
        num_qubits=1, key='a', invert_mask=(True,)
    ).on(a)
    assert cirq.measure(*cirq.LineQid.for_qid_shape((1, 2, 3)), key='a') == cirq.MeasurementGate(
        num_qubits=3, key='a', qid_shape=(1, 2, 3)
    ).on(*cirq.LineQid.for_qid_shape((1, 2, 3)))

    with pytest.raises(ValueError, match='ndarray'):
        _ = cirq.measure(np.ndarray([1, 0]))

    with pytest.raises(ValueError, match='Qid'):
        _ = cirq.measure("bork")


def test_measure_each():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert cirq.measure_each() == []
    assert cirq.measure_each(a) == [cirq.measure(a)]
    assert cirq.measure_each(a, b) == [cirq.measure(a), cirq.measure(b)]
    assert cirq.measure_each(a.with_dimension(3), b.with_dimension(3)) == [
        cirq.measure(a.with_dimension(3)),
        cirq.measure(b.with_dimension(3)),
    ]

    assert cirq.measure_each(a, b, key_func=lambda e: e.name + '!') == [
        cirq.measure(a, key='a!'),
        cirq.measure(b, key='b!'),
    ]
