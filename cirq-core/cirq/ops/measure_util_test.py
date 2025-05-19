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

from __future__ import annotations

import numpy as np
import pytest

import cirq


def test_measure_qubits():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Empty application.
    with pytest.raises(ValueError, match='empty set of qubits'):
        _ = cirq.measure()

    with pytest.raises(ValueError, match='empty set of qubits'):
        _ = cirq.measure([])

    assert cirq.measure(a) == cirq.MeasurementGate(num_qubits=1, key='a').on(a)
    assert cirq.measure([a]) == cirq.MeasurementGate(num_qubits=1, key='a').on(a)
    assert cirq.measure(a, b) == cirq.MeasurementGate(num_qubits=2, key='a,b').on(a, b)
    assert cirq.measure([a, b]) == cirq.MeasurementGate(num_qubits=2, key='a,b').on(a, b)
    qubit_generator = (q for q in (a, b))
    assert cirq.measure(qubit_generator) == cirq.MeasurementGate(num_qubits=2, key='a,b').on(a, b)
    assert cirq.measure(b, a) == cirq.MeasurementGate(num_qubits=2, key='b,a').on(b, a)
    assert cirq.measure(a, key='b') == cirq.MeasurementGate(num_qubits=1, key='b').on(a)
    assert cirq.measure(a, invert_mask=(True,)) == cirq.MeasurementGate(
        num_qubits=1, key='a', invert_mask=(True,)
    ).on(a)
    assert cirq.measure(*cirq.LineQid.for_qid_shape((1, 2, 3)), key='a') == cirq.MeasurementGate(
        num_qubits=3, key='a', qid_shape=(1, 2, 3)
    ).on(*cirq.LineQid.for_qid_shape((1, 2, 3)))
    assert cirq.measure(cirq.LineQid.for_qid_shape((1, 2, 3)), key='a') == cirq.MeasurementGate(
        num_qubits=3, key='a', qid_shape=(1, 2, 3)
    ).on(*cirq.LineQid.for_qid_shape((1, 2, 3)))
    cmap = {(0,): np.array([[0, 1], [1, 0]])}
    assert cirq.measure(a, confusion_map=cmap) == cirq.MeasurementGate(
        num_qubits=1, key='a', confusion_map=cmap
    ).on(a)

    with pytest.raises(ValueError, match='ndarray'):
        _ = cirq.measure(np.array([1, 0]))

    with pytest.raises(ValueError, match='Qid'):
        _ = cirq.measure("bork")

    with pytest.raises(ValueError, match='Qid'):
        _ = cirq.measure([a, [b]])

    with pytest.raises(ValueError, match='Qid'):
        _ = cirq.measure([a], [b])


def test_measure_each():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert cirq.measure_each() == []
    assert cirq.measure_each([]) == []
    assert cirq.measure_each(a) == [cirq.measure(a)]
    assert cirq.measure_each([a]) == [cirq.measure(a)]
    assert cirq.measure_each(a, b) == [cirq.measure(a), cirq.measure(b)]
    assert cirq.measure_each([a, b]) == [cirq.measure(a), cirq.measure(b)]
    qubit_generator = (q for q in (a, b))
    assert cirq.measure_each(qubit_generator) == [cirq.measure(a), cirq.measure(b)]
    assert cirq.measure_each(a.with_dimension(3), b.with_dimension(3)) == [
        cirq.measure(a.with_dimension(3)),
        cirq.measure(b.with_dimension(3)),
    ]

    assert cirq.measure_each(a, b, key_func=lambda e: e.name + '!') == [
        cirq.measure(a, key='a!'),
        cirq.measure(b, key='b!'),
    ]


def test_measure_single_paulistring():
    # Correct application
    q = cirq.LineQubit.range(3)
    ps = cirq.X(q[0]) * cirq.Y(q[1]) * cirq.Z(q[2])
    assert cirq.measure_single_paulistring(ps, key='a') == cirq.PauliMeasurementGate(
        ps.values(), key='a'
    ).on(*ps.keys())

    # Test with negative coefficient
    ps_neg = -cirq.Y(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1))
    assert cirq.measure_single_paulistring(ps_neg, key='1').gate == cirq.PauliMeasurementGate(
        cirq.DensePauliString('YY', coefficient=-1), key='1'
    )

    # Empty application
    with pytest.raises(ValueError, match='should be an instance of cirq.PauliString'):
        _ = cirq.measure_single_paulistring(cirq.I(q[0]) * cirq.I(q[1]))

    # Wrong type
    with pytest.raises(ValueError, match='should be an instance of cirq.PauliString'):
        _ = cirq.measure_single_paulistring(q)

    # Coefficient != +1 or -1
    with pytest.raises(ValueError, match='must have a coefficient'):
        _ = cirq.measure_single_paulistring(-2 * ps)


def test_measure_paulistring_terms():
    # Correct application
    q = cirq.LineQubit.range(3)
    ps = cirq.X(q[0]) * cirq.Y(q[1]) * cirq.Z(q[2])
    assert cirq.measure_paulistring_terms(ps) == [
        cirq.PauliMeasurementGate([cirq.X], key=str(q[0])).on(q[0]),
        cirq.PauliMeasurementGate([cirq.Y], key=str(q[1])).on(q[1]),
        cirq.PauliMeasurementGate([cirq.Z], key=str(q[2])).on(q[2]),
    ]

    # Empty application
    with pytest.raises(ValueError, match='should be an instance of cirq.PauliString'):
        _ = cirq.measure_paulistring_terms(cirq.I(q[0]) * cirq.I(q[1]))

    # Wrong type
    with pytest.raises(ValueError, match='should be an instance of cirq.PauliString'):
        _ = cirq.measure_paulistring_terms(q)
