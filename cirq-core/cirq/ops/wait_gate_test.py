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
import datetime

import pytest
import sympy

import cirq


def test_init():
    g = cirq.WaitGate(datetime.timedelta(0, 0, 5))
    assert g.duration == cirq.Duration(micros=5)

    g = cirq.WaitGate(cirq.Duration(nanos=4))
    assert g.duration == cirq.Duration(nanos=4)

    g = cirq.WaitGate(0)
    assert g.duration == cirq.Duration(0)

    with pytest.raises(ValueError, match='duration < 0'):
        _ = cirq.WaitGate(cirq.Duration(nanos=-4))

    with pytest.raises(TypeError, match='Not a `cirq.DURATION_LIKE`'):
        _ = cirq.WaitGate(2)


def test_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.WaitGate(0), cirq.WaitGate(cirq.Duration()))
    eq.make_equality_group(lambda: cirq.WaitGate(cirq.Duration(nanos=4)))


def test_protocols():
    t = sympy.Symbol('t')
    p = cirq.WaitGate(cirq.Duration(millis=5 * t))
    c = cirq.WaitGate(cirq.Duration(millis=2))
    q = cirq.LineQubit(0)

    cirq.testing.assert_implements_consistent_protocols(cirq.wait(q, nanos=0))
    cirq.testing.assert_implements_consistent_protocols(c.on(q))
    cirq.testing.assert_implements_consistent_protocols(p.on(q))

    assert cirq.has_unitary(p)
    assert cirq.has_unitary(c)
    assert cirq.is_parameterized(p)
    assert not cirq.is_parameterized(c)
    assert cirq.resolve_parameters(p, {'t': 2}) == cirq.WaitGate(cirq.Duration(millis=10))
    assert cirq.resolve_parameters(c, {'t': 2}) == c
    assert cirq.resolve_parameters_once(c, {'t': 2}) == c
    assert cirq.trace_distance_bound(p) == 0
    assert cirq.trace_distance_bound(c) == 0
    assert cirq.inverse(c) == c
    assert cirq.inverse(p) == p
    assert cirq.decompose(c.on(q)) == []
    assert cirq.decompose(p.on(q)) == []


def test_qid_shape():
    assert cirq.qid_shape(cirq.WaitGate(0, qid_shape=(2, 3))) == (2, 3)
    assert cirq.qid_shape(cirq.WaitGate(0, num_qubits=3)) == (2, 2, 2)
    with pytest.raises(ValueError, match='empty set of qubits'):
        cirq.WaitGate(0, num_qubits=0)
    with pytest.raises(ValueError, match='num_qubits'):
        cirq.WaitGate(0, qid_shape=(2, 2), num_qubits=1)


def test_json():
    q0, q1 = cirq.GridQubit.rect(1, 2)
    qtrit = cirq.GridQid(1, 2, dimension=3)
    cirq.testing.assert_json_roundtrip_works(cirq.wait(q0, nanos=10))
    cirq.testing.assert_json_roundtrip_works(cirq.wait(q0, q1, nanos=10))
    cirq.testing.assert_json_roundtrip_works(cirq.wait(qtrit, nanos=10))
    cirq.testing.assert_json_roundtrip_works(cirq.wait(qtrit, q1, nanos=10))


def test_str():
    assert str(cirq.WaitGate(cirq.Duration(nanos=5))) == 'WaitGate(5 ns)'


def test_setters_deprecated():
    duration = cirq.Duration(millis=1)
    gate = cirq.WaitGate(duration)
    assert gate.duration == duration
    with cirq.testing.assert_deprecated('mutators', deadline='v0.15'):
        new_duration = cirq.Duration(millis=2)
        gate.duration = new_duration
        assert gate.duration == new_duration
