# Copyright 2020 The Cirq Developers
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
"""Tests exclusively for FrozenCircuits.

Behavior shared with Circuit is tested with parameters in circuit_test.py.
"""

import pytest
import sympy

import cirq


def test_from_moments():
    a, b, c, d = cirq.LineQubit.range(4)
    assert cirq.FrozenCircuit.from_moments(
        [cirq.X(a), cirq.Y(b)],
        [cirq.X(c)],
        [],
        cirq.Z(d),
        [cirq.measure(a, b, key='ab'), cirq.measure(c, d, key='cd')],
    ) == cirq.FrozenCircuit(
        cirq.Moment(cirq.X(a), cirq.Y(b)),
        cirq.Moment(cirq.X(c)),
        cirq.Moment(),
        cirq.Moment(cirq.Z(d)),
        cirq.Moment(cirq.measure(a, b, key='ab'), cirq.measure(c, d, key='cd')),
    )


def test_freeze_and_unfreeze():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(a), cirq.H(b))

    f = c.freeze()
    # Circuits equal their frozen versions, similar to set(x) == frozenset(x).
    assert f == c
    assert cirq.approx_eq(f, c)

    # Freezing a FrozenCircuit will return the original.
    ff = f.freeze()
    assert ff is f

    unf = f.unfreeze()
    assert unf.moments == c.moments
    assert unf is not c

    # Unfreezing always returns a copy.
    cc = c.unfreeze()
    assert cc is not c

    fcc = cc.freeze()
    assert fcc.moments == f.moments
    assert fcc is not f


def test_immutable():
    q = cirq.LineQubit(0)
    c = cirq.FrozenCircuit(cirq.X(q), cirq.H(q))

    # Match one of two strings. The second one is message returned since python 3.11.
    with pytest.raises(
        AttributeError,
        match="(can't set attribute)|(property 'moments' of 'FrozenCircuit' object has no setter)",
    ):
        c.moments = (cirq.Moment(cirq.H(q)), cirq.Moment(cirq.X(q)))


def test_tagged_circuits():
    q = cirq.LineQubit(0)
    ops = [cirq.X(q), cirq.H(q)]
    tags = [sympy.Symbol("a"), "b"]
    circuit = cirq.Circuit(ops)
    frozen_circuit = cirq.FrozenCircuit(ops)
    tagged_circuit = cirq.FrozenCircuit(ops, tags=tags)
    # Test equality
    assert tagged_circuit.tags == tuple(tags)
    assert circuit == frozen_circuit != tagged_circuit
    assert cirq.approx_eq(circuit, frozen_circuit)
    assert cirq.approx_eq(frozen_circuit, tagged_circuit)
    # Test hash
    assert hash(frozen_circuit) != hash(tagged_circuit)
    # Test _repr_ and _json_ round trips.
    cirq.testing.assert_equivalent_repr(tagged_circuit)
    cirq.testing.assert_json_roundtrip_works(tagged_circuit)
    # Test utility methods and constructors
    assert frozen_circuit.with_tags() is frozen_circuit
    assert frozen_circuit.with_tags(*tags) == tagged_circuit
    assert tagged_circuit.with_tags("c") == cirq.FrozenCircuit(ops, tags=[*tags, "c"])
    assert tagged_circuit.untagged == frozen_circuit
    assert frozen_circuit.untagged is frozen_circuit
    # Test parameterized protocols
    assert cirq.is_parameterized(frozen_circuit) is False
    assert cirq.is_parameterized(tagged_circuit) is True
    assert cirq.parameter_names(tagged_circuit) == {"a"}
    # Tags are not propagated to diagrams yet.
    assert str(frozen_circuit) == str(tagged_circuit)
