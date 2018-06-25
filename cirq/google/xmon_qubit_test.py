# Copyright 2018 The Cirq Developers
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

from cirq import ops
from cirq.api.google.v1 import operations_pb2
from cirq.google import XmonQubit
from cirq.testing import EqualsTester


def test_xmon_qubit_init():
    q = XmonQubit(3, 4)
    assert q.row == 3
    assert q.col == 4


def test_xmon_qubit_eq():
    eq = EqualsTester()
    eq.make_equality_pair(lambda: XmonQubit(0, 0))
    eq.make_equality_pair(lambda: XmonQubit(1, 0))
    eq.make_equality_pair(lambda: XmonQubit(0, 1))
    eq.make_equality_pair(lambda: XmonQubit(50, 25))


def test_xmon_qubit_ordering():
    assert XmonQubit(0, 0) < XmonQubit(0, 1)
    assert XmonQubit(0, 0) < XmonQubit(1, 0)
    assert XmonQubit(0, 0) < XmonQubit(1, 1)
    assert XmonQubit(0, 0) <= XmonQubit(0, 0)
    assert XmonQubit(0, 0) <= XmonQubit(0, 1)
    assert XmonQubit(0, 0) <= XmonQubit(1, 0)
    assert XmonQubit(0, 0) <= XmonQubit(1, 1)

    assert XmonQubit(1, 1) > XmonQubit(0, 1)
    assert XmonQubit(1, 1) > XmonQubit(1, 0)
    assert XmonQubit(1, 1) > XmonQubit(0, 0)
    assert XmonQubit(1, 1) >= XmonQubit(1, 1)
    assert XmonQubit(1, 1) >= XmonQubit(0, 1)
    assert XmonQubit(1, 1) >= XmonQubit(1, 0)
    assert XmonQubit(1, 1) >= XmonQubit(0, 0)


def test_xmon_qubit_is_adjacent():
    assert XmonQubit(0, 0).is_adjacent(XmonQubit(0, 1))
    assert XmonQubit(0, 0).is_adjacent(XmonQubit(0, -1))
    assert XmonQubit(0, 0).is_adjacent(XmonQubit(1, 0))
    assert XmonQubit(0, 0).is_adjacent(XmonQubit(-1, 0))

    assert not XmonQubit(0, 0).is_adjacent(XmonQubit(+1, -1))
    assert not XmonQubit(0, 0).is_adjacent(XmonQubit(+1, +1))
    assert not XmonQubit(0, 0).is_adjacent(XmonQubit(-1, -1))
    assert not XmonQubit(0, 0).is_adjacent(XmonQubit(-1, +1))

    assert not XmonQubit(0, 0).is_adjacent(XmonQubit(2, 0))

    assert XmonQubit(500, 999).is_adjacent(XmonQubit(501, 999))
    assert not XmonQubit(500, 999).is_adjacent(XmonQubit(5034, 999))


def test_gate_calls_validate():
    class ValiGate(ops.Gate):
        # noinspection PyMethodMayBeStatic
        def validate_args(self, qubits):
            if len(qubits) == 3:
                raise ValueError()

    g = ValiGate()
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q10 = XmonQubit(1, 0)

    _ = g.on(q00)
    _ = g.on(q01)
    _ = g.on(q00, q10)
    with pytest.raises(ValueError):
        _ = g.on(q00, q10, q01)

    _ = g(q00)
    _ = g(q00, q10)
    with pytest.raises(ValueError):
        _ = g(q10, q01, q00)


def test_operation_init():
    q = XmonQubit(4, 5)
    g = ops.Gate()
    v = ops.Operation(g, (q,))
    assert v.gate == g
    assert v.qubits == (q,)


def test_operation_eq():
    g1 = ops.Gate()
    g2 = ops.Gate()
    r1 = [XmonQubit(1, 2)]
    r2 = [XmonQubit(3, 4)]
    r12 = r1 + r2
    r21 = r2 + r1

    eq = EqualsTester()
    eq.make_equality_pair(lambda: ops.Operation(g1, r1))
    eq.make_equality_pair(lambda: ops.Operation(g2, r1))
    eq.make_equality_pair(lambda: ops.Operation(g1, r2))
    eq.make_equality_pair(lambda: ops.Operation(g1, r12))
    eq.make_equality_pair(lambda: ops.Operation(g1, r21))


def test_to_proto():
    q = XmonQubit(5, 6)

    # Create a new message.
    proto = q.to_proto()
    assert proto.row == 5
    assert proto.col == 6

    # Populate an existing message.
    proto2 = operations_pb2.Qubit()
    q.to_proto(proto2)
    assert proto2.row == 5
    assert proto2.col == 6


def test_from_proto():
    q = XmonQubit(5, 6)
    q2 = XmonQubit.from_proto(q.to_proto())
    assert q2 == q
