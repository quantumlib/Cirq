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

from __future__ import annotations

import numpy as np
import pytest

import cirq
from cirq.devices.grid_qubit_test import _test_qid_pickled_hash


def test_init() -> None:
    q = cirq.LineQubit(1)
    assert q.x == 1

    qid = cirq.LineQid(1, dimension=3)
    assert qid.x == 1
    assert qid.dimension == 3


def test_eq() -> None:
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.LineQubit(1), lambda: cirq.LineQid(1, dimension=2))
    eq.add_equality_group(cirq.LineQubit(2))
    eq.add_equality_group(cirq.LineQubit(0))
    eq.add_equality_group(cirq.LineQid(1, dimension=3))


def test_str() -> None:
    assert str(cirq.LineQubit(5)) == 'q(5)'
    assert str(cirq.LineQid(5, dimension=3)) == 'q(5) (d=3)'


def test_repr() -> None:
    cirq.testing.assert_equivalent_repr(cirq.LineQubit(5))
    cirq.testing.assert_equivalent_repr(cirq.LineQid(5, dimension=3))


def test_cmp() -> None:
    order = cirq.testing.OrderTester()
    order.add_ascending_equivalence_group(cirq.LineQubit(0), cirq.LineQid(0, 2))
    order.add_ascending(
        cirq.LineQid(0, dimension=3),
        cirq.LineQid(1, dimension=1),
        cirq.LineQubit(1),
        cirq.LineQid(1, dimension=3),
        cirq.LineQid(2, dimension=1),
    )


def test_cmp_failure() -> None:
    with pytest.raises(TypeError, match='not supported between instances'):
        _ = 0 < cirq.LineQubit(1)
    with pytest.raises(TypeError, match='not supported between instances'):
        _ = cirq.LineQubit(1) < 0
    with pytest.raises(TypeError, match='not supported between instances'):
        _ = 0 < cirq.LineQid(1, 3)
    with pytest.raises(TypeError, match='not supported between instances'):
        _ = cirq.LineQid(1, 3) < 0


def test_line_qubit_pickled_hash() -> None:
    # Use a large number that is unlikely to be used by any other tests.
    x = 1234567891011
    q_bad = cirq.LineQubit(x)
    cirq.LineQubit._cache.pop(x)
    q = cirq.LineQubit(x)
    _test_qid_pickled_hash(q, q_bad)


def test_line_qid_pickled_hash() -> None:
    # Use a large number that is unlikely to be used by any other tests.
    x = 1234567891011
    q_bad = cirq.LineQid(x, dimension=3)
    cirq.LineQid._cache.pop((x, 3))
    q = cirq.LineQid(x, dimension=3)
    _test_qid_pickled_hash(q, q_bad)


def test_is_adjacent() -> None:
    assert cirq.LineQubit(1).is_adjacent(cirq.LineQubit(2))
    assert cirq.LineQubit(1).is_adjacent(cirq.LineQubit(0))
    assert cirq.LineQubit(2).is_adjacent(cirq.LineQubit(3))
    assert not cirq.LineQubit(1).is_adjacent(cirq.LineQubit(3))
    assert not cirq.LineQubit(2).is_adjacent(cirq.LineQubit(0))

    assert cirq.LineQubit(2).is_adjacent(cirq.LineQid(3, 3))
    assert not cirq.LineQubit(2).is_adjacent(cirq.LineQid(0, 3))


def test_neighborhood() -> None:
    assert cirq.LineQubit(1).neighbors() == {cirq.LineQubit(0), cirq.LineQubit(2)}
    restricted_qubits = [cirq.LineQubit(2), cirq.LineQubit(3)]
    assert cirq.LineQubit(1).neighbors(restricted_qubits) == {cirq.LineQubit(2)}


def test_range() -> None:
    assert cirq.LineQubit.range(0) == []
    assert cirq.LineQubit.range(1) == [cirq.LineQubit(0)]
    assert cirq.LineQubit.range(2) == [cirq.LineQubit(0), cirq.LineQubit(1)]
    assert cirq.LineQubit.range(5) == [
        cirq.LineQubit(0),
        cirq.LineQubit(1),
        cirq.LineQubit(2),
        cirq.LineQubit(3),
        cirq.LineQubit(4),
    ]

    assert cirq.LineQubit.range(0, 0) == []
    assert cirq.LineQubit.range(0, 1) == [cirq.LineQubit(0)]
    assert cirq.LineQubit.range(1, 4) == [cirq.LineQubit(1), cirq.LineQubit(2), cirq.LineQubit(3)]

    assert cirq.LineQubit.range(3, 1, -1) == [cirq.LineQubit(3), cirq.LineQubit(2)]
    assert cirq.LineQubit.range(3, 5, -1) == []
    assert cirq.LineQubit.range(1, 5, 2) == [cirq.LineQubit(1), cirq.LineQubit(3)]


def test_qid_range() -> None:
    assert cirq.LineQid.range(0, dimension=3) == []
    assert cirq.LineQid.range(1, dimension=3) == [cirq.LineQid(0, 3)]
    assert cirq.LineQid.range(2, dimension=3) == [cirq.LineQid(0, 3), cirq.LineQid(1, 3)]
    assert cirq.LineQid.range(5, dimension=3) == [
        cirq.LineQid(0, 3),
        cirq.LineQid(1, 3),
        cirq.LineQid(2, 3),
        cirq.LineQid(3, 3),
        cirq.LineQid(4, 3),
    ]

    assert cirq.LineQid.range(0, 0, dimension=4) == []
    assert cirq.LineQid.range(0, 1, dimension=4) == [cirq.LineQid(0, 4)]
    assert cirq.LineQid.range(1, 4, dimension=4) == [
        cirq.LineQid(1, 4),
        cirq.LineQid(2, 4),
        cirq.LineQid(3, 4),
    ]

    assert cirq.LineQid.range(3, 1, -1, dimension=1) == [cirq.LineQid(3, 1), cirq.LineQid(2, 1)]
    assert cirq.LineQid.range(3, 5, -1, dimension=2) == []
    assert cirq.LineQid.range(1, 5, 2, dimension=2) == [cirq.LineQid(1, 2), cirq.LineQid(3, 2)]


def test_for_qid_shape() -> None:
    assert cirq.LineQid.for_qid_shape(()) == []
    assert cirq.LineQid.for_qid_shape((4, 2, 3, 1)) == [
        cirq.LineQid(0, 4),
        cirq.LineQid(1, 2),
        cirq.LineQid(2, 3),
        cirq.LineQid(3, 1),
    ]
    assert cirq.LineQid.for_qid_shape((4, 2, 3, 1), start=5) == [
        cirq.LineQid(5, 4),
        cirq.LineQid(6, 2),
        cirq.LineQid(7, 3),
        cirq.LineQid(8, 1),
    ]
    assert cirq.LineQid.for_qid_shape((4, 2, 3, 1), step=2) == [
        cirq.LineQid(0, 4),
        cirq.LineQid(2, 2),
        cirq.LineQid(4, 3),
        cirq.LineQid(6, 1),
    ]
    assert cirq.LineQid.for_qid_shape((4, 2, 3, 1), start=5, step=-1) == [
        cirq.LineQid(5, 4),
        cirq.LineQid(4, 2),
        cirq.LineQid(3, 3),
        cirq.LineQid(2, 1),
    ]


def test_addition_subtraction() -> None:
    assert cirq.LineQubit(1) + 2 == cirq.LineQubit(3)
    assert cirq.LineQubit(3) - 1 == cirq.LineQubit(2)
    assert 1 + cirq.LineQubit(4) == cirq.LineQubit(5)
    assert 5 - cirq.LineQubit(3) == cirq.LineQubit(2)

    assert cirq.LineQid(1, 3) + 2 == cirq.LineQid(3, 3)
    assert cirq.LineQid(3, 3) - 1 == cirq.LineQid(2, 3)
    assert 1 + cirq.LineQid(4, 3) == cirq.LineQid(5, 3)
    assert 5 - cirq.LineQid(3, 3) == cirq.LineQid(2, 3)

    assert cirq.LineQid(1, dimension=3) + cirq.LineQid(3, dimension=3) == cirq.LineQid(
        4, dimension=3
    )
    assert cirq.LineQid(3, dimension=3) - cirq.LineQid(2, dimension=3) == cirq.LineQid(
        1, dimension=3
    )


def test_addition_subtraction_type_error() -> None:
    with pytest.raises(TypeError, match='dave'):
        _ = cirq.LineQubit(1) + 'dave'  # type: ignore[operator]
    with pytest.raises(TypeError, match='dave'):
        _ = cirq.LineQubit(1) - 'dave'  # type: ignore[operator]

    with pytest.raises(TypeError, match='dave'):
        _ = cirq.LineQid(1, 3) + 'dave'  # type: ignore[operator]
    with pytest.raises(TypeError, match='dave'):
        _ = cirq.LineQid(1, 3) - 'dave'  # type: ignore[operator]

    with pytest.raises(TypeError, match="Can only add LineQids with identical dimension."):
        _ = cirq.LineQid(5, dimension=3) + cirq.LineQid(3, dimension=4)

    with pytest.raises(TypeError, match="Can only subtract LineQids with identical dimension."):
        _ = cirq.LineQid(5, dimension=3) - cirq.LineQid(3, dimension=4)


def test_neg() -> None:
    assert -cirq.LineQubit(1) == cirq.LineQubit(-1)
    assert -cirq.LineQid(1, dimension=3) == cirq.LineQid(-1, dimension=3)


def test_json_dict() -> None:
    assert cirq.LineQubit(5)._json_dict_() == {'x': 5}
    assert cirq.LineQid(5, 3)._json_dict_() == {'x': 5, 'dimension': 3}


def test_for_gate() -> None:
    class NoQidGate:
        def _qid_shape_(self):
            return ()

    class QuditGate:
        def _qid_shape_(self):
            return (4, 2, 3, 1)

    assert cirq.LineQid.for_gate(NoQidGate()) == []
    assert cirq.LineQid.for_gate(QuditGate()) == [
        cirq.LineQid(0, 4),
        cirq.LineQid(1, 2),
        cirq.LineQid(2, 3),
        cirq.LineQid(3, 1),
    ]
    assert cirq.LineQid.for_gate(QuditGate(), start=5) == [
        cirq.LineQid(5, 4),
        cirq.LineQid(6, 2),
        cirq.LineQid(7, 3),
        cirq.LineQid(8, 1),
    ]
    assert cirq.LineQid.for_gate(QuditGate(), step=2) == [
        cirq.LineQid(0, 4),
        cirq.LineQid(2, 2),
        cirq.LineQid(4, 3),
        cirq.LineQid(6, 1),
    ]
    assert cirq.LineQid.for_gate(QuditGate(), start=5, step=-1) == [
        cirq.LineQid(5, 4),
        cirq.LineQid(4, 2),
        cirq.LineQid(3, 3),
        cirq.LineQid(2, 1),
    ]


def test_immutable() -> None:
    with pytest.raises(AttributeError, match="property 'x' of 'LineQubit' object has no setter"):
        q = cirq.LineQubit(5)
        q.x = 6  # type: ignore[misc]

    with pytest.raises(AttributeError, match="property 'x' of 'LineQid' object has no setter"):
        qid = cirq.LineQid(5, dimension=4)
        qid.x = 6  # type: ignore[misc]


def test_numeric() -> None:
    assert int(cirq.LineQubit(x=5)) == 5
    assert float(cirq.LineQubit(x=5)) == 5
    assert complex(cirq.LineQubit(x=5)) == 5 + 0j
    assert isinstance(int(cirq.LineQubit(x=5)), int)
    assert isinstance(float(cirq.LineQubit(x=5)), float)
    assert isinstance(complex(cirq.LineQubit(x=5)), complex)


@pytest.mark.parametrize('dtype', (np.int8, np.int64, float, np.float64))
def test_numpy_index(dtype) -> None:
    np5 = dtype(5)
    q = cirq.LineQubit(np5)
    assert hash(q) == 5
    assert q.x == 5
    assert q.dimension == 2
    assert isinstance(q.dimension, int)

    qid = cirq.LineQid(np5, dtype(3))
    hash(qid)  # doesn't throw
    assert qid.x == 5
    assert qid.dimension == 3
    assert isinstance(qid.dimension, int)


@pytest.mark.parametrize('dtype', (float, np.float64))
def test_non_integer_index(dtype) -> None:
    # Not supported type-wise, but is used in practice, so behavior needs to be preserved.
    q = cirq.LineQubit(dtype(5.5))
    assert q.x == 5.5
    assert q.x == dtype(5.5)
    assert isinstance(q.x, dtype)
