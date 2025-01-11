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
"""Tests for grid_qubit."""

import pickle

import numpy as np
import pytest

import cirq


def test_init():
    q = cirq.GridQubit(3, 4)
    assert q.row == 3
    assert q.col == 4

    q = cirq.GridQid(1, 2, dimension=3)
    assert q.row == 1
    assert q.col == 2
    assert q.dimension == 3


def test_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.GridQubit(0, 0), lambda: cirq.GridQid(0, 0, dimension=2))
    eq.make_equality_group(lambda: cirq.GridQubit(1, 0), lambda: cirq.GridQid(1, 0, dimension=2))
    eq.make_equality_group(lambda: cirq.GridQubit(0, 1), lambda: cirq.GridQid(0, 1, dimension=2))
    eq.make_equality_group(lambda: cirq.GridQid(0, 0, dimension=3))


def test_grid_qubit_pickled_hash():
    # Use a large number that is unlikely to be used by any other tests.
    row, col = 123456789, 2345678910
    q_bad = cirq.GridQubit(row, col)
    cirq.GridQubit._cache.pop((row, col))
    q = cirq.GridQubit(row, col)
    _test_qid_pickled_hash(q, q_bad)


def test_grid_qid_pickled_hash():
    # Use a large number that is unlikely to be used by any other tests.
    row, col = 123456789, 2345678910
    q_bad = cirq.GridQid(row, col, dimension=3)
    cirq.GridQid._cache.pop((row, col, 3))
    q = cirq.GridQid(row, col, dimension=3)
    _test_qid_pickled_hash(q, q_bad)


def _test_qid_pickled_hash(q: 'cirq.Qid', q_bad: 'cirq.Qid') -> None:
    """Test that hashes are not pickled with Qid instances."""
    assert q_bad is not q
    _ = hash(q_bad)  # compute hash to ensure it is cached.
    q_bad._hash = q_bad._hash + 1  # type: ignore[attr-defined]
    assert q_bad == q
    assert hash(q_bad) != hash(q)
    data = pickle.dumps(q_bad)
    q_ok = pickle.loads(data)
    assert q_ok == q
    assert hash(q_ok) == hash(q)


def test_str():
    assert str(cirq.GridQubit(5, 2)) == 'q(5, 2)'
    assert str(cirq.GridQid(5, 2, dimension=3)) == 'q(5, 2) (d=3)'


def test_circuit_info():
    assert cirq.circuit_diagram_info(cirq.GridQubit(5, 2)) == cirq.CircuitDiagramInfo(
        wire_symbols=('(5, 2)',)
    )
    assert cirq.circuit_diagram_info(cirq.GridQid(5, 2, dimension=3)) == cirq.CircuitDiagramInfo(
        wire_symbols=('(5, 2) (d=3)',)
    )


def test_repr():
    cirq.testing.assert_equivalent_repr(cirq.GridQubit(5, 2))
    cirq.testing.assert_equivalent_repr(cirq.GridQid(5, 2, dimension=3))


def test_cmp():
    order = cirq.testing.OrderTester()
    order.add_ascending_equivalence_group(cirq.GridQubit(0, 0), cirq.GridQid(0, 0, dimension=2))
    order.add_ascending(
        cirq.GridQid(0, 0, dimension=3),
        cirq.GridQid(0, 1, dimension=1),
        cirq.GridQubit(0, 1),
        cirq.GridQid(0, 1, dimension=3),
        cirq.GridQid(1, 0, dimension=1),
        cirq.GridQubit(1, 0),
        cirq.GridQid(1, 0, dimension=3),
        cirq.GridQid(1, 1, dimension=1),
        cirq.GridQubit(1, 1),
        cirq.GridQid(1, 1, dimension=3),
    )


def test_cmp_failure():
    with pytest.raises(TypeError, match='not supported between instances'):
        _ = 0 < cirq.GridQubit(0, 0)
    with pytest.raises(TypeError, match='not supported between instances'):
        _ = cirq.GridQubit(0, 0) < 0
    with pytest.raises(TypeError, match='not supported between instances'):
        _ = 0 < cirq.GridQid(1, 1, dimension=3)
    with pytest.raises(TypeError, match='not supported between instances'):
        _ = cirq.GridQid(1, 1, dimension=3) < 0


def test_is_adjacent():
    assert cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(0, 1))
    assert cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(0, -1))
    assert cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(1, 0))
    assert cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(-1, 0))

    assert not cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(+1, -1))
    assert not cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(+1, +1))
    assert not cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(-1, -1))
    assert not cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(-1, +1))

    assert not cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(2, 0))

    assert cirq.GridQubit(500, 999).is_adjacent(cirq.GridQubit(501, 999))
    assert not cirq.GridQubit(500, 999).is_adjacent(cirq.GridQubit(5034, 999))


def test_neighbors():
    assert cirq.GridQubit(1, 1).neighbors() == {
        cirq.GridQubit(1, 2),
        cirq.GridQubit(2, 1),
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 0),
    }

    # Restrict to a list of qubits
    restricted_qubits = [cirq.GridQubit(2, 1), cirq.GridQubit(2, 2)]
    assert cirq.GridQubit(1, 1).neighbors(restricted_qubits) == {cirq.GridQubit(2, 1)}


def test_square():
    assert cirq.GridQubit.square(2, top=1, left=1) == [
        cirq.GridQubit(1, 1),
        cirq.GridQubit(1, 2),
        cirq.GridQubit(2, 1),
        cirq.GridQubit(2, 2),
    ]
    assert cirq.GridQubit.square(2) == [
        cirq.GridQubit(0, 0),
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 0),
        cirq.GridQubit(1, 1),
    ]

    assert cirq.GridQid.square(2, top=1, left=1, dimension=3) == [
        cirq.GridQid(1, 1, dimension=3),
        cirq.GridQid(1, 2, dimension=3),
        cirq.GridQid(2, 1, dimension=3),
        cirq.GridQid(2, 2, dimension=3),
    ]
    assert cirq.GridQid.square(2, dimension=3) == [
        cirq.GridQid(0, 0, dimension=3),
        cirq.GridQid(0, 1, dimension=3),
        cirq.GridQid(1, 0, dimension=3),
        cirq.GridQid(1, 1, dimension=3),
    ]


def test_rect():
    assert cirq.GridQubit.rect(1, 2, top=5, left=6) == [cirq.GridQubit(5, 6), cirq.GridQubit(5, 7)]
    assert cirq.GridQubit.rect(2, 2) == [
        cirq.GridQubit(0, 0),
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 0),
        cirq.GridQubit(1, 1),
    ]

    assert cirq.GridQid.rect(1, 2, top=5, left=6, dimension=3) == [
        cirq.GridQid(5, 6, dimension=3),
        cirq.GridQid(5, 7, dimension=3),
    ]
    assert cirq.GridQid.rect(2, 2, dimension=3) == [
        cirq.GridQid(0, 0, dimension=3),
        cirq.GridQid(0, 1, dimension=3),
        cirq.GridQid(1, 0, dimension=3),
        cirq.GridQid(1, 1, dimension=3),
    ]


def test_diagram():
    s = """
-----AB-----
----ABCD----
---ABCDEF---
--ABCDEFGH--
-ABCDEFGHIJ-
ABCDEFGHIJKL
-CDEFGHIJKL-
--EFGHIJKL--
---GHIJKL---
----IJKL----
-----KL-----
"""
    assert len(cirq.GridQubit.from_diagram(s)) == 72
    assert len(cirq.GridQid.from_diagram(s, dimension=3)) == 72

    s2 = """
AB
BA"""
    assert cirq.GridQubit.from_diagram(s2) == [
        cirq.GridQubit(0, 0),
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 0),
        cirq.GridQubit(1, 1),
    ]
    assert cirq.GridQid.from_diagram(s2, dimension=3) == [
        cirq.GridQid(0, 0, dimension=3),
        cirq.GridQid(0, 1, dimension=3),
        cirq.GridQid(1, 0, dimension=3),
        cirq.GridQid(1, 1, dimension=3),
    ]

    with pytest.raises(ValueError, match="Input string has invalid character"):
        cirq.GridQubit.from_diagram('@')


def test_addition_subtraction():
    # GridQubits
    assert cirq.GridQubit(1, 2) + (2, 5) == cirq.GridQubit(3, 7)
    assert cirq.GridQubit(1, 2) + (0, 0) == cirq.GridQubit(1, 2)
    assert cirq.GridQubit(1, 2) + (-1, 0) == cirq.GridQubit(0, 2)
    assert cirq.GridQubit(1, 2) - (2, 5) == cirq.GridQubit(-1, -3)
    assert cirq.GridQubit(1, 2) - (0, 0) == cirq.GridQubit(1, 2)
    assert cirq.GridQubit(1, 2) - (-1, 0) == cirq.GridQubit(2, 2)

    assert (2, 5) + cirq.GridQubit(1, 2) == cirq.GridQubit(3, 7)
    assert (2, 5) - cirq.GridQubit(1, 2) == cirq.GridQubit(1, 3)

    assert cirq.GridQubit(1, 2) + cirq.GridQubit(3, 5) == cirq.GridQubit(4, 7)
    assert cirq.GridQubit(3, 5) - cirq.GridQubit(2, 1) == cirq.GridQubit(1, 4)
    assert cirq.GridQubit(1, -2) + cirq.GridQubit(3, 5) == cirq.GridQubit(4, 3)

    # GridQids
    assert cirq.GridQid(1, 2, dimension=3) + (2, 5) == cirq.GridQid(3, 7, dimension=3)
    assert cirq.GridQid(1, 2, dimension=3) + (0, 0) == cirq.GridQid(1, 2, dimension=3)
    assert cirq.GridQid(1, 2, dimension=3) + (-1, 0) == cirq.GridQid(0, 2, dimension=3)
    assert cirq.GridQid(1, 2, dimension=3) - (2, 5) == cirq.GridQid(-1, -3, dimension=3)
    assert cirq.GridQid(1, 2, dimension=3) - (0, 0) == cirq.GridQid(1, 2, dimension=3)
    assert cirq.GridQid(1, 2, dimension=3) - (-1, 0) == cirq.GridQid(2, 2, dimension=3)

    assert (2, 5) + cirq.GridQid(1, 2, dimension=3) == cirq.GridQid(3, 7, dimension=3)
    assert (2, 5) - cirq.GridQid(1, 2, dimension=3) == cirq.GridQid(1, 3, dimension=3)

    assert cirq.GridQid(1, 2, dimension=3) + cirq.GridQid(3, 5, dimension=3) == cirq.GridQid(
        4, 7, dimension=3
    )
    assert cirq.GridQid(3, 5, dimension=3) - cirq.GridQid(2, 1, dimension=3) == cirq.GridQid(
        1, 4, dimension=3
    )
    assert cirq.GridQid(1, -2, dimension=3) + cirq.GridQid(3, 5, dimension=3) == cirq.GridQid(
        4, 3, dimension=3
    )


@pytest.mark.parametrize('dtype', (np.int8, np.int16, np.int32, np.int64, int))
def test_addition_subtraction_numpy_array(dtype):
    assert cirq.GridQubit(1, 2) + np.array([1, 2], dtype=dtype) == cirq.GridQubit(2, 4)
    assert cirq.GridQubit(1, 2) + np.array([0, 0], dtype=dtype) == cirq.GridQubit(1, 2)
    assert cirq.GridQubit(1, 2) + np.array([-1, 0], dtype=dtype) == cirq.GridQubit(0, 2)
    assert cirq.GridQubit(1, 2) - np.array([1, 2], dtype=dtype) == cirq.GridQubit(0, 0)
    assert cirq.GridQubit(1, 2) - np.array([0, 0], dtype=dtype) == cirq.GridQubit(1, 2)
    assert cirq.GridQid(1, 2, dimension=3) - np.array([-1, 0], dtype=dtype) == cirq.GridQid(
        2, 2, dimension=3
    )

    assert cirq.GridQid(1, 2, dimension=3) + np.array([1, 2], dtype=dtype) == cirq.GridQid(
        2, 4, dimension=3
    )
    assert cirq.GridQid(1, 2, dimension=3) + np.array([0, 0], dtype=dtype) == cirq.GridQid(
        1, 2, dimension=3
    )
    assert cirq.GridQid(1, 2, dimension=3) + np.array([-1, 0], dtype=dtype) == cirq.GridQid(
        0, 2, dimension=3
    )
    assert cirq.GridQid(1, 2, dimension=3) - np.array([1, 2], dtype=dtype) == cirq.GridQid(
        0, 0, dimension=3
    )
    assert cirq.GridQid(1, 2, dimension=3) - np.array([0, 0], dtype=dtype) == cirq.GridQid(
        1, 2, dimension=3
    )
    assert cirq.GridQid(1, 2, dimension=3) - np.array([-1, 0], dtype=dtype) == cirq.GridQid(
        2, 2, dimension=3
    )


def test_unsupported_add():
    with pytest.raises(TypeError, match='1'):
        _ = cirq.GridQubit(1, 1) + 1
    with pytest.raises(TypeError, match='(1,)'):
        _ = cirq.GridQubit(1, 1) + (1,)
    with pytest.raises(TypeError, match='(1, 2, 3)'):
        _ = cirq.GridQubit(1, 1) + (1, 2, 3)
    with pytest.raises(TypeError, match='(1, 2.0)'):
        _ = cirq.GridQubit(1, 1) + (1, 2.0)

    with pytest.raises(TypeError, match='1'):
        _ = cirq.GridQubit(1, 1) - 1

    with pytest.raises(TypeError, match='[1., 2.]'):
        _ = cirq.GridQubit(1, 1) + np.array([1.0, 2.0])
    with pytest.raises(TypeError, match='[1, 2, 3]'):
        _ = cirq.GridQubit(1, 1) + np.array([1, 2, 3], dtype=int)


def test_addition_subtraction_type_error():
    with pytest.raises(TypeError, match="bort"):
        _ = cirq.GridQubit(5, 3) + "bort"
    with pytest.raises(TypeError, match="bort"):
        _ = cirq.GridQubit(5, 3) - "bort"

    with pytest.raises(TypeError, match="bort"):
        _ = cirq.GridQid(5, 3, dimension=3) + "bort"
    with pytest.raises(TypeError, match="bort"):
        _ = cirq.GridQid(5, 3, dimension=3) - "bort"

    with pytest.raises(TypeError, match="Can only add GridQids with identical dimension."):
        _ = cirq.GridQid(5, 3, dimension=3) + cirq.GridQid(3, 5, dimension=4)
    with pytest.raises(TypeError, match="Can only subtract GridQids with identical dimension."):
        _ = cirq.GridQid(5, 3, dimension=3) - cirq.GridQid(3, 5, dimension=4)


def test_neg():
    assert -cirq.GridQubit(1, 2) == cirq.GridQubit(-1, -2)
    assert -cirq.GridQid(1, 2, dimension=3) == cirq.GridQid(-1, -2, dimension=3)


def test_to_json():
    assert cirq.GridQubit(5, 6)._json_dict_() == {'row': 5, 'col': 6}

    assert cirq.GridQid(5, 6, dimension=3)._json_dict_() == {'row': 5, 'col': 6, 'dimension': 3}


def test_immutable():
    # Match one of two strings. The second one is message returned since python 3.11.
    with pytest.raises(
        AttributeError,
        match="(can't set attribute)|(property 'col' of 'GridQubit' object has no setter)",
    ):
        q = cirq.GridQubit(1, 2)
        q.col = 3

    with pytest.raises(
        AttributeError,
        match="(can't set attribute)|(property 'row' of 'GridQubit' object has no setter)",
    ):
        q = cirq.GridQubit(1, 2)
        q.row = 3

    with pytest.raises(
        AttributeError,
        match="(can't set attribute)|(property 'col' of 'GridQid' object has no setter)",
    ):
        q = cirq.GridQid(1, 2, dimension=3)
        q.col = 3

    with pytest.raises(
        AttributeError,
        match="(can't set attribute)|(property 'row' of 'GridQid' object has no setter)",
    ):
        q = cirq.GridQid(1, 2, dimension=3)
        q.row = 3

    with pytest.raises(
        AttributeError,
        match="(can't set attribute)|(property 'dimension' of 'GridQid' object has no setter)",
    ):
        q = cirq.GridQid(1, 2, dimension=3)
        q.dimension = 3


def test_complex():
    assert complex(cirq.GridQubit(row=1, col=2)) == 2 + 1j
    assert isinstance(complex(cirq.GridQubit(row=1, col=2)), complex)
