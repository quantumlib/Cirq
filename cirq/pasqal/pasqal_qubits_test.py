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
import pytest
import numpy as np

import cirq
from cirq.pasqal import ThreeDGridQubit, ThreeDQubit, TwoDQubit


def test_pasqal_qubit_init():
    q = ThreeDGridQubit(3, 4, 5)
    assert q.row == 3
    assert q.col == 4
    assert q.lay == 5


def test_comparison_key():
    assert ThreeDGridQubit(3, 4, 5)._comparison_key() == (3, 4, 5)


def test_grid_qubit_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: ThreeDGridQubit(0, 0, 0))
    eq.make_equality_group(lambda: ThreeDGridQubit(1, 0, 0))
    eq.make_equality_group(lambda: ThreeDGridQubit(0, 1, 0))
    eq.make_equality_group(lambda: ThreeDGridQubit(50, 25, 25))


def test_square():
    assert ThreeDGridQubit.square(2, top=1, left=1) == [
        ThreeDGridQubit(1, 1, 0),
        ThreeDGridQubit(1, 2, 0),
        ThreeDGridQubit(2, 1, 0),
        ThreeDGridQubit(2, 2, 0)
    ]
    assert ThreeDGridQubit.square(2) == [
        ThreeDGridQubit(0, 0, 0),
        ThreeDGridQubit(0, 1, 0),
        ThreeDGridQubit(1, 0, 0),
        ThreeDGridQubit(1, 1, 0)
    ]


def test_rec():
    assert ThreeDGridQubit.rect(1, 2, top=5, left=6) == [
        ThreeDGridQubit(5, 6, 0),
        ThreeDGridQubit(5, 7, 0)
    ]
    assert ThreeDGridQubit.rect(2, 2) == [
        ThreeDGridQubit(0, 0, 0),
        ThreeDGridQubit(0, 1, 0),
        ThreeDGridQubit(1, 0, 0),
        ThreeDGridQubit(1, 1, 0)
    ]


def test_cube():
    assert ThreeDGridQubit.cube(2, top=1, left=1, upper=1) == [
        ThreeDGridQubit(1, 1, 1),
        ThreeDGridQubit(1, 1, 2),
        ThreeDGridQubit(1, 2, 1),
        ThreeDGridQubit(1, 2, 2),
        ThreeDGridQubit(2, 1, 1),
        ThreeDGridQubit(2, 1, 2),
        ThreeDGridQubit(2, 2, 1),
        ThreeDGridQubit(2, 2, 2)
    ]
    assert ThreeDGridQubit.cube(2) == [
        ThreeDGridQubit(0, 0, 0),
        ThreeDGridQubit(0, 0, 1),
        ThreeDGridQubit(0, 1, 0),
        ThreeDGridQubit(0, 1, 1),
        ThreeDGridQubit(1, 0, 0),
        ThreeDGridQubit(1, 0, 1),
        ThreeDGridQubit(1, 1, 0),
        ThreeDGridQubit(1, 1, 1),
    ]


def test_parrallelep():
    assert ThreeDGridQubit.parallelep(1, 2, 2, top=5, left=6, upper=7) == [
        ThreeDGridQubit(5, 6, 7),
        ThreeDGridQubit(5, 6, 8),
        ThreeDGridQubit(5, 7, 7),
        ThreeDGridQubit(5, 7, 8),
    ]

    assert ThreeDGridQubit.parallelep(2, 2, 2) == [
        ThreeDGridQubit(0, 0, 0),
        ThreeDGridQubit(0, 0, 1),
        ThreeDGridQubit(0, 1, 0),
        ThreeDGridQubit(0, 1, 1),
        ThreeDGridQubit(1, 0, 0),
        ThreeDGridQubit(1, 0, 1),
        ThreeDGridQubit(1, 1, 0),
        ThreeDGridQubit(1, 1, 1)
    ]


def test_pasqal_qubit_ordering():

    for i in range(8):
        v = [int(x) for x in bin(i)[2:].zfill(3)]

        assert ThreeDGridQubit(0, 0, 0) <= ThreeDGridQubit(v[0], v[1], v[2])
        assert ThreeDGridQubit(1, 1, 1) >= ThreeDGridQubit(v[0], v[1], v[2])

        if i >= 1:
            assert ThreeDGridQubit(0, 0, 0) < ThreeDGridQubit(v[0], v[1], v[2])
        if i < 7:
            assert ThreeDGridQubit(1, 1, 1) > ThreeDGridQubit(v[0], v[1], v[2])


def test_distance():
    with pytest.raises(TypeError):
        _ = ThreeDGridQubit(0, 0, 0).distance(cirq.GridQubit(0, 0))

    for x in np.arange(-2, 3):
        for y in np.arange(-2, 3):
            for z in np.arange(-2, 3):
                assert ThreeDGridQubit(0, 0, 0).distance(
                    ThreeDGridQubit(x, y, z)) == np.sqrt(x**2 + y**2 + z**2)


def test_pasqal_qubit_is_adjacent():

    for ind in range(3):
        v = [0, 0, 0]
        for x in [-1, 1]:
            v[ind] = 1
            assert ThreeDGridQubit(0, 0, 0).is_adjacent(
                ThreeDGridQubit(v[0], v[1], v[2]))
            for u in [[x, y] for y in [-1, 1]]:
                u.insert(ind, 0)
                assert not ThreeDGridQubit(0, 0, 0).is_adjacent(
                    ThreeDGridQubit(u[0], u[1], u[2]))

    assert not ThreeDGridQubit(0, 0, 0).is_adjacent(ThreeDGridQubit(2, 0, 0))

    assert (ThreeDGridQubit(500, 999,
                            1500).is_adjacent(ThreeDGridQubit(501, 999, 1500)))
    assert not (ThreeDGridQubit(500, 999, 1500).is_adjacent(
        ThreeDGridQubit(5034, 999, 1500)))


def test_pasqal_qubit_neighbors():
    expected = {
        ThreeDGridQubit(1, 1, 2),
        ThreeDGridQubit(1, 2, 1),
        ThreeDGridQubit(2, 1, 1),
        ThreeDGridQubit(0, 1, 1),
        ThreeDGridQubit(1, 0, 1),
        ThreeDGridQubit(1, 1, 0)
    }
    assert ThreeDGridQubit(1, 1, 1).neighbors() == expected

    # Restrict to a list of qubits
    restricted_qubits = [ThreeDGridQubit(2, 1, 1), ThreeDGridQubit(2, 2, 1)]
    expected2 = {ThreeDGridQubit(2, 1, 1)}
    assert ThreeDGridQubit(1, 1, 1).neighbors(restricted_qubits) == expected2


def test_repr():
    assert repr(ThreeDGridQubit(4, -25,
                                109)) == 'pasqal.ThreeDGridQubit(4, -25, 109)'


def test_str():
    assert str(ThreeDGridQubit(4, -25, 109)) == '(4, -25, 109)'


def test_pasqal_qubit_add_subtract():
    assert ThreeDGridQubit(1, 2, 3) + (2, 5, 7) == ThreeDGridQubit(3, 7, 10)
    assert ThreeDGridQubit(1, 2, 3) + (0, 0, 0) == ThreeDGridQubit(1, 2, 3)
    assert ThreeDGridQubit(1, 2, 3) + (-1, 0, 0) == ThreeDGridQubit(0, 2, 3)
    assert ThreeDGridQubit(1, 2, 3) - (2, 5, 7) == ThreeDGridQubit(-1, -3, -4)
    assert ThreeDGridQubit(1, 2, 3) - (0, 0, 0) == ThreeDGridQubit(1, 2, 3)
    assert ThreeDGridQubit(1, 2, 3) - (-1, 0, 0) == ThreeDGridQubit(2, 2, 3)

    assert (2, 5, 7) + ThreeDGridQubit(1, 2, 3) == ThreeDGridQubit(3, 7, 10)
    assert (2, 5, 7) - ThreeDGridQubit(1, 2, 3) == ThreeDGridQubit(1, 3, 4)


def test_pasqal_qubit_neg():
    assert -ThreeDGridQubit(1, 2, 3) == ThreeDGridQubit(-1, -2, -3)


def test_pasqal_qubit_unsupported_add():
    with pytest.raises(TypeError, match='1'):
        _ = ThreeDGridQubit(1, 1, 1) + 1
    with pytest.raises(TypeError, match='(1,)'):
        _ = ThreeDGridQubit(1, 1, 1) + (1,)
    with pytest.raises(TypeError, match='(1, 2)'):
        _ = ThreeDGridQubit(1, 1, 1) + (1, 2)
    with pytest.raises(TypeError, match='(1, 2.0)'):
        _ = ThreeDGridQubit(1, 1, 1) + (1, 2.0)

    with pytest.raises(TypeError, match='1'):
        _ = ThreeDGridQubit(1, 1, 1) - 1


def test_to_json():
    q = ThreeDGridQubit(1, 1, 1)
    d = q._json_dict_()
    assert d == {
        'cirq_type': 'ThreeDGridQubit',
        'row': 1,
        'col': 1,
        'lay': 1,
    }


def test_pasqal_qubit_init_3D():
    q = ThreeDQubit(3, 4, 5)
    assert q.x == 3
    assert q.y == 4
    assert q.z == 5


def test_comparison_key_3D():
    assert ThreeDQubit(3, 4, 5)._comparison_key() == (5, 4, 3)
    coords = (np.cos(np.pi / 2), np.sin(np.pi / 2), 0)
    assert ThreeDQubit(*coords) == ThreeDQubit(0, 1, 0)


def test_pasqal_qubit_ordering_3D():
    assert ThreeDQubit(0, 0, 1) >= ThreeDQubit(1, 0, 0)
    assert ThreeDQubit(0, 0, 1) >= ThreeDQubit(0, 1, 0)
    assert ThreeDQubit(0, 1, 0) >= ThreeDQubit(1, 0, 0)
    for i in range(8):
        v = [int(x) for x in bin(i)[2:].zfill(3)]

        assert ThreeDQubit(0, 0, 0) <= ThreeDQubit(v[0], v[1], v[2])
        assert ThreeDQubit(1, 1, 1) >= ThreeDQubit(v[0], v[1], v[2])

        if i >= 1:
            assert ThreeDQubit(0, 0, 0) < ThreeDQubit(v[0], v[1], v[2])
        if i < 7:
            assert ThreeDQubit(1, 1, 1) > ThreeDQubit(v[0], v[1], v[2])


def test_distance_3D():
    with pytest.raises(TypeError):
        _ = ThreeDQubit(0, 0, 0).distance(cirq.GridQubit(0, 0))

    for x in np.arange(-2, 3):
        for y in np.arange(-2, 3):
            for z in np.arange(-2, 3):
                assert ThreeDQubit(0, 0, 0).distance(ThreeDQubit(
                    x, y, z)) == np.sqrt(x**2 + y**2 + z**2)


def test_grid_qubit_eq_3D():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: ThreeDQubit(0, 0, 0))
    eq.make_equality_group(lambda: ThreeDQubit(1, 0, 0))
    eq.make_equality_group(lambda: ThreeDQubit(0, 1, 0))
    eq.make_equality_group(lambda: ThreeDQubit(50, 25, 25))


def test_cube_3D():
    assert ThreeDQubit.cube(2, x0=1, y0=1, z0=1) == [
        ThreeDQubit(1, 1, 1),
        ThreeDQubit(2, 1, 1),
        ThreeDQubit(1, 2, 1),
        ThreeDQubit(2, 2, 1),
        ThreeDQubit(1, 1, 2),
        ThreeDQubit(2, 1, 2),
        ThreeDQubit(1, 2, 2),
        ThreeDQubit(2, 2, 2)
    ]
    assert ThreeDQubit.cube(2) == [
        ThreeDQubit(0, 0, 0),
        ThreeDQubit(1, 0, 0),
        ThreeDQubit(0, 1, 0),
        ThreeDQubit(1, 1, 0),
        ThreeDQubit(0, 0, 1),
        ThreeDQubit(1, 0, 1),
        ThreeDQubit(0, 1, 1),
        ThreeDQubit(1, 1, 1),
    ]


def test_parrallelep_3D():
    assert ThreeDQubit.parallelep(1, 2, 2, x0=5, y0=6, z0=7) == [
        ThreeDQubit(5, 6, 7),
        ThreeDQubit(5, 7, 7),
        ThreeDQubit(5, 6, 8),
        ThreeDQubit(5, 7, 8),
    ]

    assert ThreeDQubit.parallelep(2, 2, 2) == [
        ThreeDQubit(0, 0, 0),
        ThreeDQubit(1, 0, 0),
        ThreeDQubit(0, 1, 0),
        ThreeDQubit(1, 1, 0),
        ThreeDQubit(0, 0, 1),
        ThreeDQubit(1, 0, 1),
        ThreeDQubit(0, 1, 1),
        ThreeDQubit(1, 1, 1)
    ]


def test_square_2D():
    assert TwoDQubit.square(2, x0=1, y0=1) == [
        TwoDQubit(1, 1),
        TwoDQubit(2, 1),
        TwoDQubit(1, 2),
        TwoDQubit(2, 2)
    ]
    assert TwoDQubit.square(2) == [
        TwoDQubit(0, 0),
        TwoDQubit(1, 0),
        TwoDQubit(0, 1),
        TwoDQubit(1, 1)
    ]


def test_rec_2D():
    assert TwoDQubit.rect(1, 2, x0=5,
                          y0=6) == [TwoDQubit(5, 6),
                                    TwoDQubit(5, 7)]
    assert TwoDQubit.rect(2, 2) == [
        TwoDQubit(0, 0),
        TwoDQubit(1, 0),
        TwoDQubit(0, 1),
        TwoDQubit(1, 1)
    ]


def test_triangular_2D():
    assert TwoDQubit.triangular_lattice(1) == [
        TwoDQubit(0.0, 0.0),
        TwoDQubit(0.5, 0.8660254037844386),
        TwoDQubit(1.0, 0.0),
        TwoDQubit(1.5, 0.8660254037844386)
    ]

    assert TwoDQubit.triangular_lattice(1, x0=5., y0=6.1) == [
        TwoDQubit(5.0, 6.1),
        TwoDQubit(5.5, 6.966025403784438),
        TwoDQubit(6.0, 6.1),
        TwoDQubit(6.5, 6.966025403784438)
    ]


def test_repr_():
    assert repr(ThreeDQubit(4, -25, 109)) == 'pasqal.ThreeDQubit(4, -25, 109)'
    assert repr(TwoDQubit(4, -25)) == 'pasqal.TwoDQubit(4, -25)'


def test_str_():
    assert str(ThreeDQubit(4, -25, 109)) == '(4, -25, 109)'
    assert str(TwoDQubit(4, -25)) == '(4, -25)'


def test_to_json_():
    q = ThreeDQubit(1.3, 1, 1)
    d = q._json_dict_()
    assert d == {
        'cirq_type': 'ThreeDQubit',
        'x': 1.3,
        'y': 1,
        'z': 1,
    }
    q = TwoDQubit(1.3, 1)
    d = q._json_dict_()
    assert d == {
        'cirq_type': 'TwoDQubit',
        'x': 1.3,
        'y': 1,
    }
