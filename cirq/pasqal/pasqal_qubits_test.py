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
from cirq.pasqal import ThreeDQubit


def test_pasqal_qubit_init():
    q = ThreeDQubit(3, 4, 5)
    assert q.x == 3
    assert q.y == 4
    assert q.z == 5


def test_comparison_key():
    assert ThreeDQubit(3, 4, 5)._comparison_key() == (5, 4, 3)


def test_grid_qubit_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: ThreeDQubit(0, 0, 0))
    eq.make_equality_group(lambda: ThreeDQubit(1, 0, 0))
    eq.make_equality_group(lambda: ThreeDQubit(0, 1, 0))
    eq.make_equality_group(lambda: ThreeDQubit(50, 25, 25))


def test_square():
    assert ThreeDQubit.square(2, x0=1, y0=1) == [
        ThreeDQubit(1, 1, 0),
        ThreeDQubit(2, 1, 0),
        ThreeDQubit(1, 2, 0),
        ThreeDQubit(2, 2, 0)
    ]
    assert ThreeDQubit.square(2) == [
        ThreeDQubit(0, 0, 0),
        ThreeDQubit(1, 0, 0),
        ThreeDQubit(0, 1, 0),
        ThreeDQubit(1, 1, 0)
    ]


def test_rec():
    assert ThreeDQubit.rect(1, 2, x0=5, y0=6) == [
        ThreeDQubit(5, 6, 0),
        ThreeDQubit(5, 7, 0)
    ]
    assert ThreeDQubit.rect(2, 2) == [
        ThreeDQubit(0, 0, 0),
        ThreeDQubit(1, 0, 0),
        ThreeDQubit(0, 1, 0),
        ThreeDQubit(1, 1, 0)
    ]


def test_cube():
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


def test_parrallelep():
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


def test_triangular():
    assert ThreeDQubit.triangular_lattice(1) == [
        ThreeDQubit(0.0, 0.0, 0),
        ThreeDQubit(0.5, 0.8660254037844386, 0),
        ThreeDQubit(1.0, 0.0, 0),
        ThreeDQubit(1.5, 0.8660254037844386, 0)
    ]

    assert ThreeDQubit.triangular_lattice(1, x0=5., y0=6.1) == [
        ThreeDQubit(5.0, 6.1, 0),
        ThreeDQubit(5.5, 6.966025403784438, 0),
        ThreeDQubit(6.0, 6.1, 0),
        ThreeDQubit(6.5, 6.966025403784438, 0)
    ]


def test_pasqal_qubit_ordering():
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


def test_distance():
    with pytest.raises(TypeError):
        _ = ThreeDQubit(0, 0, 0).distance(cirq.GridQubit(0, 0))

    for x in np.arange(-2, 3):
        for y in np.arange(-2, 3):
            for z in np.arange(-2, 3):
                assert ThreeDQubit(0, 0, 0).distance(
                    ThreeDQubit(x, y, z)) == np.sqrt(x**2 + y**2 + z**2)


def test_repr():
    assert repr(ThreeDQubit(4, -25,
                                109)) == 'pasqal.ThreeDQubit(4, -25, 109)'


def test_str():
    assert str(ThreeDQubit(4, -25, 109)) == '(4, -25, 109)'


def test_pasqal_qubit_add_subtract():
    assert ThreeDQubit(1, 2, 3) + (2, 5, 7) == ThreeDQubit(3, 7, 10)
    assert ThreeDQubit(1, 2, 3) + (0, 0, 0) == ThreeDQubit(1, 2, 3)
    assert ThreeDQubit(1, 2, 3) + (-1, 0, 0) == ThreeDQubit(0, 2, 3)
    assert ThreeDQubit(1, 2, 3) - (2, 5, 7) == ThreeDQubit(-1, -3, -4)
    assert ThreeDQubit(1, 2, 3) - (0, 0, 0) == ThreeDQubit(1, 2, 3)
    assert ThreeDQubit(1, 2, 3) - (-1, 0, 0) == ThreeDQubit(2, 2, 3)

    assert (2, 5, 7) + ThreeDQubit(1, 2, 3) == ThreeDQubit(3, 7, 10)
    assert (2, 5, 7) - ThreeDQubit(1, 2, 3) == ThreeDQubit(1, 3, 4)


def test_pasqal_qubit_neg():
    assert -ThreeDQubit(1, 2, 3) == ThreeDQubit(-1, -2, -3)


def test_pasqal_qubit_unsupported_add():
    with pytest.raises(TypeError, match='1'):
        _ = ThreeDQubit(1, 1, 1) + 1
    with pytest.raises(TypeError, match='(1,)'):
        _ = ThreeDQubit(1, 1, 1) + (1,)
    with pytest.raises(TypeError, match='(1, 2)'):
        _ = ThreeDQubit(1, 1, 1) + (1, 2)
    with pytest.raises(TypeError, match='(1, 2.0)'):
        _ = ThreeDQubit(1, 1, 1) + (1, 2.0)

    with pytest.raises(TypeError, match='1'):
        _ = ThreeDQubit(1, 1, 1) - 1


def test_to_json():
    q = ThreeDQubit(1.3, 1, 1)
    d = q._json_dict_()
    assert d == {
        'cirq_type': 'ThreeDQubit',
        'x': 1.3,
        'y': 1,
        'z': 1,
    }
