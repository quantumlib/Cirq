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
import cupy as np

import cirq
from cirq.pasqal import ThreeDQubit, TwoDQubit


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
