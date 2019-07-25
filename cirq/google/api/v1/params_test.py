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

import cirq
import cirq.google as cg


def test_gen_sweep_points():
    points = [0.5, 1.0, 1.5, 2.0, 2.5]
    sweep = {'parameter_key': 'foo', 'points': {'points': list(points)}}
    out = cg.api.v1.params._sweep_from_single_param_sweep_proto_dict(sweep)
    assert out == cirq.Points('foo', [0.5, 1.0, 1.5, 2.0, 2.5])


def test_gen_sweep_linspace():
    sweep = {
        'parameter_key': 'foo',
        'linspace': {
            'first_point': 0,
            'last_point': 10,
            'num_points': 11
        }
    }
    out = cg.api.v1.params._sweep_from_single_param_sweep_proto_dict(sweep)
    assert out == cirq.Linspace('foo', 0, 10, 11)


def test_gen_param_sweep_zip():
    s1 = {'parameter_key': 'foo', 'points': {'points': [1, 2, 3]}}
    s2 = {'parameter_key': 'bar', 'points': {'points': [4, 5]}}
    sweep = {'sweeps': [s1, s2]}
    out = cg.api.v1.params._sweep_from_param_sweep_zip_proto_dict(sweep)
    assert out == cirq.Points('foo', [1, 2, 3]) + cirq.Points('bar', [4, 5])


def test_gen_empty_param_sweep():
    out = cg.sweep_from_proto_dict({})
    assert out == cirq.UnitSweep


def test_gen_param_sweep():
    s1 = {'parameter_key': 'foo', 'points': {'points': [1, 2, 3]}}
    s2 = {'parameter_key': 'bar', 'points': {'points': [4, 5]}}
    ps = {'sweep': {'factors': [{'sweeps': [s1]}, {'sweeps': [s2]}]}}
    out = cg.sweep_from_proto_dict(ps)
    assert out == cirq.Product(cirq.Zip(cirq.Points('foo', [1, 2, 3])),
                               cirq.Zip(cirq.Points('bar', [4, 5])))


def test_empty_param_sweep_keys():
    assert cg.sweep_from_proto_dict({}).keys == []


def test_sweep_from_proto_dict_missing_type():
    s1 = {
        'parameter_key': 'foo',
    }
    ps = {
        'sweep': {
            'factors': [
                {
                    'sweeps': [s1]
                },
            ]
        }
    }
    with pytest.raises(ValueError):
        cg.sweep_from_proto_dict(ps)


def test_param_sweep_keys():
    s11 = {
        'parameter_key': 'foo',
        'points': {
            'points': range(5)
        },
    }
    s12 = {
        'parameter_key': 'bar',
        'points': {
            'points': range(7)
        },
    }

    s21 = {
        'parameter_key': 'baz',
        'points': {
            'points': range(11)
        },
    }
    s22 = {'parameter_key': 'qux', 'points': {'points': range(13)}}
    ps = {
        'sweep': {
            'factors': [{
                'sweeps': [s11, s12],
            }, {
                'sweeps': [s21, s22]
            }]
        }
    }
    out = cg.sweep_from_proto_dict(ps)
    assert out.keys == ['foo', 'bar', 'baz', 'qux']


def test_empty_param_sweep_size():
    assert len(cg.sweep_from_proto_dict({})) == 1


def test_param_sweep_size():
    s11 = {
        'parameter_key': '11',
        'linspace': {
            'first_point': 0,
            'last_point': 10,
            'num_points': 5
        }
    }
    s12 = {'parameter_key': '12', 'points': {'points': range(7)}}
    s21 = {
        'parameter_key': '21',
        'linspace': {
            'first_point': 0,
            'last_point': 10,
            'num_points': 11
        }
    }
    s22 = {'parameter_key': '22', 'points': {'points': range(13)}}
    ps = {
        'sweep': {
            'factors': [{
                'sweeps': [s11, s12],
            }, {
                'sweeps': [s21, s22]
            }]
        }
    }
    # Sweeps sx1 and sx2 are zipped, so should use num number of points.
    # These are then producted, so this should multiply number of points.
    assert len(cg.sweep_from_proto_dict(ps)) == 5 * 11


def test_param_sweep_size_no_sweeps():
    ps = {'sweep': {'factors': [{}, {}]}}
    assert len(cg.sweep_from_proto_dict(ps)) == 1


def example_sweeps():
    empty_sweep = {}
    empty_product = {'sweep': {}}
    empty_zip = {'sweep': {'factors': [{}, {}]}}
    s11 = {
        'parameter_key': '11',
        'linspace': {
            'first_point': 0,
            'last_point': 10,
            'num_points': 5
        }
    }
    s12 = {'parameter_key': '12', 'points': {'points': range(7)}}
    s21 = {
        'parameter_key': '21',
        'linspace': {
            'first_point': 0,
            'last_point': 10,
            'num_points': 11
        }
    }
    s22 = {'parameter_key': '22', 'points': {'points': range(13)}}
    full_sweep = {
        'sweep': {
            'factors': [{
                'sweeps': [s11, s12],
            }, {
                'sweeps': [s21, s22]
            }]
        }
    }
    return [empty_sweep, empty_product, empty_zip, full_sweep]


@pytest.mark.parametrize('param_sweep', example_sweeps())
def test_param_sweep_size_versus_gen(param_sweep):
    sweep = cg.sweep_from_proto_dict(param_sweep)
    print(sweep)
    predicted_size = len(sweep)
    out = list(sweep)
    assert len(out) == predicted_size


@pytest.mark.parametrize('sweep,expected', [
    (cirq.UnitSweep, cirq.UnitSweep),
    (cirq.Linspace('a', 0, 10,
                   25), cirq.Product(cirq.Zip(cirq.Linspace('a', 0, 10, 25)))),
    (cirq.Points(
        'a', [1, 2, 3]), cirq.Product(cirq.Zip(cirq.Points('a', [1, 2, 3])))),
    (
        cirq.Zip(cirq.Linspace('a', 0, 1, 5), cirq.Points('b', [1, 2, 3])),
        cirq.Product(
            cirq.Zip(cirq.Linspace('a', 0, 1, 5), cirq.Points('b', [1, 2, 3]))),
    ),
    (
        cirq.Product(cirq.Linspace('a', 0, 1, 5), cirq.Points('b', [1, 2, 3])),
        cirq.Product(cirq.Zip(cirq.Linspace('a', 0, 1, 5)),
                     cirq.Zip(cirq.Points('b', [1, 2, 3]))),
    ),
    (
        cirq.Product(
            cirq.Zip(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [4, 5, 6])),
            cirq.Linspace('c', 0, 1, 5),
        ),
        cirq.Product(
            cirq.Zip(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [4, 5, 6])),
            cirq.Zip(cirq.Linspace('c', 0, 1, 5)),
        ),
    ),
    (
        cirq.Product(
            cirq.Zip(cirq.Linspace('a', 0, 1, 5), cirq.Points('b', [1, 2, 3])),
            cirq.Zip(cirq.Linspace('c', 0, 1, 8),
                     cirq.Points('d', [1, 0.5, 0.25, 0.125])),
        ),
        cirq.Product(
            cirq.Zip(cirq.Linspace('a', 0, 1, 5), cirq.Points('b', [1, 2, 3])),
            cirq.Zip(cirq.Linspace('c', 0, 1, 8),
                     cirq.Points('d', [1, 0.5, 0.25, 0.125])),
        ),
    ),
])
def test_sweep_to_proto_dict(sweep, expected):
    proto = cg.sweep_to_proto_dict(sweep)
    out = cg.sweep_from_proto_dict(proto)
    assert out == expected


class MySweep(cirq.study.sweeps.SingleSweep):
    """A sweep that is not serializable."""

    def _tuple(self):
        pass

    def _values(self):
        pass

    def __len__(self):
        pass


@pytest.mark.parametrize('bad_sweep', [
    cirq.Zip(
        cirq.Product(cirq.Linspace('a', 0, 10, 25), cirq.Linspace(
            'b', 0, 10, 25))),
    cirq.Product(cirq.Zip(MySweep(key='a')))
])
def test_sweep_to_proto_fail(bad_sweep):
    with pytest.raises(ValueError):
        cg.sweep_to_proto_dict(bad_sweep)
