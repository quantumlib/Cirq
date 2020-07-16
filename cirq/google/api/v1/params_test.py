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
import cirq.google.api.v1.params as params
from cirq.google.api.v1 import params_pb2


def test_gen_sweep_points():
    points = [0.5, 1.0, 1.5, 2.0, 2.5]
    sweep = params_pb2.SingleSweep(
        parameter_key='foo', points=params_pb2.Points(points=list(points)))
    out = params._sweep_from_single_param_sweep_proto(sweep)
    assert out == cirq.Points('foo', [0.5, 1.0, 1.5, 2.0, 2.5])


def test_gen_sweep_linspace():
    sweep = params_pb2.SingleSweep(parameter_key='foo',
                                   linspace=params_pb2.Linspace(first_point=0,
                                                                last_point=10,
                                                                num_points=11))
    out = params._sweep_from_single_param_sweep_proto(sweep)
    assert out == cirq.Linspace('foo', 0, 10, 11)


def test_gen_param_sweep_zip():
    sweep = params_pb2.ZipSweep(sweeps=[
        params_pb2.SingleSweep(parameter_key='foo',
                               points=params_pb2.Points(points=[1, 2, 3])),
        params_pb2.SingleSweep(parameter_key='bar',
                               points=params_pb2.Points(points=[4, 5]))
    ])
    out = params._sweep_from_param_sweep_zip_proto(sweep)
    assert out == cirq.Points('foo', [1, 2, 3]) + cirq.Points('bar', [4, 5])


def test_gen_empty_param_sweep():
    out = params.sweep_from_proto(params_pb2.ParameterSweep())
    assert out == cirq.UnitSweep


def test_gen_param_sweep():
    ps = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep(factors=[
        params_pb2.ZipSweep(sweeps=[
            params_pb2.SingleSweep(parameter_key='foo',
                                   points=params_pb2.Points(points=[1, 2, 3]))
        ]),
        params_pb2.ZipSweep(sweeps=[
            params_pb2.SingleSweep(parameter_key='bar',
                                   points=params_pb2.Points(points=[4, 5]))
        ])
    ]))
    out = params.sweep_from_proto(ps)
    assert out == cirq.Product(cirq.Zip(cirq.Points('foo', [1, 2, 3])),
                               cirq.Zip(cirq.Points('bar', [4, 5])))


def test_empty_param_sweep_keys():
    assert params.sweep_from_proto(params_pb2.ParameterSweep()).keys == []


def test_sweep_from_proto_missing_type():
    ps = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep(factors=[
        params_pb2.ZipSweep(
            sweeps=[params_pb2.SingleSweep(parameter_key='foo')])
    ]))
    with pytest.raises(ValueError):
        params.sweep_from_proto(ps)


def test_param_sweep_keys():
    ps = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep(factors=[
        params_pb2.ZipSweep(sweeps=[
            params_pb2.SingleSweep(parameter_key='foo',
                                   points=params_pb2.Points(points=range(5))),
            params_pb2.SingleSweep(parameter_key='bar',
                                   points=params_pb2.Points(points=range(7)))
        ]),
        params_pb2.ZipSweep(sweeps=[
            params_pb2.SingleSweep(parameter_key='baz',
                                   points=params_pb2.Points(points=range(11))),
            params_pb2.SingleSweep(parameter_key='qux',
                                   points=params_pb2.Points(points=range(13)))
        ])
    ]))
    out = params.sweep_from_proto(ps)
    assert out.keys == ['foo', 'bar', 'baz', 'qux']


def test_empty_param_sweep_size():
    assert len(params.sweep_from_proto(params_pb2.ParameterSweep())) == 1


def test_param_sweep_size():
    ps = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep(factors=[
        params_pb2.ZipSweep(sweeps=[
            params_pb2.SingleSweep(parameter_key='11',
                                   linspace=params_pb2.Linspace(first_point=0,
                                                                last_point=10,
                                                                num_points=5)),
            params_pb2.SingleSweep(parameter_key='12',
                                   points=params_pb2.Points(points=range(7)))
        ]),
        params_pb2.ZipSweep(sweeps=[
            params_pb2.SingleSweep(parameter_key='21',
                                   linspace=params_pb2.Linspace(first_point=0,
                                                                last_point=10,
                                                                num_points=11)),
            params_pb2.SingleSweep(parameter_key='22',
                                   points=params_pb2.Points(points=range(13)))
        ])
    ]))
    # Sweeps sx1 and sx2 are zipped, so should use num number of points.
    # These are then producted, so this should multiply number of points.
    assert len(params.sweep_from_proto(ps)) == 5 * 11


def test_param_sweep_size_no_sweeps():
    ps = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep(
        factors=[params_pb2.ZipSweep(),
                 params_pb2.ZipSweep()]))
    assert len(params.sweep_from_proto(ps)) == 1


def example_sweeps():
    empty_sweep = params_pb2.ParameterSweep()
    empty_product = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep())
    empty_zip = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep(
        factors=[params_pb2.ZipSweep(),
                 params_pb2.ZipSweep()]))
    full_sweep = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep(
        factors=[
            params_pb2.ZipSweep(sweeps=[
                params_pb2.SingleSweep(
                    parameter_key='11',
                    linspace=params_pb2.Linspace(
                        first_point=0, last_point=10, num_points=5)),
                params_pb2.SingleSweep(parameter_key='12',
                                       points=params_pb2.Points(
                                           points=range(7)))
            ]),
            params_pb2.ZipSweep(sweeps=[
                params_pb2.SingleSweep(
                    parameter_key='21',
                    linspace=params_pb2.Linspace(
                        first_point=0, last_point=10, num_points=11)),
                params_pb2.SingleSweep(parameter_key='22',
                                       points=params_pb2.Points(
                                           points=range(13)))
            ])
        ]))
    return [empty_sweep, empty_product, empty_zip, full_sweep]


@pytest.mark.parametrize('param_sweep', example_sweeps())
def test_param_sweep_size_versus_gen(param_sweep):
    sweep = params.sweep_from_proto(param_sweep)
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
def test_sweep_to_proto(sweep, expected):
    proto = params.sweep_to_proto(sweep)
    out = params.sweep_from_proto(proto)
    assert out == expected


class MySweep(cirq.study.sweeps.SingleSweep):
    """A sweep that is not serializable."""

    def _tuple(self):
        pass

    def _values(self):
        return ()

    def __len__(self):
        return 0


@pytest.mark.parametrize('bad_sweep', [
    cirq.Zip(
        cirq.Product(cirq.Linspace('a', 0, 10, 25), cirq.Linspace(
            'b', 0, 10, 25))),
    cirq.Product(cirq.Zip(MySweep(key='a')))
])
def test_sweep_to_proto_fail(bad_sweep):
    with pytest.raises(ValueError):
        params.sweep_to_proto(bad_sweep)
