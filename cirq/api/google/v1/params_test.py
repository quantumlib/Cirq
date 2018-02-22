import pytest

from cirq.api.google.v1 import params
from cirq.api.google.v1.params_pb2 import (
    ParameterSweep,
    ParameterSweepZip,
    SingleParameterSweep,
)


def test_gen_sweep_points():
    points = [0.5, 1.0, 1.5, 2.0, 2.5]
    sweep = SingleParameterSweep()
    sweep.sweep_points.points.extend(points)
    out = params._gen_single_param_sweep(sweep)
    assert list(out) == points


def test_gen_sweep_linspace():
    sweep = SingleParameterSweep()
    sweep.sweep_linspace.first_point = 0
    sweep.sweep_linspace.last_point = 10
    sweep.sweep_linspace.num_points = 11
    out = params._gen_single_param_sweep(sweep)
    assert list(out) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_gen_sweep_linspace_one_point():
    sweep = SingleParameterSweep()
    sweep.sweep_linspace.first_point = 0
    sweep.sweep_linspace.last_point = 10
    sweep.sweep_linspace.num_points = 1
    out = params._gen_single_param_sweep(sweep)
    assert list(out) == [0]


def test_gen_param_sweep_zip():
    sweep = ParameterSweepZip()
    s1 = sweep.sweeps.add()
    s1.parameter_name = 'foo'
    s1.sweep_points.points.extend([1, 2, 3])
    s2 = sweep.sweeps.add()
    s2.parameter_name = 'bar'
    s2.sweep_points.points.extend([4, 5])
    out = params._gen_param_sweep_zip(sweep)
    assert list(out) == [(('foo', f), ('bar', b))
                         for f, b in zip([1, 2, 3], [4, 5])]


def test_gen_empty_param_sweep():
    ps = ParameterSweep()
    out = params.gen_param_sweep(ps)
    assert list(out) == [()]


def test_gen_param_sweep():
    ps = ParameterSweep()
    f1 = ps.sweep.factors.add()
    s1 = f1.sweeps.add()
    s1.parameter_name = 'foo'
    s1.sweep_points.points.extend([1, 2, 3])
    f2 = ps.sweep.factors.add()
    s2 = f2.sweeps.add()
    s2.parameter_name = 'bar'
    s2.sweep_points.points.extend([4, 5])
    out = params.gen_param_sweep(ps)
    assert list(out) == [(('foo', f), ('bar', b))
                         for f in [1, 2, 3] for b in [4, 5]]


def test_empty_param_sweep_names():
    ps = ParameterSweep()
    assert params.param_sweep_names(ps) == []


def test_param_sweep_names():
    ps = ParameterSweep()
    f1 = ps.sweep.factors.add()
    f1.sweeps.add().parameter_name = 'foo'
    f1.sweeps.add().parameter_name = 'bar'
    f2 = ps.sweep.factors.add()
    f2.sweeps.add().parameter_name = 'baz'
    f2.sweeps.add().parameter_name = 'qux'
    assert params.param_sweep_names(ps) == ['foo', 'bar', 'baz', 'qux']


def test_empty_param_sweep_size():
    ps = ParameterSweep()
    assert params.param_sweep_size(ps) == 1


def test_param_sweep_size():
    ps = ParameterSweep()
    f1 = ps.sweep.factors.add()
    f1.sweeps.add().sweep_linspace.num_points = 5
    f1.sweeps.add().sweep_points.points.extend(range(7))
    f2 = ps.sweep.factors.add()
    f2.sweeps.add().sweep_linspace.num_points = 11
    f2.sweeps.add().sweep_points.points.extend(range(13))
    assert params.param_sweep_size(ps) == 5 * 11


def test_param_sweep_size_no_sweeps():
    ps = ParameterSweep()
    ps.sweep.factors.add()
    ps.sweep.factors.add()
    assert params.param_sweep_size(ps) == 0


def example_sweeps():
    empty_sweep = ParameterSweep()

    empty_product = ParameterSweep()

    empty_zip = ParameterSweep()
    empty_zip.sweep.factors.add()
    empty_zip.sweep.factors.add()

    full_sweep = ParameterSweep()
    f1 = full_sweep.sweep.factors.add()
    s11 = f1.sweeps.add()
    s11.sweep_linspace.first_point = 0
    s11.sweep_linspace.last_point = 10
    s11.sweep_linspace.num_points = 5
    s12 = f1.sweeps.add()
    s12.sweep_points.points.extend(range(7))

    f2 = full_sweep.sweep.factors.add()
    s21 = f2.sweeps.add()
    s21.sweep_linspace.first_point = 0
    s21.sweep_linspace.last_point = 10
    s21.sweep_linspace.num_points = 11
    s22 = f2.sweeps.add()
    s22.sweep_points.points.extend(range(13))

    return [empty_sweep, empty_product, empty_zip, full_sweep]


@pytest.mark.parametrize('param_sweep', example_sweeps())
def test_param_sweep_size_versus_gen(param_sweep):
    predicted_size = params.param_sweep_size(param_sweep)
    out = list(params.gen_param_sweep(param_sweep))
    assert len(out) == predicted_size
