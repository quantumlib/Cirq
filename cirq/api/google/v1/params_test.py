from cirq.api.google.v1 import params
from cirq.api.google.v1.params_pb2 import (
    ParameterSweep,
    ParameterSweepZip,
    ParameterSweepZipProduct,
    SingleParameterSweep,
)


def test_gen_sweep_points():
    points = [0.5, 1.0, 1.5, 2.0, 2.5]
    sweep = SingleParameterSweep()
    sweep.sweep_points.points.extend(points)
    out = params.gen_single_param_sweep(sweep)
    assert list(out) == points


def test_gen_sweep_linspace():
    sweep = SingleParameterSweep()
    sweep.sweep_linspace.first_point = 0
    sweep.sweep_linspace.last_point = 10
    sweep.sweep_linspace.num_points = 11
    out = params.gen_single_param_sweep(sweep)
    assert list(out) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_gen_sweep_linspace_one_point():
    sweep = SingleParameterSweep()
    sweep.sweep_linspace.first_point = 0
    sweep.sweep_linspace.last_point = 10
    sweep.sweep_linspace.num_points = 1
    out = params.gen_single_param_sweep(sweep)
    assert list(out) == [0]


def test_gen_param_sweep_zip():
    sweep = ParameterSweepZip()
    s1 = sweep.sweeps.add()
    s1.parameter_name = 'foo'
    s1.sweep_points.points.extend([1, 2, 3])
    s2 = sweep.sweeps.add()
    s2.parameter_name = 'bar'
    s2.sweep_points.points.extend([4, 5])
    out = params.gen_param_sweep_zip(sweep)
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
