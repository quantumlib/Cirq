from cirq.api.google.v1.params_pb2 import (
    ParameterSweep,
    ParameterSweepZip,
    SingleParameterSweep,
)
from cirq.study.sweeps import (
    Linspace, Points, Product, Sweep, Unit, Zip,
)


def sweep_from_proto(param_sweep: ParameterSweep) -> Sweep:
    if not param_sweep.HasField('sweep'):
        return Unit
    return Product(*[_sweep_from_param_sweep_zip(f)
                     for f in param_sweep.sweep.factors])


def _sweep_from_param_sweep_zip(param_sweep_zip: ParameterSweepZip) -> Sweep:
    return Zip(*[_sweep_from_single_param_sweep(sweep)
                 for sweep in param_sweep_zip.sweeps])


def _sweep_from_single_param_sweep(
        single_param_sweep: SingleParameterSweep) -> Sweep:
    key = single_param_sweep.parameter_name
    which = single_param_sweep.WhichOneof('sweep')
    if which == 'sweep_points':
        sp = single_param_sweep.sweep_points
        return Points(key, list(sp.points))
    elif which == 'sweep_linspace':
        sl = single_param_sweep.sweep_linspace
        return Linspace(key, sl.first_point, sl.last_point, sl.num_points)
    else:
        raise ValueError('unknown single param sweep type: {}'.format(which))
