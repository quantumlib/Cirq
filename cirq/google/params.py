from typing import cast

from cirq.api.google.v1 import params_pb2
from cirq.study.sweeps import (
    Linspace, Points, Product, SingleParameterSweep, Sweep, Unit, Zip,
)


def sweep_to_proto(
        sweep: Sweep,
        msg: params_pb2.ParameterSweep = None) -> params_pb2.ParameterSweep:
    """Converts sweep into an equivalent protobuf representation."""
    sweep = _to_zip_product(sweep)
    if msg is None:
        msg = params_pb2.ParameterSweep()
    for factor in sweep.factors:
        _sweep_zip_to_proto(cast(Zip, factor), msg=msg.sweep.factors.add())
    return msg



def _to_zip_product(sweep: Sweep) -> Product:
    """Converts sweep to a product of zips of single sweeps, if possible."""
    if not isinstance(sweep, Product):
        sweep = Product(sweep)
    if not all(isinstance(f, Zip) for f in sweep.factors):
        factors = [f if isinstance(f, Zip) else Zip(f) for f in sweep.factors]
        sweep = Product(*factors)
    for factor in sweep.factors:
        for term in cast(Zip, factor).sweeps:
            if not isinstance(term, SingleParameterSweep):
                raise ValueError('cannot convert to zip-product form: {}'
                                 .format(sweep))
    return sweep


def _sweep_zip_to_proto(
        sweep: Zip,
        msg: params_pb2.ParameterSweepZip) -> params_pb2.ParameterSweepZip:
    if msg is None:
        msg = params_pb2.ParameterSweepZip()
    for s in sweep.sweeps:
        _single_param_sweep_to_proto(cast(SingleParameterSweep, s),
                                     msg=msg.sweeps.add())
    return msg


def _single_param_sweep_to_proto(
        sweep: SingleParameterSweep,
        msg: params_pb2.SingleParameterSweep = None
) -> params_pb2.SingleParameterSweep:
    if msg is None:
        msg = params_pb2.SingleParameterSweep()
    if isinstance(sweep, Linspace):
        msg.parameter_name = sweep.key
        msg.sweep_linspace.first_point = sweep.start
        msg.sweep_linspace.last_point = sweep.stop
        msg.sweep_linspace.num_points = sweep.length
    elif isinstance(sweep, Points):
        msg.parameter_name = sweep.key
        msg.sweep_points.points.extend(sweep.points)
    else:
        raise ValueError('invalid single-parameter sweep: {}'.format(sweep))
    return msg


def sweep_from_proto(param_sweep: params_pb2.ParameterSweep) -> Sweep:
    if not param_sweep.HasField('sweep'):
        return Unit
    return Product(*[_sweep_from_param_sweep_zip(f)
                     for f in param_sweep.sweep.factors])


def _sweep_from_param_sweep_zip(
        param_sweep_zip: params_pb2.ParameterSweepZip) -> Sweep:
    return Zip(*[_sweep_from_single_param_sweep(sweep)
                 for sweep in param_sweep_zip.sweeps])


def _sweep_from_single_param_sweep(
        single_param_sweep: params_pb2.SingleParameterSweep) -> Sweep:
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
