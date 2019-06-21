from typing import Optional

from cirq.api.google import v2
from cirq.study import sweeps

SweepFunction = v2.run_context_pb2.SweepFunction


def sweep_to_proto(
        sweep: sweeps.Sweep,
        msg: Optional[v2.run_context_pb2.Sweep]=None,
) -> v2.run_context_pb2.Sweep:
    if msg is None:
        msg = v2.run_context_pb2.Sweep()
    if sweep is sweeps.UnitSweep:
        pass
    elif isinstance(sweep, sweeps.Product):
        msg.sweep_function.function_type = SweepFunction.PRODUCT
        for factor in sweep.factors:
            sweep_to_proto(factor, msg=msg.sweep_function.sweeps.add())
    elif isinstance(sweep, sweeps.Zip):
        msg.sweep_function.function_type = SweepFunction.ZIP
        for s in sweep.sweeps:
            sweep_to_proto(s, msg=msg.sweep_function.sweeps.add())
    elif isinstance(sweep, sweeps.Linspace):
        msg.single_sweep.parameter_key = sweep.key
        msg.single_sweep.linspace.first_point = sweep.start
        msg.single_sweep.linspace.last_point = sweep.stop
        msg.single_sweep.linspace.num_points = sweep.length
    elif isinstance(sweep, sweeps.Points):
        msg.single_sweep.parameter_key = sweep.key
        for point in sweep.points:
            msg.single_sweep.points.points.append(point)
    else:
        raise ValueError('cannot convert to v2.Sweep proto: {}'.format(sweep))
    return msg


def sweep_from_proto(msg: v2.run_context_pb2.Sweep) -> sweeps.Sweep:
    which = msg.WhichOneof('sweep')
    if which is None:
        return sweeps.UnitSweep
    elif which == 'sweep_function':
        factors = [sweep_from_proto(m) for m in msg.sweep_function.sweeps]
        func_type = msg.sweep_function.function_type
        if func_type == SweepFunction.PRODUCT:
            return sweeps.Product(*factors)
        elif func_type == SweepFunction.ZIP:
            return sweeps.Zip(*factors)
        else:
            raise ValueError('invalid sweep function type: {}'.format(func_type))
    elif which == 'single_sweep':
        key = msg.single_sweep.parameter_key
        if msg.single_sweep.WhichOneof('sweep') == 'linspace':
            return sweeps.Linspace(
                key=key,
                start=msg.single_sweep.linspace.first_point,
                stop=msg.single_sweep.linspace.last_point,
                length=msg.single_sweep.linspace.num_points)
        elif msg.single_sweep.WhichOneof('sweep') == 'points':
            return sweeps.Points(key=key, points=msg.single_sweep.points.points)
        else:
            raise ValueError('single sweep type not set: {}'.format(msg))
    else:
        raise ValueError('sweep type not set: {}'.format(msg))
