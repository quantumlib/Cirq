# Copyright 2019 The Cirq Developers
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

from typing import Optional, Dict, List

from cirq import value
from cirq.google.api.v2 import run_context_pb2
from cirq.study import sweeps


def sweep_to_proto(
    sweep: sweeps.Sweep,
    *,
    out: Optional[run_context_pb2.Sweep] = None,
) -> run_context_pb2.Sweep:
    """Converts a Sweep to v2 protobuf message.

    Args:
        sweep: The sweep to convert.
        out: Optional message to be populated. If not given, a new message will
            be created.

    Returns:
        Populated sweep protobuf message.
    """
    if out is None:
        out = run_context_pb2.Sweep()
    if sweep is sweeps.UnitSweep:
        pass
    elif isinstance(sweep, sweeps.Product):
        out.sweep_function.function_type = run_context_pb2.SweepFunction.PRODUCT
        for factor in sweep.factors:
            sweep_to_proto(factor, out=out.sweep_function.sweeps.add())
    elif isinstance(sweep, sweeps.Zip):
        out.sweep_function.function_type = run_context_pb2.SweepFunction.ZIP
        for s in sweep.sweeps:
            sweep_to_proto(s, out=out.sweep_function.sweeps.add())
    elif isinstance(sweep, sweeps.Linspace):
        out.single_sweep.parameter_key = sweep.key
        out.single_sweep.linspace.first_point = sweep.start
        out.single_sweep.linspace.last_point = sweep.stop
        out.single_sweep.linspace.num_points = sweep.length
    elif isinstance(sweep, sweeps.Points):
        out.single_sweep.parameter_key = sweep.key
        out.single_sweep.points.points.extend(sweep.points)
    elif isinstance(sweep, sweeps.ListSweep):
        sweep_dict: Dict[str, List[value.TParamVal]] = {}
        for param_resolver in sweep:
            for key in param_resolver:
                if key not in sweep_dict:
                    sweep_dict[key] = []
                sweep_dict[key].append(param_resolver.value_of(key))
        out.sweep_function.function_type = run_context_pb2.SweepFunction.ZIP
        for key in sweep_dict:
            sweep_to_proto(sweeps.Points(key, sweep_dict[key]), out=out.sweep_function.sweeps.add())
    else:
        raise ValueError('cannot convert to v2 Sweep proto: {}'.format(sweep))
    return out


def sweep_from_proto(msg: run_context_pb2.Sweep) -> sweeps.Sweep:
    """Creates a Sweep from a v2 protobuf message."""
    which = msg.WhichOneof('sweep')
    if which is None:
        return sweeps.UnitSweep
    if which == 'sweep_function':
        factors = [sweep_from_proto(m) for m in msg.sweep_function.sweeps]
        func_type = msg.sweep_function.function_type
        if func_type == run_context_pb2.SweepFunction.PRODUCT:
            return sweeps.Product(*factors)
        if func_type == run_context_pb2.SweepFunction.ZIP:
            return sweeps.Zip(*factors)

        raise ValueError('invalid sweep function type: {}'.format(func_type))
    if which == 'single_sweep':
        key = msg.single_sweep.parameter_key
        if msg.single_sweep.WhichOneof('sweep') == 'linspace':
            return sweeps.Linspace(
                key=key,
                start=msg.single_sweep.linspace.first_point,
                stop=msg.single_sweep.linspace.last_point,
                length=msg.single_sweep.linspace.num_points,
            )
        if msg.single_sweep.WhichOneof('sweep') == 'points':
            return sweeps.Points(key=key, points=msg.single_sweep.points.points)

        raise ValueError('single sweep type not set: {}'.format(msg))

    # coverage: ignore
    raise ValueError('sweep type not set: {}'.format(msg))
