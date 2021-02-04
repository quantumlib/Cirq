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
from typing import cast

from cirq.google.api.v1 import params_pb2
from cirq.study import sweeps


def sweep_to_proto(sweep: sweeps.Sweep, repetitions: int = 1) -> params_pb2.ParameterSweep:
    """Converts sweep into an equivalent protobuf representation."""
    product_sweep = None
    if not sweep == sweeps.UnitSweep:
        sweep = _to_zip_product(sweep)
        product_sweep = params_pb2.ProductSweep(
            factors=[_sweep_zip_to_proto(cast(sweeps.Zip, factor)) for factor in sweep.factors]
        )
    msg = params_pb2.ParameterSweep(repetitions=repetitions, sweep=product_sweep)
    return msg


def _to_zip_product(sweep: sweeps.Sweep) -> sweeps.Product:
    """Converts sweep to a product of zips of single sweeps, if possible."""
    if not isinstance(sweep, sweeps.Product):
        sweep = sweeps.Product(sweep)
    if not all(isinstance(f, sweeps.Zip) for f in sweep.factors):
        factors = [f if isinstance(f, sweeps.Zip) else sweeps.Zip(f) for f in sweep.factors]
        sweep = sweeps.Product(*factors)
    for factor in sweep.factors:
        for term in cast(sweeps.Zip, factor).sweeps:
            if not isinstance(term, sweeps.SingleSweep):
                raise ValueError('cannot convert to zip-product form: {}'.format(sweep))
    return sweep


def _sweep_zip_to_proto(sweep: sweeps.Zip) -> params_pb2.ZipSweep:
    sweep_list = [_single_param_sweep_to_proto(cast(sweeps.SingleSweep, s)) for s in sweep.sweeps]
    return params_pb2.ZipSweep(sweeps=sweep_list)


def _single_param_sweep_to_proto(sweep: sweeps.SingleSweep) -> params_pb2.SingleSweep:
    if isinstance(sweep, sweeps.Linspace):
        return params_pb2.SingleSweep(
            parameter_key=sweep.key,
            linspace=params_pb2.Linspace(
                first_point=sweep.start, last_point=sweep.stop, num_points=sweep.length
            ),
        )
    elif isinstance(sweep, sweeps.Points):
        return params_pb2.SingleSweep(
            parameter_key=sweep.key, points=params_pb2.Points(points=sweep.points)
        )
    else:
        raise ValueError('invalid single-parameter sweep: {}'.format(sweep))


def sweep_from_proto(param_sweep: params_pb2.ParameterSweep) -> sweeps.Sweep:
    if param_sweep.HasField('sweep') and len(param_sweep.sweep.factors) > 0:
        return sweeps.Product(
            *[_sweep_from_param_sweep_zip_proto(f) for f in param_sweep.sweep.factors]
        )
    return sweeps.UnitSweep


def _sweep_from_param_sweep_zip_proto(param_sweep_zip: params_pb2.ZipSweep) -> sweeps.Sweep:
    if len(param_sweep_zip.sweeps) > 0:
        return sweeps.Zip(
            *[_sweep_from_single_param_sweep_proto(sweep) for sweep in param_sweep_zip.sweeps]
        )
    return sweeps.UnitSweep


def _sweep_from_single_param_sweep_proto(
    single_param_sweep: params_pb2.SingleSweep,
) -> sweeps.Sweep:
    key = single_param_sweep.parameter_key
    if single_param_sweep.HasField('points'):
        points = single_param_sweep.points
        return sweeps.Points(key, list(points.points))
    if single_param_sweep.HasField('linspace'):
        sl = single_param_sweep.linspace
        return sweeps.Linspace(key, sl.first_point, sl.last_point, sl.num_points)

    raise ValueError('Single param sweep type undefined')
