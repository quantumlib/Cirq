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

import sympy

import cirq
from cirq.study import sweeps
from cirq_google.api.v1 import params_pb2


def sweep_to_proto(sweep: cirq.Sweep, repetitions: int = 1) -> params_pb2.ParameterSweep:
    """Converts sweep into an equivalent protobuf representation."""
    product_sweep = None
    if sweep != cirq.UnitSweep:
        sweep = _to_zip_product(sweep)
        product_sweep = params_pb2.ProductSweep(
            factors=[_sweep_zip_to_proto(cast(cirq.Zip, factor)) for factor in sweep.factors]
        )
    msg = params_pb2.ParameterSweep(repetitions=repetitions, sweep=product_sweep)
    return msg


def _to_zip_product(sweep: cirq.Sweep) -> cirq.Product:
    """Converts sweep to a product of zips of single sweeps, if possible."""
    if not isinstance(sweep, cirq.Product):
        sweep = cirq.Product(sweep)
    if not all(isinstance(f, cirq.Zip) for f in sweep.factors):
        factors = [f if isinstance(f, cirq.Zip) else cirq.Zip(f) for f in sweep.factors]
        sweep = cirq.Product(*factors)
    for factor in sweep.factors:
        for term in cast(cirq.Zip, factor).sweeps:
            if not isinstance(term, sweeps.SingleSweep):
                raise ValueError(f'cannot convert to zip-product form: {sweep}')
    return sweep


def _sweep_zip_to_proto(sweep: cirq.Zip) -> params_pb2.ZipSweep:
    sweep_list = [_single_param_sweep_to_proto(cast(sweeps.SingleSweep, s)) for s in sweep.sweeps]
    return params_pb2.ZipSweep(sweeps=sweep_list)


def _single_param_sweep_to_proto(sweep: sweeps.SingleSweep) -> params_pb2.SingleSweep:
    if isinstance(sweep, cirq.Linspace) and not isinstance(sweep.key, sympy.Expr):
        return params_pb2.SingleSweep(
            parameter_key=sweep.key,
            linspace=params_pb2.Linspace(
                first_point=sweep.start, last_point=sweep.stop, num_points=sweep.length
            ),
        )
    elif isinstance(sweep, cirq.Points) and not isinstance(sweep.key, sympy.Expr):
        return params_pb2.SingleSweep(
            parameter_key=sweep.key, points=params_pb2.Points(points=sweep.points)
        )
    else:
        raise ValueError(f'invalid single-parameter sweep: {sweep}')


def sweep_from_proto(param_sweep: params_pb2.ParameterSweep) -> cirq.Sweep:
    if param_sweep.HasField('sweep') and len(param_sweep.sweep.factors) > 0:
        return cirq.Product(
            *[_sweep_from_param_sweep_zip_proto(f) for f in param_sweep.sweep.factors]
        )
    return cirq.UnitSweep


def _sweep_from_param_sweep_zip_proto(param_sweep_zip: params_pb2.ZipSweep) -> cirq.Sweep:
    if len(param_sweep_zip.sweeps) > 0:
        return cirq.Zip(
            *[_sweep_from_single_param_sweep_proto(sweep) for sweep in param_sweep_zip.sweeps]
        )
    return cirq.UnitSweep


def _sweep_from_single_param_sweep_proto(single_param_sweep: params_pb2.SingleSweep) -> cirq.Sweep:
    key = single_param_sweep.parameter_key
    if single_param_sweep.HasField('points'):
        points = single_param_sweep.points
        return cirq.Points(key, list(points.points))
    if single_param_sweep.HasField('linspace'):
        sl = single_param_sweep.linspace
        return cirq.Linspace(key, sl.first_point, sl.last_point, sl.num_points)

    raise ValueError('Single param sweep type undefined')
