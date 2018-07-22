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

from cirq.api.google.v1 import params_pb2
from cirq.study.sweeps import (
    Linspace, Points, Product, SingleSweep, Sweep, UnitSweep, Zip,
)


def sweep_to_proto(
        sweep: Sweep,
        msg: params_pb2.ParameterSweep = None) -> params_pb2.ParameterSweep:
    """Converts sweep into an equivalent protobuf representation."""
    if msg is None:
        msg = params_pb2.ParameterSweep()
    if not sweep == UnitSweep:
        sweep = _to_zip_product(sweep)
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
            if not isinstance(term, SingleSweep):
                raise ValueError('cannot convert to zip-product form: {}'
                                 .format(sweep))
    return sweep


def _sweep_zip_to_proto(
        sweep: Zip,
        msg: params_pb2.ZipSweep) -> params_pb2.ZipSweep:
    if msg is None:
        msg = params_pb2.ZipSweep()
    for s in sweep.sweeps:
        _single_param_sweep_to_proto(cast(SingleSweep, s),
                                     msg=msg.sweeps.add())
    return msg


def _single_param_sweep_to_proto(
        sweep: SingleSweep,
        msg: params_pb2.SingleSweep = None
) -> params_pb2.SingleSweep:
    if msg is None:
        msg = params_pb2.SingleSweep()
    if isinstance(sweep, Linspace):
        msg.parameter_key = sweep.key
        msg.linspace.first_point = sweep.start
        msg.linspace.last_point = sweep.stop
        msg.linspace.num_points = sweep.length
    elif isinstance(sweep, Points):
        msg.parameter_key = sweep.key
        msg.points.points.extend(sweep.points)
    else:
        raise ValueError('invalid single-parameter sweep: {}'.format(sweep))
    return msg


def sweep_from_proto(param_sweep: params_pb2.ParameterSweep) -> Sweep:
    if not param_sweep.HasField('sweep'):
        return UnitSweep
    return Product(*[_sweep_from_param_sweep_zip(f)
                     for f in param_sweep.sweep.factors])


def _sweep_from_param_sweep_zip(
        param_sweep_zip: params_pb2.ZipSweep) -> Sweep:
    return Zip(*[_sweep_from_single_param_sweep(sweep)
                 for sweep in param_sweep_zip.sweeps])


def _sweep_from_single_param_sweep(
        single_param_sweep: params_pb2.SingleSweep) -> Sweep:
    key = single_param_sweep.parameter_key
    which = single_param_sweep.WhichOneof('sweep')
    if which == 'points':
        sp = single_param_sweep.points
        return Points(key, list(sp.points))
    elif which == 'linspace':
        sl = single_param_sweep.linspace
        return Linspace(key, sl.first_point, sl.last_point, sl.num_points)
    else:
        raise ValueError('unknown single param sweep type: {}'.format(which))
