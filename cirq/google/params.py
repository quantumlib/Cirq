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
from typing import cast, Dict

from cirq.study.sweeps import (
    Linspace, Points, Product, SingleSweep, Sweep, UnitSweep, Zip,
)


def sweep_to_proto_dict(sweep: Sweep, repetitions: int=1) -> Dict:
    """Converts sweep into an equivalent protobuf representation."""
    msg = {}  # type: Dict
    if not sweep == UnitSweep:
        sweep = _to_zip_product(sweep)
        msg['sweep'] = {
            'factors': [_sweep_zip_to_proto_dict(cast(Zip, factor)) for factor
                        in sweep.factors]
        }
    msg['repetitions'] = repetitions
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


def _sweep_zip_to_proto_dict(
        sweep: Zip) -> Dict:
    sweeps = [_single_param_sweep_to_proto_dict(cast(SingleSweep, s)) for s
              in sweep.sweeps]
    return {'sweeps': sweeps}


def _single_param_sweep_to_proto_dict(
        sweep: SingleSweep,
) -> Dict:
    msg = {}  # type: Dict
    msg['parameter_key'] = sweep.key
    if isinstance(sweep, Linspace):
        msg['linspace'] = {
            'first_point': sweep.start,
            'last_point': sweep.stop,
            'num_points': sweep.length
        }
    elif isinstance(sweep, Points):
        msg['points'] = {'points': sweep.points}
    else:
        raise ValueError('invalid single-parameter sweep: {}'.format(sweep))
    return msg


def sweep_from_proto_dict(param_sweep: Dict) -> Sweep:
    if 'sweep' in param_sweep and 'factors' in param_sweep['sweep']:
        return Product(*[_sweep_from_param_sweep_zip_proto_dict(f)
                         for f in param_sweep['sweep']['factors']])
    return UnitSweep


def _sweep_from_param_sweep_zip_proto_dict(
        param_sweep_zip: Dict) -> Sweep:
    if 'sweeps' in param_sweep_zip:
        return Zip(*[_sweep_from_single_param_sweep_proto_dict(sweep)
                     for sweep in param_sweep_zip['sweeps']])
    return UnitSweep


def _sweep_from_single_param_sweep_proto_dict(
        single_param_sweep: Dict) -> Sweep:
    key = single_param_sweep['parameter_key']
    if 'points' in single_param_sweep:
        points = single_param_sweep['points']
        return Points(key, list(points['points']))
    elif 'linspace' in single_param_sweep:
        sl = single_param_sweep['linspace']
        return Linspace(key, sl['first_point'], sl['last_point'],
                        sl['num_points'])
    else:
        raise ValueError('Single param sweep type undefined')
