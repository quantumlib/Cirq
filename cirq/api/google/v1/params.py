from typing import Iterable, List, Tuple

import numpy as np

from cirq.api.google.v1.params_pb2 import (
    ParameterSweep,
    ParameterSweepZip,
    ParameterSweepZipProduct,
    SingleParameterSweep,
)


Params = Tuple[Tuple[str, float], ...]


def param_sweep_params(param_sweep: ParameterSweep) -> List[str]:
    """Get the list of parameter names in the given parameter sweep."""
    return [sweep.parameter_name
            for factor in param_sweep.sweep.factors
            for sweep in factor.sweeps]


def param_sweep_size(param_sweep: ParameterSweep) -> int:
    """Get the size (number of param settings) for the given parameter sweep."""
    if not param_sweep.HasField('sweep'):
        return 1
    size = 1
    for factor in param_sweep.sweep.factors:
        size *= min(single_param_sweep_size(sweep) for sweep in factor.sweeps)
    return size


def single_param_sweep_size(single_param_sweep: SingleParameterSweep) -> int:
    which = single_param_sweep.WhichOneof('sweep')
    if which == 'sweep_points':
        sweep = single_param_sweep.sweep_points
        return len(sweep.points)
    elif which == 'sweep_linspace':
        sweep = single_param_sweep.sweep_linspace
        return sweep.num_points
    else:
        raise ValueError('unknown single param sweep type: {}'.format(which))


def gen_param_sweep(param_sweep: ParameterSweep) -> Iterable[Params]:
    if not param_sweep.HasField('sweep'):
        return [()]
    return gen_param_sweep_zip_product(param_sweep.sweep)


def gen_param_sweep_zip_product(
        param_sweep_zip_product: ParameterSweepZipProduct) -> Iterable[Params]:
    def _gen(factors):
        if not factors:
            yield ()
        else:
            first, rest = factors[0], factors[1:]
            for first_values in gen_param_sweep_zip(first):
                for rest_values in _gen(rest):
                    yield first_values + rest_values

    return _gen(param_sweep_zip_product.factors)


def gen_param_sweep_zip(
        param_sweep_zip: ParameterSweepZip) -> Iterable[Params]:
    single_param_sweeps = param_sweep_zip.sweeps
    names = [sweep.parameter_name for sweep in single_param_sweeps]
    gens = [gen_single_param_sweep(sweep) for sweep in single_param_sweeps]
    for values in zip(*gens):
        yield tuple(zip(names, values))


def gen_single_param_sweep(
        single_param_sweep: SingleParameterSweep) -> Iterable[float]:
    which = single_param_sweep.WhichOneof('sweep')
    if which == 'sweep_points':
        sweep = single_param_sweep.sweep_points
        return iter(sweep.points)
    elif which == 'sweep_linspace':
        sweep = single_param_sweep.sweep_linspace
        return np.linspace(sweep.first_point, sweep.last_point,
                           sweep.num_points)
    else:
        raise ValueError('unknown single param sweep type: {}'.format(which))
