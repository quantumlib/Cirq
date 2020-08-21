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

"""Defines which types are Sweepable."""

from typing import Dict, Iterable, Iterator, List, Union, cast
import itertools
import sympy

from cirq._doc import document
from cirq.study.resolver import ParamResolver, ParamResolverOrSimilarType
from cirq.study.sweeps import ListSweep, Points, Sweep, UnitSweep, Zip


SweepableDictType = Dict[Union[str, sympy.Symbol], Iterable[
    Union[float, str, sympy.Basic]]]
document(
    SweepableDictType,  # type: ignore
    """Dictionary from symbols to sequence of values taken.""")

Sweepable = Union[ParamResolverOrSimilarType, Sweep, SweepableDictType,
                  Iterable[Union[ParamResolverOrSimilarType, Sweep, SweepableDictType]], None]
document(
    Sweepable,  # type: ignore
    """An object or collection of objects representing a parameter sweep.""")


def to_resolvers(sweepable: Sweepable) -> Iterator[ParamResolver]:
    """Convert a Sweepable to a list of ParamResolvers."""
    for sweep in to_sweeps(sweepable):
        if sweep is None:
            yield ParamResolver({})
        elif isinstance(sweep, ParamResolver):
            yield sweep
        elif isinstance(sweep, Sweep):
            yield from sweep
        elif isinstance(sweep, dict):
            yield ParamResolver(cast(Dict, sweep))
        elif isinstance(sweep, Iterable) and not isinstance(sweep, str):
            for item in cast(Iterable, sweep):
                yield from to_resolvers(item)
        else:
            raise TypeError(f'Unrecognized sweep type: {type(sweep)}.\n'
                            f'sweep: {sweep}')


def to_sweeps(sweepable: Sweepable) -> List[Sweep]:
    """Converts a Sweepable to a list of Sweeps."""
    if sweepable is None:
        return [UnitSweep]
    if isinstance(sweepable, ParamResolver):
        return [_resolver_to_sweep(sweepable)]
    if isinstance(sweepable, Sweep):
        return [sweepable]
    if isinstance(sweepable, dict):
        # change dictionary of lists to list of dictionaries
        # of single values using Cartesian product.
        newsweepable = {}
        for key, value in sweepable.items():
            if isinstance(value, Iterable):
                newsweepable[key] = value
            else:
                newsweepable[key] = [value]
        expandsweepable = [
            dict(zip(newsweepable.keys(), v))
            for v in itertools.product(*newsweepable.values())
        ]
        return [
            _resolver_to_sweep(ParamResolver(cast(Dict, dictitem)))
            for dictitem in expandsweepable
        ]
    if isinstance(sweepable, Iterable) and not isinstance(sweepable, str):
        return [
            sweep for item in sweepable for sweep in to_sweeps(
                cast(Sweepable, item))
        ]
    raise TypeError(f'Unrecognized sweepable type: {type(sweepable)}.\n'
                    f'sweepable: {sweepable}')


def to_sweep(sweep_or_resolver_list: Union[Sweep, ParamResolverOrSimilarType,
                                           Iterable[ParamResolverOrSimilarType]]
            ) -> 'Sweep':
    """Converts the argument into a ``cirq.Sweep``.

    Args:
        sweep_or_resolver_list: The object to try to turn into a
            ``cirq.Sweep`` . A ``cirq.Sweep``, a single ``cirq.ParamResolver``,
            or a list of ``cirq.ParamResolver`` s.

    Returns:
        A sweep equal to or containing the argument.
    """
    if isinstance(sweep_or_resolver_list, Sweep):
        return sweep_or_resolver_list
    if isinstance(sweep_or_resolver_list, (ParamResolver, dict)):
        resolver = cast(ParamResolverOrSimilarType, sweep_or_resolver_list)
        return ListSweep([resolver])
    if isinstance(sweep_or_resolver_list, Iterable):
        resolver_iter = cast(Iterable[ParamResolverOrSimilarType],
                             sweep_or_resolver_list)
        return ListSweep(resolver_iter)
    raise TypeError(
        'Unexpected sweep-like value: {}'.format(sweep_or_resolver_list))


def _resolver_to_sweep(resolver: ParamResolver) -> Sweep:
    params = resolver.param_dict
    if not params:
        return UnitSweep
    return Zip(
        *[Points(key, [cast(float, value)]) for key, value in params.items()])
