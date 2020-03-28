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

from cirq._doc import document
from cirq.study.resolver import ParamResolver, ParamResolverOrSimilarType
from cirq.study.sweeps import ListSweep, Points, Sweep, UnitSweep, Zip


Sweepable = Union[Dict[str, float], ParamResolver, Sweep, Iterable[
    Union[Dict[str, float], ParamResolver, Sweep]], None]
document(
    Sweepable,  # type: ignore
    """An object or collection of objects representing a parameter sweep.""")


def to_resolvers(sweepable: Sweepable) -> Iterator[ParamResolver]:
    """Convert a Sweepable to a list of ParamResolvers."""
    if sweepable is None:
        yield ParamResolver({})
    elif isinstance(sweepable, ParamResolver):
        yield sweepable
    elif isinstance(sweepable, Sweep):
        yield from sweepable
    elif isinstance(sweepable, dict):
        yield ParamResolver(cast(Dict, sweepable))
    elif isinstance(sweepable, Iterable) and not isinstance(sweepable, str):
        for item in cast(Iterable, sweepable):
            yield from to_resolvers(item)
    else:
        raise TypeError(f'Unrecognized sweepable type: {type(sweepable)}.\n'
                        f'sweepable: {sweepable}')


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
                cast(Union[Dict[str, float], ParamResolver, Sweep], item))
        ]
    raise TypeError(f'Unrecognized sweepable type: {type(sweepable)}.\n'
                    f'sweepable: {sweepable}')


def to_sweep(sweep_or_resolver_list: Union['Sweep', ParamResolverOrSimilarType,
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
