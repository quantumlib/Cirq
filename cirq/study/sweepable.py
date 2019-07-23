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

from typing import cast, Iterable, List, Union

from cirq.study.resolver import ParamResolver
from cirq.study.sweeps import Points, Sweep, UnitSweep, Zip


Sweepable = Union[
    ParamResolver, Iterable[ParamResolver], Sweep, Iterable[Sweep]]


def to_resolvers(sweepable: Sweepable) -> List[ParamResolver]:
    """Convert a Sweepable to a list of ParamResolvers."""
    if isinstance(sweepable, ParamResolver):
        return [sweepable]
    if isinstance(sweepable, Sweep):
        return list(sweepable)
    if isinstance(sweepable, Iterable):
        iterable = cast(Iterable, sweepable)
        return list(iterable) if isinstance(next(iter(iterable)),
                                            ParamResolver) else sum(
            [list(s) for s in iterable], [])
    raise TypeError('Unexpected Sweepable type.')


def to_sweeps(sweepable: Sweepable) -> List[Sweep]:
    """Converts a Sweepable to a list of Sweeps."""
    if isinstance(sweepable, ParamResolver):
        return [_resolver_to_sweep(sweepable)]
    if isinstance(sweepable, Sweep):
        return [sweepable]
    if isinstance(sweepable, Iterable):
        items = list(sweepable)
        if all(isinstance(item, Sweep) for item in items):
            return cast(List[Sweep], items)
        elif all(isinstance(item, ParamResolver) for item in items):
            return [_resolver_to_sweep(cast(ParamResolver, p)) for p in items]
    raise TypeError('Unexpected Sweepable: {}'.format(sweepable))


def _resolver_to_sweep(resolver: ParamResolver) -> Sweep:
    params = resolver.param_dict
    if not params:
        return UnitSweep
    return Zip(*[Points(key, [value]) for key, value in params.items()])
