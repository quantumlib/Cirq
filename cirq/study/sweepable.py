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

import collections

from typing import cast, Iterable, List, Union

from cirq.study.resolver import ParamResolver
from cirq.study.sweeps import Sweep


Sweepable = Union[
    ParamResolver, Iterable[ParamResolver], Sweep, Iterable[Sweep]]


def to_resolvers(sweepable: Sweepable) -> List[ParamResolver]:
    """Convert a Sweepable to a list of ParamResolvers."""
    if isinstance(sweepable, ParamResolver):
        return [sweepable]
    elif isinstance(sweepable, Sweep):
        return list(sweepable)
    elif isinstance(sweepable, collections.Iterable):
        iterable = cast(collections.Iterable, sweepable)
        return list(iterable) if isinstance(next(iter(iterable)),
                                            ParamResolver) else sum(
            [list(s) for s in iterable], [])
    raise TypeError('Unexpected Sweepable type.')