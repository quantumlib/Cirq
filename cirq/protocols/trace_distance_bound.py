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

from typing import Any, TypeVar
from typing_extensions import Protocol


TDefault = TypeVar('TDefault')


class SupportsTraceDistanceBound(Protocol):
    """An effect with known bounds on how easy it is to detect.

    Used when deciding whether or not an operation is negligible. For example,
    the trace distance between the states before and after a Z**0.00000001
    operation is very close to 0, so it would typically be considered
    negligible.
    """

    def _trace_distance_bound_(self) -> float:
        """A maximum on the trace distance between `val`'s input and output.

        Generally this method is used when deciding whether to keep gates, so
        only the behavior near 0 is important. Approximations that overestimate
        the maximum trace distance are permitted. If, for any case, the bound
        exceeds 1, this function will return 1.  Underestimates are not
        permitted.
        """

def trace_distance_bound(val: Any) -> float:
    """Returns a maximum on the trace distance between this effect's input
    and output.  This method makes use of the effect's `_trace_distance_bound_`
    method to determine the maximum bound on the trace difference between
    before and after the effect.

    Args:
        val: The effect of which the bound should be calculated

    Returns:
        If `val` has a _trace_distance_bound_ method and its result is not
        NotImplemented, that result is returned. Otherwise, 1 is returned.
        Result is capped at a maximum of 1, even if the underlying function
        produces a result greater than 1.

    """
    getter = getattr(val, '_trace_distance_bound_', None)
    result = NotImplemented if getter is None else getter()

    if result is not NotImplemented and result < 1.0:
        return result
    return 1.0
