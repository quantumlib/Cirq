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

from typing import Any, TypeVar, Optional, Sequence, Union

import numpy as np
from typing_extensions import Protocol

from cirq._doc import doc_private
from cirq.protocols import unitary_protocol

TDefault = TypeVar('TDefault')


class SupportsTraceDistanceBound(Protocol):
    """An effect with known bounds on how easy it is to detect.

    Used when deciding whether or not an operation is negligible. For example,
    the trace distance between the states before and after a Z**0.00000001
    operation is very close to 0, so it would typically be considered
    negligible.
    """

    @doc_private
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
    and output.

    This method attempts a number of strategies to calculate this value.

    Strategy 1:
        Use the effect's `_trace_distance_bound_` method.

    Strategy 2:
        If the effect is unitary, calculate the trace distance bound from the
        eigenvalues of the unitary matrix.

    Args:
        val: The effect of which the bound should be calculated

    Returns:
        If any of the strategies return a result that is not Notimplemented and
        not None, that result is returned. Otherwise, 1.0 is returned.
        Result is capped at a maximum of 1.0, even if the underlying function
        produces a result greater than 1.0

    """
    strats = [_strat_from_trace_distance_bound_method, _strat_distance_from_unitary]

    for strat in strats:
        result = strat(val)
        if result is None:
            break
        if result is not NotImplemented:
            return result

    return 1.0


def _strat_from_trace_distance_bound_method(val: Any) -> Optional[float]:
    """Attempts to use a specialized method."""
    getter = getattr(val, '_trace_distance_bound_', None)
    result = NotImplemented if getter is None else getter()

    if result is None:
        return None

    if result is not NotImplemented:
        return min(1.0, result)

    return NotImplemented


def _strat_distance_from_unitary(val: Any) -> Optional[float]:
    """Attempts to compute a value's trace_distance_bound from its unitary."""
    u = unitary_protocol.unitary(val, default=None)

    if u is None:
        return NotImplemented

    if u.shape == (2, 2):
        squared = 1 - (0.5 * abs(u[0][0] + u[1][1])) ** 2
        if squared <= 0:
            return 0.0
        return squared**0.5

    return trace_distance_from_angle_list(np.angle(np.linalg.eigvals(u)))  # type: ignore[arg-type]


def trace_distance_from_angle_list(angle_list: Union[Sequence[float], np.ndarray]) -> float:
    """Given a list of arguments of the eigenvalues of a unitary matrix,
    calculates the trace distance bound of the unitary effect.

    The maximum provided angle should not exceed the minimum provided angle
    by more than 2π.
    """
    angles = np.sort(angle_list)
    maxim = 2 * np.pi + angles[0] - angles[-1]
    for i in range(1, len(angles)):
        maxim = max(maxim, angles[i] - angles[i - 1])
    if maxim <= np.pi:
        return 1.0
    return max(0.0, np.sin(0.5 * maxim))
