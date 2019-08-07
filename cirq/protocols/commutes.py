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
"""Protocol for determining commutativity."""

from typing import Any, TypeVar, Union

import numpy as np

from cirq import linalg
from cirq.type_workarounds import NotImplementedType

TDefault = TypeVar('TDefault')


def commutes(left_val: Any,
             right_val,
             atol: Union[int, float] = 1e-8,
             default: TDefault = NotImplemented) -> Union[bool, TDefault]:
    """Determines whether two values commute.

    This is determined by any one of the following techniques:

    - Either value has a `_commutes_with_` method that returns something besides
      NotImplemented. The return value is the boolean value returned by the
      method. left_val._commutes_with_ is attempted first.
    - Both values are matrices. The return value indicates whether these two
      matrices commute.

    If none of these techniques succeeds, it is assumed that the values do not
    commute. The order in which techniques are attempted is
    unspecified.

    Args:
        left_val: The first value.
        right_val: The second value.
        atol: The minimum absolute tolerance. See np.isclose() documentation for
              details. Defaults to 1e-8 which matches np.isclose() default
              absolute tolerance.

    Returns:
        `True` or `False` if `left_val` and `right_val` commute or not,
        respectively. `default` if commutativity cannot be determined;
        `default` defaults to `NotImplemented`.
    """

    strats = [
        _strat_commutes_from_commutes_with,
        _strat_commutes_from_matrix,
    ]
    for strat in strats:
        result = strat(left_val, right_val, atol)
        if result is not NotImplemented:
            return result
    return default


def _strat_commutes_from_commutes_with(left_val: Any,
                                       right_val: Any,
                                       atol: Union[int, float] = 1e-8
                                      ) -> Union[NotImplementedType, bool]:
    """Attempts to determine commutativity via the objects' _commutes_with_
    method."""

    for a, b in [(left_val, right_val), (right_val, left_val)]:
        getter = getattr(a, '_commutes_with_', None)
        if getter is None:
            continue
        val = getter(b, atol)
        if val is not NotImplemented:
            return val
    return NotImplemented


def _strat_commutes_from_matrix(left_val: np.ndarray,
                                right_val: np.ndarray,
                                atol: Union[int, float] = 1e-8
                               ) -> Union[NotImplementedType, bool]:
    """Attempts to determine commutativity of matrices."""
    if not all(isinstance(val, np.ndarray) for val in (left_val, right_val)):
        return NotImplemented
    if left_val.shape != right_val.shape:
        return NotImplemented
    return linalg.commutes(left_val, right_val, atol=atol)
