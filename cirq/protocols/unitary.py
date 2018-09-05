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

from typing import Any, Optional, TypeVar, Union

import numpy as np
from typing_extensions import Protocol

from cirq import abc


# This is a special value used by the unitary method to determine whether or not
# the caller provided a 'default' argument. It must of type np.ndarray to ensure
# the method has the correct type signature in that case. It is checked for
# using `is`, so it won't have a false positive if the user provides a different
# np.array([]) value.
RaiseTypeErrorIfNotProvided = np.array([])

TDefault = TypeVar('TDefault')


class SupportsUnitary(Protocol):
    """An object that may be describable by a unitary matrix."""

    @abc.abstractmethod
    def _unitary_(self) -> Optional[np.ndarray]:
        """A unitary matrix describing this value, e.g. the matrix of a gate.

        This method is used by the global cirq.unitary method. If this method is
        not present, or returns None, it is assumed that the receiving object
        doesn't have a unitary matrix (resulting in a TypeError being raised by
        the cirq.unitary method).

        Returns:
            A unitary matrix describing this value, or None if there is no such
            matrix.
        """


def unitary(val: Any,
            default: TDefault = RaiseTypeErrorIfNotProvided
            ) -> Union[np.ndarray, TDefault]:
    """Returns a unitary matrix describing the given value.

    Args:
        val: The value to describe with a unitary matrix.
        default: Determines the fallback behavior when `val` doesn't have
            a unitary matrix. If `default` is not set, a TypeError is raised. If
            default is set to a value, that value is returned.

    Returns:
        If `val` has a _unitary_ method and its result is not None, that result
        is returned. Otherwise, if a default value was specified, the default
        value is returned.

    Raises:
        TypeError: `val` doesn't have a _unitary_ method (or that method
            returned None) and also no default value was specified.
    """
    get = getattr(val, '_unitary_', None)
    result = None if get is None else get()

    if result is not None:
        return result
    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if get is None:
        raise TypeError("object of type '{}' "
                        "has no _unitary_ method.".format(type(val)))
    raise TypeError("object of type '{}' does have a _unitary_ method, "
                    "but it returned None.".format(type(val)))
