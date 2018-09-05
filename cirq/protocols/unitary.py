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

from typing import Any, Optional

import numpy as np
from typing_extensions import Protocol

from cirq import abc


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


def try_get_unitary(val: Any) -> Optional[np.ndarray]:
    """Returns a unitary matrix describing the given value, or else None.

    Returns:
        If the given value has a _unitary_ method, its result is returned.
        Otherwise None is returned.
    """
    get = getattr(val, '_unitary_', None)
    if get is None:
        return None
    return get()


def unitary(val: Any) -> np.ndarray:
    """Returns a unitary matrix describing the given value, or else raises.

    Returns:
        If the given value has a _unitary_ method and its result is not None,
        that result is returned.

    Raises:
        TypeError: The given value does not have a _unitary_ method, or that
            method returned None.
    """
    get = getattr(val, '_unitary_', None)
    if get is None:
        raise TypeError("object of type '{}' has no _unitary_ method.".format(
            type(val)))
    result = get()
    if result is None:
        raise TypeError("object of type '{}' does have a _unitary_ method, "
                        "but it returned None.".format(type(val)))
    return result
