# Copyright 2019 The Cirq Developers
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
"""Protocol for object that have measurement keys."""

from typing import Any, Iterable, Union, TYPE_CHECKING, Tuple, Dict, List, \
    Optional, DefaultDict

from typing_extensions import Protocol

from cirq._doc import document
from cirq.protocols.channel import has_channel
import numpy as np

from cirq.protocols.decompose_protocol import \
    _try_decompose_into_operations_and_qubits

if TYPE_CHECKING:
    import cirq

# This is a special indicator value used by the inverse method to determine
# whether or not the caller provided a 'default' argument.
RaiseTypeErrorIfNotProvided = ([],)  # type: Any


class SupportsMeasurementKey(Protocol):
    r"""An object that is a measurement and has a measurement key.

    Measurement keys are used in referencing the results of a measurement.

    Note: Measurements, in contrast to general quantum channels, are
    distinguished by the recording of the quantum operation that occurred.
    That is a general quantum channel may enact the evolution
        \rho \rightarrow \sum_k A_k \rho A_k^\dagger
    where as a measurement enacts the evolution
        \rho \rightarrow A_k \rho A_k^\dagger
    conditional on the measurement outcome being k.
    """

    @document
    def _measurement_key_(self) -> str:
        """Return the key that will be used to identify this measurement.

        When a measurement occurs, either on hardware, or in a simulation,
        this is the key value under which the results of the measurement
        will be stored.
        """

    @document
    def _measurement_keys_(self) -> Iterable[str]:
        """Return the key that will be used to identify this measurement.

        When a measurement occurs, either on hardware, or in a simulation,
        this is the key value under which the results of the measurement
        will be stored.
        """


def measurement_key(val: Any, default: Any = RaiseTypeErrorIfNotProvided):
    """Get the measurement key for the given value.

    Args:
        val: The value which has the measurement key..
        default: Determines the fallback behavior when `val` doesn't have
            a measurement key. If `default` is not set, a TypeError is raised.
            If default is set to a value, that value is returned if the value
            does not have `_measurement_key_`.

    Returns:
        If `val` has a `_measurement_key_` method and its result is not
        `NotImplemented`, that result is returned. Otherwise, if a default
        value was specified, the default value is returned.

    Raises:
        TypeError: `val` doesn't have a _measurement_key_ method (or that method
            returned NotImplemented) and also no default value was specified.
    """
    result = measurement_keys(val)

    if len(result) == 1:
        return result[0]

    if len(result) > 1:
        raise TypeError(f'Got multiple measurement keys ({result!r}) '
                        f'from {val!r}.')

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    raise TypeError(f"object of type '{type(val)}' had no measurement keys.")


def measurement_keys(val: Any,
                     include_decompose: bool = True) -> Tuple[str, ...]:
    """Get the measurement key for the given value.

    Args:
        val: The value which has the measurement key..
        default: Determines the fallback behavior when `val` doesn't have
            a measurement key. If `default` is not set, a TypeError is raised.
            If default is set to a value, that value is returned if the value
            does not have `_measurement_key_`.

    Returns:
        If `val` has a `_measurement_key_` method and its result is not
        `NotImplemented`, that result is returned. Otherwise, if a default
        value was specified, the default value is returned.
    """
    getter = getattr(val, '_measurement_keys_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return tuple(result)

    getter = getattr(val, '_measurement_key_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return result,

    if include_decompose:
        operations, _, _ = _try_decompose_into_operations_and_qubits(val)
        if operations is not None:
            return tuple(key for op in operations for key in measurement_keys(op))

    return ()


def is_measurement(val: Any, include_decompose: bool = True) -> bool:
    """Returns whether or not the given value is a measurement.

    A measurement must implement the `measurement_key` protocol and have a
    channel, as represented by the `has_channel` protocol returning true.
    """
    return bool(measurement_keys(val, include_decompose=include_decompose))
