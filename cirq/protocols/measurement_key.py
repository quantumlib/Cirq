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

from typing import Any

from typing_extensions import Protocol

from cirq.protocols import has_channel

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

    def _measurement_key_(self) -> str:
        """Return the key that will be used to identify this measurement.

        When a measurement occurs, either on hardware, or in a simulation,
        this is the key value under which the results of the measurement
        will be stored.
        """


def measurement_key(
        val: Any,
        default: Any = RaiseTypeErrorIfNotProvided):
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
    getter = getattr(val, '_measurement_key_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented:
        return result

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if getter is None:
        raise TypeError(
                "object of type '{}' has no _measurement_key_ method."
                    .format(type(val)))

    raise TypeError("object of type '{}' does have a _measurement_key_ method, "
                    "but it returned NotImplemented.".format(type(val)))


def is_measurement(val: Any) -> bool:
    """Returns whether or not the given value is a measurement.

    A measurement must implement the `measurement_key` protocol and have a
    channel, as represented by the `has_channel` protocol returning true.
    """
    return measurement_key(val, None) is not None and has_channel(val)
