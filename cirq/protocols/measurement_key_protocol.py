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

from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

from cirq._doc import document

from cirq.protocols.decompose_protocol import \
    _try_decompose_into_operations_and_qubits


# This is a special indicator value used by the inverse method to determine
# whether or not the caller provided a 'default' argument.
RaiseTypeErrorIfNotProvided = ([],)  # type: Any


class SupportsMeasurementKey(Protocol):
    r"""An object that is a measurement and has a measurement key or keys.

    Measurement keys are used in referencing the results of a measurement.

    Users are free to implement either `_measurement_key_` returning one string
    or `_measurement_keys_` returning an iterable of strings.

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
        """Return the keys for measurements performed by the receiving object.

        When a measurement occurs, either on hardware, or in a simulation,
        these are the key values under which the results of the measurements
        will be stored.
        """


def measurement_key(val: Any, default: Any = RaiseTypeErrorIfNotProvided):
    """Get the single measurement key for the given value.

    Args:
        val: The value which has one measurement key.
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
        ValueError: `val` has multiple measurement keys.
    """
    result = measurement_keys(val)

    if len(result) == 1:
        return result[0]

    if len(result) > 1:
        raise ValueError(f'Got multiple measurement keys ({result!r}) '
                         f'from {val!r}.')

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    raise TypeError(f"Object of type '{type(val)}' had no measurement keys.")


def measurement_keys(val: Any, *,
                     allow_decompose: bool = True) -> Tuple[str, ...]:
    """Gets the measurement keys of measurements within the given value.

    Args:
        val: The value which has the measurement key.
        allow_decompose: Defaults to True. When true, composite operations that
            don't directly specify their measurement keys will be decomposed in
            order to find measurement keys within the decomposed operations. If
            not set, composite operations will appear to have no measurement
            keys. Used by internal methods to stop redundant decompositions from
            being performed.

    Returns:
        The measurement keys of the value. If the value has no measurement,
        the result is the empty tuple.
    """
    getter = getattr(val, '_measurement_keys_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return tuple(result)

    getter = getattr(val, '_measurement_key_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return result,

    if allow_decompose:
        operations, _, _ = _try_decompose_into_operations_and_qubits(val)
        if operations is not None:
            return tuple(
                key for op in operations for key in measurement_keys(op))

    return ()


def is_measurement(val: Any) -> bool:
    """Determines whether or not the given value is a measurement.

    Measurements are identified by the fact that `cirq.measurement_keys` returns
    a non-empty result for them.
    """
    return bool(measurement_keys(val))
