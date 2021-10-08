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

from typing import AbstractSet, Any, Dict, Optional, Tuple

from typing_extensions import Protocol

from cirq._compat import deprecated, deprecated_parameter, _warn_or_error
from cirq._doc import doc_private
from cirq import value

# This is a special indicator value used by the inverse method to determine
# whether or not the caller provided a 'default' argument.
RaiseTypeErrorIfNotProvided = ([],)  # type: Any


class SupportsMeasurementKey(Protocol):
    r"""An object that is a measurement and has a measurement key or keys.

    Measurement keys are used in referencing the results of a measurement.

    Users are free to implement one of the following. Do not implement multiple
    of these returning different values. The protocol behavior will be
    unexpected in such a case.
    1. `_measurement_key_objs_` returning an iterable of `MeasurementKey`s
    2. `_measurement_key_obj_` returning one `MeasurementKey`
    3. `_measurement_key_names_` returning an iterable of strings
    4. `_measurement_key_name_` returning one string

    Note: Measurements, in contrast to general quantum channels, are
    distinguished by the recording of the quantum operation that occurred.
    That is a general quantum channel may enact the evolution
        $$
        \rho \rightarrow \sum_k A_k \rho A_k^\dagger
        $$
    where as a measurement enacts the evolution
        $$
        \rho \rightarrow A_k \rho A_k^\dagger
        $$
    conditional on the measurement outcome being $k$.
    """

    @doc_private
    def _is_measurement_(self) -> bool:
        """Return if this object is (or contains) a measurement."""

    @doc_private
    def _measurement_key_obj_(self) -> value.MeasurementKey:
        """Return the key object that will be used to identify this measurement.

        When a measurement occurs, either on hardware, or in a simulation,
        this is the key value under which the results of the measurement
        will be stored.
        """

    @doc_private
    def _measurement_key_objs_(self) -> AbstractSet[value.MeasurementKey]:
        """Return the key objects for measurements performed by the receiving object.

        When a measurement occurs, either on hardware, or in a simulation,
        these are the key values under which the results of the measurements
        will be stored.
        """

    @doc_private
    def _measurement_key_name_(self) -> str:
        """Return the string key that will be used to identify this measurement.

        When a measurement occurs, either on hardware, or in a simulation,
        this is the key value under which the results of the measurement
        will be stored.
        """

    @doc_private
    def _measurement_key_names_(self) -> AbstractSet[str]:
        """Return the string keys for measurements performed by the receiving object.

        When a measurement occurs, either on hardware, or in a simulation,
        these are the key values under which the results of the measurements
        will be stored.
        """

    @doc_private
    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]):
        """Return a copy of this object with the measurement keys remapped.

        This method allows measurement keys to be reassigned at runtime.
        """


@deprecated(deadline='v0.13', fix='use cirq.measurement_key_name instead')
def measurement_key(val: Any, default: Any = RaiseTypeErrorIfNotProvided):
    return measurement_key_name(val, default)


def measurement_key_obj(val: Any, default: Any = RaiseTypeErrorIfNotProvided):
    """Get the single measurement key object for the given value.

    Args:
        val: The value which has one measurement key.
        default: Determines the fallback behavior when `val` doesn't have
            a measurement key. If `default` is not set, a TypeError is raised.
            If default is set to a value, that value is returned if the value
            does not have `_measurement_key_name_`.

    Returns:
        If `val` has a `_measurement_key_obj[s]_` method and its result is not
        `NotImplemented`, that result is returned. Otherwise, if a default
        value was specified, the default value is returned.

    Raises:
        TypeError: `val` doesn't have a _measurement_key_obj[s]_ method (or that method
            returned NotImplemented) and also no default value was specified.
        ValueError: `val` has multiple measurement keys.
    """
    result = measurement_key_objs(val)

    if len(result) == 1:
        return next(iter(result))

    if len(result) > 1:
        raise ValueError(f'Got multiple measurement keys ({result!r}) from {val!r}.')

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    raise TypeError(f"Object of type '{type(val)}' had no measurement keys.")


def measurement_key_name(val: Any, default: Any = RaiseTypeErrorIfNotProvided):
    """Get the single measurement key for the given value.

    Args:
        val: The value which has one measurement key.
        default: Determines the fallback behavior when `val` doesn't have
            a measurement key. If `default` is not set, a TypeError is raised.
            If default is set to a value, that value is returned if the value
            does not have `_measurement_key_name_`.

    Returns:
        If `val` has a `_measurement_key_name_` method and its result is not
        `NotImplemented`, that result is returned. Otherwise, if a default
        value was specified, the default value is returned.

    Raises:
        TypeError: `val` doesn't have a _measurement_key_name_ method (or that method
            returned NotImplemented) and also no default value was specified.
        ValueError: `val` has multiple measurement keys.
    """
    result = measurement_key_names(val)

    if len(result) == 1:
        return next(iter(result))

    if len(result) > 1:
        raise ValueError(f'Got multiple measurement keys ({result!r}) from {val!r}.')

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    raise TypeError(f"Object of type '{type(val)}' had no measurement keys.")


def _measurement_key_objs_from_magic_methods(
    val: Any,
) -> Optional[AbstractSet[value.MeasurementKey]]:
    """Uses the measurement key related magic methods to get the `MeasurementKey`s for this
    object."""

    getter = getattr(val, '_measurement_key_objs_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return set(result)

    getter = getattr(val, '_measurement_key_obj_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return {result}
    return result


def _measurement_key_names_from_magic_methods(val: Any) -> Optional[AbstractSet[str]]:
    """Uses the measurement key related magic methods to get the key strings for this object."""

    getter = getattr(val, '_measurement_key_names_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return set(result)
    getter = getattr(val, '_measurement_keys_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        _warn_or_error(
            f'_measurement_keys_ was used but is deprecated.\n'
            f'It will be removed in cirq v0.13.\n'
            f'Use _measurement_key_names_ instead.\n'
        )
        return set(result)

    getter = getattr(val, '_measurement_key_name_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return {result}
    getter = getattr(val, '_measurement_key_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        _warn_or_error(
            f'_measurement_key_ was used but is deprecated.\n'
            f'It will be removed in cirq v0.13.\n'
            f'Use _measurement_key_name_ instead.\n'
        )
        return {result}

    return result


@deprecated(deadline='v0.13', fix='use cirq.measurement_key_names instead')
def measurement_keys(val: Any, *, allow_decompose: bool = True):
    return measurement_key_names(val, allow_decompose=allow_decompose)


def measurement_key_objs(val: Any) -> AbstractSet[value.MeasurementKey]:
    """Gets the measurement key objects of measurements within the given value.

    Args:
        val: The value which has the measurement key.

    Returns:
        The measurement key objects of the value. If the value has no measurement,
        the result is the empty set.
    """
    result = _measurement_key_objs_from_magic_methods(val)
    if result is not NotImplemented and result is not None:
        return result
    key_strings = _measurement_key_names_from_magic_methods(val)
    if key_strings is not NotImplemented and key_strings is not None:
        return {value.MeasurementKey.parse_serialized(key_str) for key_str in key_strings}
    return set()


@deprecated_parameter(
    deadline='v0.14',
    fix='This protocol no longer uses decomposition, so allow_decompose should be removed',
    func_name='measurement_key_names',
    parameter_desc='allow_decompose',
    match=lambda args, kwargs: 'allow_decompose' in kwargs,
)
def measurement_key_names(val: Any, *, allow_decompose: bool = True) -> AbstractSet[str]:
    """Gets the measurement key strings of measurements within the given value.

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
        the result is the empty set.
    """
    result = _measurement_key_names_from_magic_methods(val)
    if result is not NotImplemented and result is not None:
        return result
    key_objs = _measurement_key_objs_from_magic_methods(val)
    if key_objs is not NotImplemented and key_objs is not None:
        return {str(key_obj) for key_obj in key_objs}
    return set()


def _is_measurement_from_magic_method(val: Any) -> Optional[bool]:
    """Uses `is_measurement` magic method to determine if this object is a measurement."""
    getter = getattr(val, '_is_measurement_', None)
    return NotImplemented if getter is None else getter()


@deprecated_parameter(
    deadline='v0.14',
    fix='This protocol no longer uses decomposition, so allow_decompose should be removed',
    func_name='is_measurement',
    parameter_desc='allow_decompose',
    match=lambda args, kwargs: 'allow_decompose' in kwargs,
)
def is_measurement(val: Any, allow_decompose: bool = True) -> bool:
    """Determines whether or not the given value is a measurement (or contains one).

    Measurements are identified by the fact that any of them may have an `_is_measurement_` method
    or `cirq.measurement_keys` returns a non-empty result for them.

    Args:
        val: The value which to evaluate.
        allow_decompose: Defaults to True. When true, composite operations that
            don't directly specify their `_is_measurement_` property will be decomposed in
            order to find any measurements keys within the decomposed operations.
    """
    result = _is_measurement_from_magic_method(val)
    if isinstance(result, bool):
        return result

    keys = measurement_key_objs(val)
    return keys is not NotImplemented and bool(keys)


def with_measurement_key_mapping(val: Any, key_map: Dict[str, str]):
    """Remaps the target's measurement keys according to the provided key_map.

    This method can be used to reassign measurement keys at runtime, or to
    assign measurement keys from a higher-level object (such as a Circuit).
    """
    getter = getattr(val, '_with_measurement_key_mapping_', None)
    return NotImplemented if getter is None else getter(key_map)


def with_key_path(val: Any, path: Tuple[str, ...]):
    """Adds the path to the target's measurement keys.

    The path usually refers to an identifier or a list of identifiers from a subcircuit that
    used to contain the target. Since a subcircuit can be repeated and reused, these paths help
    differentiate the actual measurement keys.
    """
    getter = getattr(val, '_with_key_path_', None)
    return NotImplemented if getter is None else getter(path)
