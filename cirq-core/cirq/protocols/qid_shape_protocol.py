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

from types import NotImplementedType
from typing import Any, Sequence, Tuple, TypeVar, Union

from typing_extensions import Protocol

from cirq import ops
from cirq._doc import document, doc_private

# This is a special indicator value used by the methods to determine whether or
# not the caller provided a 'default' argument. It must be of type
# Tuple[int, ...] to ensure the method has the correct type signature in that
# case. It is checked for using `is`, so it won't have a false positive if the
# user provides a different (0,) value.
RaiseTypeErrorIfNotProvided: Any = (0,)
# Equal integers outside the range [-5, 256] aren't identically equal with `is`.
RaiseTypeErrorIfNotProvidedInt: Any = -(2**512)

TDefault = TypeVar('TDefault')


class SupportsExplicitQidShape(Protocol):
    """A unitary, channel, mixture or other object that operates on a known
    number qubits/qudits/qids, each with a specific number of quantum levels."""

    @doc_private
    def _qid_shape_(self) -> Union[Tuple[int, ...], NotImplementedType]:
        """A tuple specifying the number of quantum levels of each qid this
        object operates on, e.g. (2, 2, 2) for a three-qubit gate.

        This method is used by the global `cirq.qid_shape` method (and by
        `cirq.num_qubits` if `_num_qubits_` is not defined). If this
        method is not present, or returns NotImplemented, it is assumed that the
        receiving object operates on qubits. (The ability to return
        NotImplemented is useful when a class cannot know if it has a shape
        until runtime.)

        The order of values in the tuple is always implicit with respect to the
        object being called. For example, for gates the tuple must be ordered
        with respect to the list of qubits that the gate is applied to. For
        operations, the tuple is ordered to match the list returned by its
        `qubits` attribute.

        Returns:
            The qid shape of this value, or NotImplemented if the shape is
            unknown.
        """


class SupportsExplicitNumQubits(Protocol):
    """A unitary, channel, mixture or other object that operates on a known
    number of qubits."""

    @document
    def _num_qubits_(self) -> Union[int, NotImplementedType]:
        """The number of qubits, qudits, or qids this object operates on.

        This method is used by the global `cirq.num_qubits` method (and by
        `cirq.qid_shape` if `_qid_shape_` is not defined.  If this
        method is not present, or returns NotImplemented, it will fallback
        to using the length of `_qid_shape_`.

        Returns:
            An integer specifying the number of qubits, qudits or qids.
        """


def qid_shape(
    val: Any, default: TDefault = RaiseTypeErrorIfNotProvided
) -> Union[Tuple[int, ...], TDefault]:
    """Returns a tuple describing the number of quantum levels of each
    qubit/qudit/qid `val` operates on.

    Args:
        val: The value to get the shape of.
        default: Determines the fallback behavior when `val` doesn't have
            a shape. If `default` is not set, a TypeError is raised. If
            default is set to a value, that value is returned.

    Returns:
        If `val` has a `_qid_shape_` method and its result is not
        NotImplemented, that result is returned. Otherwise, if `val` has a
        `_num_qubits_` method, the shape with `num_qubits` qubits is returned
        e.g. `(2,)*num_qubits`. If neither method returns a value other than
        NotImplemented and a default value was specified, the default value is
        returned.

    Raises:
        TypeError: `val` doesn't have either a `_qid_shape_` or a `_num_qubits_`
            method (or they returned NotImplemented) and also no default value
            was specified.
    """
    getter = getattr(val, '_qid_shape_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented:
        return result

    # Check if val is a list of qids
    if isinstance(val, Sequence) and all(isinstance(q, ops.Qid) for q in val):
        return tuple(q.dimension for q in val)

    # Fallback to _num_qubits_
    num_getter = getattr(val, '_num_qubits_', None)
    num_qubits = NotImplemented if num_getter is None else num_getter()
    if num_qubits is not NotImplemented:
        return (2,) * num_qubits

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if getter is not None:
        raise TypeError(
            f"object of type '{type(val)}' does have a _qid_shape_ method, "
            "but it returned NotImplemented."
        )
    if num_getter is not None:
        raise TypeError(
            f"object of type '{type(val)}' does have a _num_qubits_ method, "
            "but it returned NotImplemented."
        )
    raise TypeError(f"object of type '{type(val)}' has no _num_qubits_ or _qid_shape_ methods.")


def num_qubits(
    val: Any, default: TDefault = RaiseTypeErrorIfNotProvidedInt
) -> Union[int, TDefault]:
    """Returns the number of qubits, qudits, or qids `val` operates on.

    Args:
        val: The value to get the number of qubits from.
        default: Determines the fallback behavior when `val` doesn't have
            a number of qubits. If `default` is not set, a TypeError is raised.
            If default is set to a value, that value is returned.

    Returns:
        If `val` has a `_num_qubits_` method and its result is not
        NotImplemented, that result is returned. Otherwise, if `val` has a
        `_qid_shape_` method, the number of qubits is computed from the length
        of the shape and returned e.g. `len(shape)`. If neither method returns a
        value other than NotImplemented and a default value was specified, the
        default value is returned.

    Raises:
        TypeError: `val` doesn't have either a `_num_qubits_` or a `_qid_shape_`
            method (or they returned NotImplemented) and also no default value
            was specified.
    """
    num_getter = getattr(val, '_num_qubits_', None)
    num_qubits = NotImplemented if num_getter is None else num_getter()
    if num_qubits is not NotImplemented:
        return num_qubits

    # Fallback to _qid_shape_
    getter = getattr(val, '_qid_shape_', None)
    shape = NotImplemented if getter is None else getter()
    if shape is not NotImplemented:
        return len(shape)

    # Check if val is a list of qids
    if isinstance(val, Sequence) and all(isinstance(q, ops.Qid) for q in val):
        return len(val)

    if default is not RaiseTypeErrorIfNotProvidedInt:
        return default

    if num_getter is not None:
        raise TypeError(
            f"object of type '{type(val)}' does have a _num_qubits_ method, "
            "but it returned NotImplemented."
        )
    if getter is not None:
        raise TypeError(
            f"object of type '{type(val)}' does have a _qid_shape_ method, "
            "but it returned NotImplemented."
        )
    raise TypeError(f"object of type '{type(val)}' has no _num_qubits_ or _qid_shape_ methods.")
