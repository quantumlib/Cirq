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

# This is a special value to indicate that a type error should be returned.
# This is used within phase_by to raise an error if no underlying
# implementation of _phase_by_ exists.
from cirq._doc import doc_private

RaiseTypeErrorIfNotProvided: Any = ([],)

TDefault = TypeVar('TDefault')


class SupportsPhase(Protocol):
    """An effect that can be phased around the Z axis of target qubits."""

    @doc_private
    def _phase_by_(self: Any, phase_turns: float, qubit_index: int):
        """Returns a phased version of the effect.

        Specifically, returns an object with matrix P U P^-1 (up to global
        phase) where U is the given object's matrix and
        P = Z(qubit_index)**(2 * phase_turns). For example, an X gate phased
        by 90 degrees would be a Y gate.

        Args:
            phase_turns: The amount to phase the gate, in fractions of a whole
                turn. Multiply by 2π to get radians.
            qubit_index: The index of the target qubit the phasing applies to.
        Returns:
            The phased gate or operation.
        """


def phase_by(
    val: Any, phase_turns: float, qubit_index: int, default: TDefault = RaiseTypeErrorIfNotProvided
):
    """Returns a phased version of the effect.

    For example, an X gate phased by 90 degrees would be a Y gate.
    This works by calling `val`'s _phase_by_ method and returning
    the result.

    Args:
        val: The value to describe with a unitary matrix.
        phase_turns: The amount to phase the gate, in fractions of a whole
            turn. Multiply by 2π to get radians.
        qubit_index: The index of the target qubit the phasing applies to. For
            operations this is the index of the qubit within the operation's
            qubit list. For gates it's the index of the qubit within the tuple
            of qubits taken by the gate's `on` method.
        default: The default value to return if `val` can't be phased. If not
            specified, an error is raised when `val` can't be phased.

    Returns:
        If `val` has a _phase_by_ method and its result is not NotImplemented,
        that result is returned. Otherwise, the function will return the
        default value provided or raise a TypeError if none was provided.

    Raises:
        TypeError:
            `val` doesn't have a _phase_by_ method (or that method returned
            NotImplemented) and no `default` was specified.
    """
    getter = getattr(val, '_phase_by_', None)
    result = NotImplemented if getter is None else getter(phase_turns, qubit_index)

    if result is not NotImplemented:
        return result
    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if getter is None:
        raise TypeError(f"object of type '{type(val)}' has no _phase_by_ method.")
    raise TypeError(
        "object of type '{}' does have a _phase_by_ method, "
        "but it returned NotImplemented.".format(type(val))
    )
