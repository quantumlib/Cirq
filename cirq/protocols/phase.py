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

# This is a special value to indicate that the gate should be returned
# unchanged if underlying implementation of _phase_by_ exists.
ReturnSelfIfNotProvided = 1.234567

TDefault = TypeVar('TDefault')


class SupportsPhase(Protocol):
    """An effect that can be phased around the Z axis of target qubits."""

    def _phase_by_(self: Any, phase_turns: float, qubit_index: int):
        """Returns a phased version of the effect.
        For example, an X gate phased by 90 degrees would be a Y gate.
        Args:
            phase_turns: The amount to phase the gate, in fractions of a whole
                turn.
            qubit_index: The index of the target qubit the phasing applies to.
        Returns:
            The phased gate or operation.
        """


def phase_by(val: Any, phase_turns: float, qubit_index: int,
             default: TDefault = ReturnSelfIfNotProvided):
    """Returns a phased version of the effect.
    For example, an X gate phased by 90 degrees would be a Y gate.
    This works by calling `val`'s _phase_by_ method and returning
    the result.

    Args:
        val: The value to describe with a unitary matrix.
        phase_turns: The amount to phase the gate, in fractions of a whole
            turn.
        qubit_index: The index of the target qubit the phasing applies to.

    Returns:
        If `val` has a _phase_by_ method and its result is not NotImplemented,
        that result is returned. Otherwise, `val` is returned unchanged.
    """
    getter = getattr(val, '_phase_by_', None)
    result = NotImplemented if getter is None else getter(
        phase_turns, qubit_index)

    if result is not NotImplemented:
        return result
    if default is ReturnSelfIfNotProvided:
        return val
    else:
        return default
