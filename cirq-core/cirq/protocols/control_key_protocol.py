# Copyright 2021 The Cirq Developers
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
"""Protocol for object that have control keys."""

from typing import Any, FrozenSet, TYPE_CHECKING, Union

from typing_extensions import Protocol

from cirq._doc import doc_private
from cirq.protocols import measurement_key_protocol
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq


class SupportsControlKey(Protocol):
    """An object that is a has a classical control key or keys.

    Control keys are used in referencing the results of a measurement.

    Users should implement `_control_keys_` returning an iterable of
    `MeasurementKey`.
    """

    @doc_private
    def _control_keys_(self) -> Union[FrozenSet['cirq.MeasurementKey'], NotImplementedType, None]:
        """Return the keys for controls referenced by the receiving object.

        Returns:
            The measurement keys the value is controlled by. If the value is not
            classically controlled, the result is the empty tuple.
        """


def control_keys(val: Any) -> FrozenSet['cirq.MeasurementKey']:
    """Gets the keys that the value is classically controlled by.

    Args:
        val: The object that may be classically controlled.

    Returns:
        The measurement keys the value is controlled by. If the value is not
        classically controlled, the result is the empty tuple.
    """
    getter = getattr(val, '_control_keys_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return result

    return frozenset()


def measurement_keys_touched(val: Any) -> FrozenSet['cirq.MeasurementKey']:
    """Returns all the measurement keys used by the value.

    This would be the case if the value is or contains a measurement gate, or
    if the value is or contains a conditional operation.

    Args:
        val: The object that may interact with measurements.

    Returns:
        The measurement keys used by the value..
    """
    return measurement_key_protocol.measurement_key_objs(val) | control_keys(val)
