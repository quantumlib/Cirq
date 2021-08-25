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

from typing import AbstractSet, Any, Iterable

from typing_extensions import Protocol

from cirq._doc import doc_private
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits


class SupportsControlKey(Protocol):
    """An object that is a control and has a control key or keys.

    Control keys are used in referencing the results of a measurement.

    Users are free to implement either `_control_key_names_` returning an
    iterable of strings.
    """

    @doc_private
    def _control_key_names_(self) -> Iterable[str]:
        """Return the keys for controls performed by the receiving object.

        When a control occurs, either on hardware, or in a simulation,
        these are the key values under which the results of the controls
        will be stored.
        """


def control_key_names(val: Any, *, allow_decompose: bool = True) -> AbstractSet[str]:
    """Gets the control keys of controls within the given value.

    Args:
        val: The value which has the control key.
        allow_decompose: Defaults to True. When true, composite operations that
            don't directly specify their control keys will be decomposed in
            order to find control keys within the decomposed operations. If
            not set, composite operations will appear to have no control
            keys. Used by internal methods to stop redundant decompositions from
            being performed.

    Returns:
        The control keys of the value. If the value has no control,
        the result is the empty tuple.
    """
    getter = getattr(val, '_control_key_names_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return set(result)

    if allow_decompose:
        operations, _, _ = _try_decompose_into_operations_and_qubits(val)
        if operations is not None:
            return {key for op in operations for key in control_key_names(op)}

    return set()
