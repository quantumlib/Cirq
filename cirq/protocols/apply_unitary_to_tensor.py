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

from typing import Any, Union, Sequence, TypeVar

import numpy as np
from typing_extensions import Protocol

from cirq import linalg
from cirq.protocols.unitary import unitary


# This is a special indicator value used by the unitary method to determine
# whether or not the caller provided a 'default' argument. It must be of type
# np.ndarray to ensure the method has the correct type signature in that case.
# It is checked for using `is`, so it won't have a false positive if the user
# provides a different np.array([]) value.
RaiseTypeErrorIfNotProvided = np.array([])  # type: np.ndarray

TDefault = TypeVar('TDefault')


class SupportsApplyUnitaryToTensor(Protocol):
    """An object that can be efficiently left-multiplied into tensors."""

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, type(NotImplemented)]:
        """Left-multiplies a unitary effect onto a tensor with good performance.

        This method is given both the target tensor and workspace of the same
        shape and dtype. The method then either performs inline modifications of
        the target tensor and returns it, or writes its output into the
        workspace tensor and returns that. This signature makes it possible to
        write specialized simulation methods that run without performing large
        allocations, significantly increasing simulation performance.

        Args:
            target_tensor: The input tensor that needs to be left-multiplied by
                the unitary effect of the receiving object.
            available_buffer: Pre-allocated workspace with the same shape and
                dtype as the target tensor.
            axes: Which axes the unitary effect is being applied to (e.g. the
                qubits that the gate is operating on).

        Returns:
            If the receiving object is not able to apply its unitary effect,
            NotImplemented should be returned.

            Otherwise, the result should be either target_tensor or
            available_buffer; whichever now contains the output. It is permitted
            to allocate and return a new np.ndarray holding the output, but not
            recommended (because this implies lower performance and higher
            memory usage).
        """


def apply_unitary_to_tensor(val: Any,
                            target_tensor: np.ndarray,
                            available_buffer: np.ndarray,
                            axes: Sequence[int],
                            default: TDefault = RaiseTypeErrorIfNotProvided
                            ) -> Union[np.ndarray, TDefault]:
    """Left-multiplies a object's unitary effect onto a tensor.

    If `val` defines an _apply_unitary_to_tensor_ method, that method will be
    used to apply `val`'s unitary effect to the target tensor. Otherwise, if
    `val` defines a _unitary_ method, its unitary matrix will be retrieved and
    applied using a generic method. Otherwise the application fails, and either
    an exception is raised or the specified default value is returned.

    Args:
        val: The value with a unitary effect to apply to the target tensor.
        target_tensor: The input tensor that needs to be left-multiplied by
            the unitary effect of `val`. Note that this value may be mutated
            inline into the output.
        available_buffer: Pre-allocated workspace with the same shape and
            dtype as the target tensor. Note that the output may be written
            into this buffer.
        axes: Which axes the unitary effect is being applied to (e.g. the
            qubits that the gate is operating on).
        default: What should be returned if `val` doesn't have a unitary effect.
            If not specified, an exception is raised instead of returning
            a default value.

    Returns:
        The np.ndarray containing the result (typically either
        `target_tensor` or `available_buffer`), or else `default` if `val` had
         no unitary effect and a default value was specified.

    Raises:
        TypeError: `val` doesn't have a unitary effect and `default` wasn't
            specified.
    """

    # Check if the specialized method is present.
    getter = getattr(val, '_apply_unitary_to_tensor_', None)
    if getter is not None:
        result = getter(target_tensor, available_buffer, axes)
        if result is not NotImplemented:
            return result

    # Fallback to using the object's _unitary_ matrix.
    matrix = unitary(val, None)
    if matrix is not None:
        return linalg.targeted_left_multiply(
            matrix.astype(target_tensor.dtype).reshape((2,) * (2 * len(axes))),
            target_tensor,
            axes,
            out=available_buffer)

    # Don't know how to apply. Fallback to specified default behavior.
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError("object of type '{}' "
                    "has no _apply_unitary_to_tensor_ "
                    "or _unitary_ methods "
                    "(or they returned NotImplemented).".format(type(val)))
