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

from typing import Any, TypeVar, Union, Optional, Sequence, Tuple

import numpy as np
from typing_extensions import Protocol

from cirq import extension, linalg

from cirq.protocols.unitary import unitary, SupportsUnitary


class SupportsApplyUnitaryToTensor(Protocol):
    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  out: Optional[np.ndarray] = None,
                                  ) -> Union[Tuple[np.ndarray, np.ndarray],
                                             type(NotImplemented)]:
        """
        Args:
            target_tensor:
            available_buffer:
            axes:
            out:

        Returns:
        """


def apply_unitary_to_tensor(val: Union[SupportsApplyUnitaryToTensor,
                                       SupportsUnitary],
                            target_tensor: np.ndarray,
                            available_buffer: np.ndarray,
                            axes: Sequence[int],
                            default: Any = None
                            ) -> np.ndarray:
    getter = getattr(val, '_apply_unitary_to_tensor_', None)
    if getter is not None:
        result = getter(target_tensor, available_buffer, axes)
        if result is not NotImplemented:
            return result

    matrix = unitary(val, None)
    matrix = matrix.astype(target_tensor.dtype).reshape((2,) * (2 * len(axes)))
    if matrix is not None:
        return linalg.targeted_left_multiply(matrix,
                                             target_tensor,
                                             axes,
                                             out=available_buffer)

    if default is not None:
        return default

    raise TypeError("object of type '{}' "
                    "has no _apply_unitary_to_tensor_ "
                    "or _unitary_ methods "
                    "(or they returned NotImplemented).".format(type(val)))
