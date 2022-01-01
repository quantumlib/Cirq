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

from typing import Sequence, Tuple, Optional

import numpy as np

from cirq import protocols
from cirq.ops import raw_types


class DimensionAdapter(raw_types.Gate):
    def __init__(self, gate: 'cirq.Gate', slices: Sequence[Tuple[int, slice]]):
        self._gate = gate
        self._slices = tuple(slices)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return tuple(i for i, _ in self._slices)

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        slices = tuple(s for _, s in self._slices)
        my_args = protocols.ApplyUnitaryArgs(
            target_tensor=args.target_tensor,
            available_buffer=args.available_buffer,
            slices=slices,
            axes=args.axes,
        )
        return protocols.apply_unitary(self._gate, args=my_args)
