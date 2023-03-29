# Copyright 2023 The Cirq Developers
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

from typing import Any, Dict, Iterable, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from cirq import value
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality()
class PostSelectionGate(raw_types.Gate):
    r"""An n-qudit gate simulating post-selection on a given control state or set of states.

    This gate is only for simulation, and cannot be run on a real device as it violates the laws of
    quantum mechanics.
    """

    def __init__(self, qid_shape: Sequence[int], controls: Iterable[Sequence[int]]):
        r"""Creates an n-qudit gate simulating post-selection.

        Args:
            qid_shape: The shape of qubits this gate applies to.
            controls: The post-selection criteria. Only the dimensions of the waveform that match
                this criteria will be allowed through. The waveform will be renormalized after
                selection.
        """
        self._controls = tuple(tuple(c) for c in controls)
        self._qid_shape = tuple(qid_shape)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def _has_unitary_(self) -> bool:
        return True

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> np.ndarray:
        args.available_buffer[...] = 0
        for product in self._controls:
            bits = value.big_endian_digits_to_int(product, base=self._qid_shape)
            subspace_index = args.subspace_index(big_endian_bits_int=bits)
            args.available_buffer[subspace_index] += args.target_tensor[subspace_index]
        norm = np.linalg.norm(args.available_buffer)
        if norm == 0:
            raise ValueError('Waveform does not contain any post-selected values.')
        args.available_buffer /= norm
        return args.available_buffer

    def _value_equality_values_(self) -> Any:
        return self._controls, self._qid_shape

    def _json_dict_(self) -> Dict[str, Any]:
        return {'qid_shape': self._qid_shape, 'controls': self._controls}

    @classmethod
    def _from_json_dict_(cls, qid_shape, controls, **kwargs):
        return cls(controls=controls, qid_shape=qid_shape)

    def __repr__(self) -> str:
        return (
            f'cirq.PostSelectionGate(qid_shape={self._qid_shape}, controls={repr(self._controls)})'
        )
