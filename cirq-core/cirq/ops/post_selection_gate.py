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
    r"""A gate simulating post-selection from a given subspace or set of subspaces.

    This gate is only for simulation, and cannot be run on a real device as it violates the laws of
    quantum mechanics.
    """

    def __init__(self, qid_shape: Sequence[int], subspaces: Iterable[Sequence[int]]):
        r"""Creates a gate simulating post-selection.

        Args:
            qid_shape: The shape of qubits this gate applies to.
            subspaces: The post-selection subspace, provided as a set of index tuples. For example,
                |01><01| + |22><22| on a set of 3D qudits would be passed in as [(0, 1), (2, 2)].
                The waveform will be projected onto the subspace and renormalized. If the
                projection onto the subspace is zero, a ValueError will be raised.
        """
        self._subspaces = tuple(sorted(set(tuple(s) for s in subspaces)))
        self._qid_shape = tuple(qid_shape)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def _has_unitary_(self) -> bool:
        return True

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> np.ndarray:
        args.available_buffer[...] = 0
        for subspace in self._subspaces:
            args.available_buffer[subspace] += args.target_tensor[subspace]
        norm = np.linalg.norm(args.available_buffer)
        if np.isclose(norm, 0):
            raise ValueError('Waveform does not contain any post-selected values.')
        args.available_buffer /= norm
        return args.available_buffer

    def _value_equality_values_(self) -> Any:
        return self._subspaces, self._qid_shape

    def _json_dict_(self) -> Dict[str, Any]:
        return {'qid_shape': self._qid_shape, 'subspaces': self._subspaces}

    @classmethod
    def _from_json_dict_(cls, qid_shape, subspaces, **kwargs):
        return cls(subspaces=subspaces, qid_shape=qid_shape)

    def __repr__(self) -> str:
        subspaces = repr(self._subspaces)
        return f'cirq.PostSelectionGate(qid_shape={self._qid_shape}, subspaces={subspaces})'
