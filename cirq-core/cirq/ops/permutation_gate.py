# Copyright 2020 The Cirq Developers
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

from typing import Any, Dict, Sequence, Tuple, TYPE_CHECKING

from cirq import protocols, value
from cirq.ops import raw_types, swap_gates

if TYPE_CHECKING:
    import cirq


@value.value_equality
class QubitPermutationGate(raw_types.Gate):
    r"""A qubit permutation gate specified by a permutation list.

    For a permutation list $[p_0, p_1,\dots,p_{n-1}]$ this gate has the unitary

    $$
    \sum_{x_0,x_1,\dots,x_{n-1} \in \{0, 1\}} |x_{p_0}, x_{p_1}, \dots, x_{p_{n-1}}\rangle
                                              \langle x_0, x_1, \dots, x_{n-1}|
    $$
    """

    def __init__(self, permutation: Sequence[int]):
        """Create a `cirq.QubitPermutationGate`.

        Args:
            permutation: A shuffled sequence of integers from 0 to
                len(permutation) - 1. The entry at offset `i` is the result
                of permuting `i`.

        Raises:
            ValueError: If the supplied permutation is not valid (empty, repeated indices, indices
                out of range).
        """
        if not permutation:
            raise ValueError(f"Invalid permutation (empty): {permutation}")

        if len(set(permutation)) < len(permutation):
            raise ValueError(f"Invalid permutation {permutation} Each index must appear only once.")

        invalid_indices = [x for x in permutation if not 0 <= x < len(permutation)]
        if len(invalid_indices) > 0:
            raise ValueError(
                f"All indices have to satisfy 0 <= i < {len(permutation)}.\n"
                f"Invalid indices: {invalid_indices}"
            )

        self._permutation = tuple(permutation)

    @property
    def permutation(self) -> Tuple[int, ...]:
        return self._permutation

    def _value_equality_values_(self):
        return self.permutation

    def num_qubits(self):
        return len(self.permutation)

    def _has_unitary_(self):
        return True

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        permutation = [p for p in self.permutation]

        for i in range(len(permutation)):

            if permutation[i] == -1:
                continue
            cycle = [i]
            while permutation[cycle[-1]] != i:
                cycle.append(permutation[cycle[-1]])

            for j in cycle:
                permutation[j] = -1

            for idx in cycle[1:]:
                yield swap_gates.SWAP(qubits[cycle[0]], qubits[idx])

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs'):
        # Compute the permutation index list.
        permuted_axes = list(range(len(args.target_tensor.shape)))
        for i in range(len(args.axes)):
            j = self.permutation[i]
            ai = args.axes[i]
            aj = args.axes[j]
            assert args.target_tensor.shape[ai] == args.target_tensor.shape[aj]
            permuted_axes[aj] = ai

        # Delegate to numpy to do the permuted copy.
        args.available_buffer[...] = args.target_tensor.transpose(permuted_axes)
        return args.available_buffer

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Tuple[str, ...]:
        return tuple(f'[{i}>{self.permutation[i]}]' for i in range(len(self.permutation)))

    def __repr__(self) -> str:
        return f'cirq.QubitPermutationGate(permutation={self.permutation!r})'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, attribute_names=['permutation'])

    @classmethod
    def _from_json_dict_(cls, permutation: Sequence[int], **kwargs):
        return cls(permutation)
