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

import enum
import itertools
from typing import Dict, Optional, Sequence, Tuple, Union

from cirq import ops
from cirq.contrib.acquaintance.gates import (
    acquaint, operations_to_part_lens)
from cirq.contrib.acquaintance.permutation import (
    PermutationGate, SwapPermutationGate)
from cirq.contrib.acquaintance.shift import (
    CircularShiftGate)
from cirq.contrib.acquaintance.shift_swap_network import (
    ShiftSwapNetworkGate)

@enum.unique
class GraphType(enum.Enum):
    BIPARTITE = 1
    COMPLETE = 2
    NONE = 3

    def __repr__(self):
        return ('cirq.contrib.acquaintance.double_bipartite.'
                'GraphType.' +
                self.name)


class DoubleBipartiteSwapNetworkGate(PermutationGate):
    """Acquaints pairs of qubits from one set with pairs of qubits from another set.

    Works by applying a complete swap network to each set, and replacing each acquaintance layer with another swap network.

    Args:
        n_left_qubits: Number of qubits in left part.
        n_right_qubits: Number of qubits in right part. Defaults to
            n_left_qubits.
        swap_gate: The gate used to swap logical indices.
        shifted: Whether or not the overall effect is to shift the parts. If
            None (the default), determined by minimal 
        balanced: If true, only insert acquaintance layers before every other
            permutation layer.
        acquaintance_graph: Indicates the type of swap network to replace the
            intermediate acquaintance layers with. Defaults to BIPARTITE.
    """

    def __init__(self,
                 n_left_qubits: int,
                 n_right_qubits: Optional[int] = None, *,
                 swap_gate: ops.Gate = ops.SWAP,
                 shifted: Optional[bool] = None,
                 balanced: bool = False,
                 acquaintance_graph:
                     Union[str, GraphType] =
                        'BIPARTITE'
                 ) -> None:

        assert n_left_qubits > 0
        self.n_left_qubits = n_left_qubits
        if n_right_qubits is None:
            n_right_qubits = n_left_qubits
        assert n_right_qubits > 0
        self.n_right_qubits = n_right_qubits

        assert (not balanced) or (not (
            (n_left_qubits % 2) or (n_right_qubits % 2)))

        self.swap_gate = swap_gate
        self.shifted = shifted
        self.balanced = balanced
        self.acquaintance_graph = (
                acquaintance_graph if isinstance(acquaintance_graph, GraphType)
                else GraphType[acquaintance_graph])

    def num_qubits(self):
        return self.n_left_qubits + self.n_right_qubits

    def _decompose_(self, qubits: Sequence[ops.QubitId]) -> ops.OP_TREE:
        qubit_sets = {
            'left': qubits[:self.n_left_qubits],
            'right': qubits[self.n_left_qubits:]}
        qubit_set_sizes = {side: len(qubit_set)
                for side, qubit_set in qubit_sets.items()}

        get_acquaintances = lambda i, side: tuple(
                acquaint(*qubit_sets[side][i: i + 2])
                for i in range(i % 2, qubit_set_sizes[side] - 1, 2))
        get_swaps = lambda i, side: tuple(
                SwapPermutationGate(self.swap_gate)(*qubit_sets[side][i: i + 2])
                for i in range(i % 2, qubit_set_sizes[side] - 1, 2))

        sides = ('left', 'right')
        for i in range(qubit_set_sizes['left']):
            for j in range(qubit_set_sizes['right']):
                if (not self.balanced) or ((i + j + 1) % 2):
                    acquaintances = {
                            'left':  get_acquaintances(i, 'left'),
                            'right':  get_acquaintances(j, 'right')}
                    if self.acquaintance_graph == GraphType.NONE:
                        yield ops.Moment(sum(
                            (acquaintances[side] for side in sides), ()))
                    else:
                        part_lens = {side: operations_to_part_lens(
                            qubit_sets[side], acquaintances[side])
                            for side in sides}
                        if self.acquaintance_graph == GraphType.BIPARTITE:
                            swap_network = ShiftSwapNetworkGate(
                                    part_lens[sides[0]], part_lens[sides[1]],
                                    self.swap_gate)
                            yield ops.Moment([swap_network(*qubits)])

                            sides = sides[::-1]
                            qubit_sets[sides[0]] = (
                                    qubits[:qubit_set_sizes[sides[0]]])
                            qubit_sets[sides[1]] = (
                                    qubits[qubit_set_sizes[sides[0]]:])
                        else:
                            raise NotImplementedError(
                                    'No decomposition implemented for ' +
                                    str(self.acquaintance_graph))

                swap_moment = (
                        ops.Moment(get_swaps(i, 'left') +
                                   get_swaps(j, 'right')) if
                        (j == qubit_set_sizes['right'] - 1) else
                        ops.Moment(get_swaps(j, 'right')))
                yield swap_moment

        if ((self.shifted is not None) and
            (self.shifted != self.naturally_shifted)):
            shift_gate = CircularShiftGate(
                    len(qubits),
                    qubit_set_sizes[sides[0]],
                    swap_gate=self.swap_gate)
            yield shift_gate(*qubits)

    @property
    def naturally_shifted(self):
        if self.acquaintance_graph == GraphType.NONE:
            return False
        elif self.acquaintance_graph == GraphType.BIPARTITE:
            n_shift_layers = self.n_left_qubits * self.n_right_qubits
            if self.balanced:
                n_shift_layers /= 2
            return bool(n_shift_layers % 2)

        raise NotImplementedError(
                str(self.acquaintance_graph) + 'not implemented')


    def permutation(self) -> Dict[int, int]:
        left_indices = range(self.n_left_qubits)[::-1]
        right_indices = range(self.n_left_qubits, self.num_qubits())
        if bool(self.n_left_qubits % 2):
            right_indices = right_indices[::-1]

        if self.shifted is None:
            shifted = self.naturally_shifted
        else:
            shifted = self.shifted

        index_sets = (reversed if shifted else lambda _: _)([
                left_indices, right_indices])

        return dict(zip(itertools.chain(*index_sets),
                        range(self.num_qubits())))

    def __repr__(self):
        args = (repr(self.n_left_qubits),)
        if self.n_left_qubits != self.n_right_qubits:
            args += (repr(self.n_right_qubits),)
        if self.swap_gate != ops.SWAP:
            args += ('swap_gate=' + repr(self.swap_gate),)
        if self.shifted is not None:
            args += ('shifted=' + repr(self.shifted),)
        if self.balanced:
            args += ('balanced=' + repr(self.balanced),)
        if self.acquaintance_graph != GraphType.BIPARTITE:
            args += ("acquaintance_graph='{}'".format(
                self.acquaintance_graph.name),)
        return ('cirq.contrib.acquaintance.double_bipartite.'
                'DoubleBipartiteSwapNetworkGate' +
                '({})'.format(', '.join(args)))

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.n_left_qubits == other.n_left_qubits and
                self.n_left_qubits == other.n_left_qubits and
                self.swap_gate == other.swap_gate and
                self.balanced == other.balanced and
                self.shifted == other.shifted and
                self.acquaintance_graph == other.acquaintance_graph)

    def acquaintance_size(self) -> int:
        return 2 if self.acquaintance_graph == GraphType.NONE else 4
