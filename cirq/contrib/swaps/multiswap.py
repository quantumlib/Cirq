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

from typing import Sequence, Union

import cirq
from cirq.ops import gate_features, Gate, SWAP

class MultiswapGate(cirq.CompositeGate,
                    cirq.TextDiagrammable):
    """Swaps two sets of qubits.

    Args:
        multiplicities: a sequence of two ints indicating the sizes of the two
            sets of qubits to swap.
        swap_gate: the gate to use when decomposing the multiswap gate.
    """

    def __init__(self, 
                 multiplicities: Union[Sequence[int], int],
                 swap_gate: Gate=SWAP) -> None:
        if isinstance(multiplicities, int):
            self.multiplicities = (multiplicities,) * 2
        elif len(multiplicities) != 2:
            raise ValueError('Multiplicities must have length 2.')
        elif min(multiplicities) < 1:
            raise ValueError('Multiplicities must be at least 1.')
        else:
            self.multiplicities = tuple(multiplicities)
        self.swap_gate = swap_gate

    def __repr__(self):
        return 'multiSWAP'

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return ((self.multiplicities == other.multiplicities) and
                (self.swap_gate == other.swap_gate))

    def default_decompose(self, qubits):
        n = sum(self.multiplicities)
        mins = (list(range(self.multiplicities[0] - 1, 0, -1)) + 
                list(range(self.multiplicities[1])))
        maxs = (list(range(self.multiplicities[0], n)) +
                list(range(n - 1, self.multiplicities[1], -1)))
        for i, j in zip(mins, maxs):
            for k in range(i, j, 2):
                yield self.swap_gate(*qubits[k:k+2])


    def text_diagram_info(self,
                          args: gate_features.TextDiagramInfoArgs):
        left_symbol, right_symbol = (
            ('╲╱', '╱╲') if args.use_unicode_characters else
            ('\\/', '/\\'))
        wire_symbols = ((left_symbol,) * self.multiplicities[0] +
                        (right_symbol,) * self.multiplicities[1])
        return gate_features.TextDiagramInfo(
                wire_symbols=wire_symbols)
