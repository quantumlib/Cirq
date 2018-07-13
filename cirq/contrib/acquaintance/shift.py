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

from itertools import chain

import cirq
from cirq.ops import gate_features, Gate, SWAP

class CircularShiftGate(cirq.Gate,
                        cirq.CompositeGate,
                        cirq.TextDiagrammable):
    """Swaps two sets of qubits.

    Args:
        shift: how many positions to circularly left shift the qubits.
        swap_gate: the gate to use when decomposing.
    """

    def __init__(self, 
                 shift: int,
                 swap_gate: Gate=SWAP) -> None:
        self.shift = shift
        self.swap_gate = swap_gate

    def __repr__(self):
        return 'CircularShiftGate'

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return ((self.shift == other.shift) and
                (self.swap_gate == other.swap_gate))

    def default_decompose(self, qubits):
        n = len(qubits)
        left_shift = self.shift % n
        right_shift = n - left_shift
        mins = chain(range(left_shift - 1, 0, -1),
                     range(right_shift))
        maxs = chain(range(left_shift, n),
                     range(n - 1, right_shift, -1))
        for i, j in zip(mins, maxs):
            for k in range(i, j, 2):
                yield self.swap_gate(*qubits[k:k+2])

    def text_diagram_info(self,
                          args: gate_features.TextDiagramInfoArgs):
        if args.known_qubit_count is None:
            return NotImplemented
        direction_symbols = (
            ('╲', '╱') if args.use_unicode_characters else
            ('\\', '/'))
        wire_symbols = tuple(
                direction_symbols[int(i >= self.shift)] +
                str(i) +
                direction_symbols[int(i < self.shift)]
                for i in range(args.known_qubit_count))
        return gate_features.TextDiagramInfo(
                wire_symbols=wire_symbols)

CircularShiftGates = type('CircularShiftGates', (dict,), {'__missing__':
    (lambda self, key: self.setdefault(key, CircularShiftGate(key)))})
SHIFT = CircularShiftGates()
