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

from typing import List, Sequence, Tuple, Union

import numpy as np

from cirq import value, protocols
from cirq.ops import raw_types, gate_features, common_gates, eigen_gate, op_tree
from cirq.ops.pauli import Pauli
from cirq.ops.clifford_gate import SingleQubitCliffordGate


pauli_eigen_map = {
    Pauli.X: (np.array([[0.5,  0.5], [0.5,   0.5]]),
              np.array([[0.5, -0.5], [-0.5,  0.5]])),
    Pauli.Y: (np.array([[0.5, -0.5j], [0.5j,  0.5]]),
              np.array([[0.5,  0.5j], [-0.5j, 0.5]])),
    Pauli.Z: (np.diag([1, 0]),
              np.diag([0, 1])),
}


@value.value_equality
class PauliInteractionGate(eigen_gate.EigenGate,
                           gate_features.InterchangeableQubitsGate,
                           gate_features.TwoQubitGate):
    CZ = None  # type: PauliInteractionGate
    CNOT = None  # type: PauliInteractionGate

    def __init__(self,
                 pauli0: Pauli, invert0: bool,
                 pauli1: Pauli, invert1: bool,
                 *,
                 exponent: Union[value.Symbol, float] = 1.0) -> None:
        """
        Args:
            pauli0: The interaction axis for the first qubit.
            invert0: Whether to condition on the +1 or -1 eigenvector of the
                first qubit's interaction axis.
            pauli1: The interaction axis for the second qubit.
            invert1: Whether to condition on the +1 or -1 eigenvector of the
                second qubit's interaction axis.
            exponent: Determines the amount of phasing to apply to the vector
                equal to the tensor product of the two conditions.
        """
        super().__init__(exponent=exponent)
        self.pauli0 = pauli0
        self.invert0 = invert0
        self.pauli1 = pauli1
        self.invert1 = invert1

    def _value_equality_values_(self):
        return (self.pauli0, self.invert0,
                self.pauli1, self.invert1,
                self._canonical_exponent)

    def qubit_index_to_equivalence_group_key(self, index: int) -> int:
        if self.pauli0 == self.pauli1 and self.invert0 == self.invert1:
            return 0
        return index

    def _with_exponent(self, exponent: Union[value.Symbol, float]
                       ) -> 'PauliInteractionGate':
        return PauliInteractionGate(self.pauli0, self.invert0,
                                    self.pauli1, self.invert1,
                                    exponent=exponent)

    def _eigen_shifts(self) -> List[float]:
        return [0.0, 1.0]

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        comp1 = np.kron(pauli_eigen_map[self.pauli0][not self.invert0],
                        pauli_eigen_map[self.pauli1][not self.invert1])
        comp0 = np.eye(4) - comp1
        return [(0, comp0), (1, comp1)]

    def _decompose_(self, qubits: Sequence[raw_types.QubitId]
                          ) -> op_tree.OP_TREE:
        q0, q1 = qubits
        right_gate0 = SingleQubitCliffordGate.from_single_map(
            z_to=(self.pauli0, self.invert0))
        right_gate1 = SingleQubitCliffordGate.from_single_map(
            z_to=(self.pauli1, self.invert1))

        left_gate0 = right_gate0**-1
        left_gate1 = right_gate1**-1
        yield left_gate0(q0)
        yield left_gate1(q1)
        yield common_gates.CZ(q0, q1)**self._exponent
        yield right_gate0(q0)
        yield right_gate1(q1)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        labels = {Pauli.X: 'X', Pauli.Y: 'Y', Pauli.Z: '@'}
        l0 = labels[self.pauli0]
        l1 = labels[self.pauli1]
        # Add brackets around letter if inverted
        l0, l1 = ('(-{})'.format(l) if inv else l
                  for l, inv in ((l0, self.invert0), (l1, self.invert1)))
        return protocols.CircuitDiagramInfo(
            wire_symbols=(l0, l1),
            exponent=self._diagram_exponent(args))

    def __repr__(self):
        base = 'cirq.PauliInteractionGate({!r}, {!s}, {!r}, {!s})'.format(
            self.pauli0, self.invert0, self.pauli1, self.invert1)
        if self._exponent == 1:
            return base

        return '({}**{!r})'.format(base, self._exponent)


PauliInteractionGate.CZ = PauliInteractionGate(Pauli.Z, False, Pauli.Z, False)
PauliInteractionGate.CNOT = PauliInteractionGate(
    Pauli.Z, False, Pauli.X, False)
