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

"""Quantum gates native to ion trap systems."""
from typing import (
    Union, Tuple, Optional, List, Callable, cast, Iterable, Sequence,
)

import numpy as np

from cirq import value, protocols
from cirq.ops import gate_features, eigen_gate

class CxNOTGate(eigen_gate.EigenGate,
                gate_features.TwoQubitGate,
                gate_features.InterchangeableQubitsGate):
    """Implements the two-qubit Molmer-Sorensen entangling gate in ion traps.

    Effectively a performe a +pi/4 rotation around XX axis in the two-qubit bloch sphere.
    """

    def __init__(self, *,  # Forces keyword args.
                 exponent: Optional[Union[value.Symbol, float]] = None) -> None:

        """
        Initializes the gate.

        The exponent argument may be specified to raise the gate operation to a certain power.
        If no exponent argument is given, the default value of one is used.

        Args:
            exponent: Raise the gate to a certain power.
        """
        super().__init__(exponent=value.chosen_angle_to_half_turns(
            half_turns=exponent))

    def _eigen_components(self):
        return [
            (0.25, np.array([[0.5, 0, 0, -0.5], [0, 0.5, -0.5, 0], [0, -0.5, 0.5, 0], [-0.5, 0, 0, 0.5]])),
            (-0.25, np.array([[0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5]]))
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 8

    def _with_exponent(self,
                       exponent: Union[value.Symbol, float]) -> 'CxNOTGate':
        return CxNOTGate(exponent=exponent)

    @property
    def exponent(self) -> Union[value.Symbol, float]:
        return self._exponent

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('X', 'X'),
            exponent=self.exponent)

    def __str__(self) -> str:
        if self.exponent == 1:
            return 'CxNOT'
        return 'CxNOT**{!r}'.format(self.exponent)

    def __repr__(self) -> str:
        if self.exponent == 1:
            return 'ion.CxNOT'
        return '(ion.CxNOT**{!r})'.format(self.exponent)
