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

"""Operations native to iontrap systems."""
from typing import (
    Union, Tuple, Optional, List, Callable, cast, Iterable, Sequence,
)

import numpy as np

from cirq import value, linalg, protocols
from cirq.ops import gate_features, eigen_gate, raw_types, gate_operation
from cirq.type_workarounds import NotImplementedType



class MSGate(eigen_gate.EigenGate,
             gate_features.TwoQubitGate,
             gate_features.InterchangeableQubitsGate):
    """
    Initializes the gate.

    The Mølmer–Sørensen gate, native two-qubit operation in ion traps.
    Effectively equivalents to rotation around the XX axis in the two-qubit bloch sphere.

    The gate implements the following unitary:
    [cos(t) 0 0 -isin(t)]
    [0 cos(t) -isin(t) 0]
    [0 -isin(t) cos(t) 0]
    [-isin(t) 0 0 cos(t)]
    """

    def __init__(self, *,  # Forces keyword args.
                 rads: Optional[float] = None,
                 exponent: Optional[Union[value.Symbol, float]] = None) -> None:
        """

        At most one of rads or exponent may be specified.
        If more are specified, the result is considered ambiguous and an
        error is thrown. If no argument is given, the default value of
        rads = pi/4 is used.

        Args:
            rads: The angle of rotation, in radians.
            exponent: The power that the gate operation will be raised to.
        """

        if rads is not None:
            power = 4 * rads / np.pi
        else:
            power = value.chosen_angle_to_half_turns(
                    half_turns=exponent)

        super().__init__(exponent=power)

    @property
    def exponent(self) -> Union[value.Symbol, float]:
        return self._exponent

    def _eigen_components(self):
        return [
            (0.25, np.array([[0.5, 0, 0, -0.5],
                            [0, 0.5, -0.5, 0],
                            [0, -0.5, 0.5, 0],
                            [-0.5, 0, 0, 0.5]])),
            (-0.25, np.array([[0.5, 0, 0, 0.5],
                            [0, 0.5, 0.5, 0],
                            [0, 0.5, 0.5, 0],
                            [0.5, 0, 0, 0.5]]))
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 8

    def _with_exponent(self,
                       exponent: Union[value.Symbol, float]) -> 'MSGate':
        return MSGate(exponent=exponent)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('MS', 'MS'),
            exponent=self.exponent)

    def __str__(self) -> str:
        if self.exponent == 1:
            return 'MS'
        return 'MS**{!r}'.format(self.exponent)

    def __repr__(self) -> str:
        if self.exponent == 1:
            return 'cirq.MS'
        return '(cirq.MS**{!r})'.format(self.exponent)


# Rotate around the XX axis in the two-qubit bloch sphere by pi/4
MS = MSGate()
