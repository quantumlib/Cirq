# Copyright 2025 The Cirq Developers
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
"""An instance of FSimGate that works naturally on Google's Sycamore chip"""

import numpy as np

import cirq
from cirq._doc import document


class WillowGate(cirq.FSimGate):
    """The Willow gate is a two-qubit gate equivalent to FSimGate(π/2, π/9).

    The unitary of this gate is simulated as:

        [[1, 0, 0, 0],
         [0, 0, -1j, 0],
         [0, -1j, 0, 0],
         [0, 0, 0, exp(- 1j * π/9)]]

    This gate can be performed on the Google's Willow chip.  Note that
    this gate will be transformed to a "ISWAP-like" gate on hardware
    and that the C-phase angle (phi) may change from processor to
    processor.  The specified value is provided only for simulation
    purposes.
    """

    def __init__(self):
        super().__init__(theta=np.pi / 2, phi=np.pi / 9)

    def __repr__(self) -> str:
        return 'cirq_google.WILLOW'

    def __str__(self) -> str:
        return 'WILLOW'

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return 'WILLOW', 'WILLOW'

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(self, [])


WILLOW = WillowGate()
document(
    WILLOW,
    """The Willow gate is a two-qubit gate equivalent to FSimGate(π/2, π/9).

    The unitary of this gate is simulated as:

        [[1, 0, 0, 0],
         [0, 0, -1j, 0],
         [0, -1j, 0, 0],
         [0, 0, 0, exp(- 1j * π/9)]]

    This gate can be performed on the Google's Willow chip.  Note that
    this gate will be transformed to a "ISWAP-like" gate on hardware
    and that the C-phase value (phi) may change from processor to
    processor.  The specified value is provided only for simulation
    purposes.""",
)
