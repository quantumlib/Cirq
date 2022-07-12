# Copyright 2019 The Cirq Developers
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


class SycamoreGate(cirq.FSimGate):
    """The Sycamore gate is a two-qubit gate equivalent to FSimGate(π/2, π/6).

    The unitary of this gate is

        [[1, 0, 0, 0],
         [0, 0, -1j, 0],
         [0, -1j, 0, 0],
         [0, 0, 0, exp(- 1j * π/6)]]

    This gate can be performed on the Google's Sycamore chip and
    is close to the gates that were used to demonstrate beyond
    classical resuts used in this paper:
    https://www.nature.com/articles/s41586-019-1666-5
    """

    def __init__(self):
        super().__init__(theta=np.pi / 2, phi=np.pi / 6)

    def __repr__(self) -> str:
        return 'cirq_google.SYC'

    def __str__(self) -> str:
        return 'SYC'

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return 'SYC', 'SYC'

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(self, [])


SYC = SycamoreGate()
document(
    SYC,
    """The Sycamore gate is a two-qubit gate equivalent to FSimGate(π/2, π/6).

           The unitary of this gate is

               [[1, 0, 0, 0],
                [0, 0, -1j, 0],
                [0, -1j, 0, 0],
                [0, 0, 0, exp(- 1j * π/6)]]

           This gate can be performed on the Google's Sycamore chip and
           is close to the gates that were used to demonstrate quantum
           supremacy used in this paper:
           https://www.nature.com/articles/s41586-019-1666-5
           """,
)
