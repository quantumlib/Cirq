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
"""Native gates for IonQ hardware"""

from typing import Sequence, Union

import cmath
import math
import cirq
from cirq import protocols
from cirq._doc import document
import numpy as np


class GPIGate(cirq.Gate):
    """The GPI gate is a single qubit gate.
    The unitary of this gate is
        [[0, e^{-i\\phi}],
         [e^{-i\\phi}, 0]]
    """

    def __init__(self, *, phi):
        self.phi = phi

    def _unitary_(self) -> np.ndarray:
        top = cmath.exp(self.phi * 1j)
        bot = cmath.exp(-self.phi * 1j)
        return np.array([[0, top], [bot, 0]])

    def __repr__(self) -> str:
        return 'cirq_ionq.GPI'

    def __str__(self) -> str:
        return 'GPI'

    def _num_qubits_(self) -> int:
        return 1

    @property
    def phase(self) -> float:
        return self.phi

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('GPI',), exponent=self.phase
        )

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(self, [])


GPI = GPIGate(phi=0)
document(
    GPI,
    """The GPI gate is a single qubit gate.
    The unitary of this gate is
        [[0, e^{-i\\phi}],
         [e^{-i\\phi}, 0]]
    It is driven by Rabi laser.
    https://ionq.com/best-practices
    """,
)


class GPI2Gate(cirq.Gate):
    """The GPI2 gate is a single qubit gate.
    The unitary of this gate is
        \\frac{1}{/\\sqrt{2}}[[1, -i*e^{-i\\phi}],
         [-i*e^{-i\\phi}, 1]]
    """

    def __init__(self, *, phi):
        self.phi = phi

    def _unitary_(self) -> np.ndarray:
        top = -1j * cmath.exp(self.phase * -1j)
        bot = -1j * cmath.exp(self.phase * 1j)
        return np.array([[1, top], [bot, 1]]) / math.sqrt(2)

    @property
    def phase(self) -> float:
        return self.phi

    def __repr__(self) -> str:
        return 'cirq_ionq.GPI2'

    def __str__(self) -> str:
        return 'GPI2'

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('GPI2',), exponent=self.phase
        )

    def _num_qubits_(self) -> int:
        return 1

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(self, [])


GPI2 = GPI2Gate(phi=0)
document(
    GPI2,
    """The GPI2 gate is a single qubit gate.
    The unitary of this gate is
        \\frac{1}{/\\sqrt{2}}[[1, -i*e^{-i\\phi}],
         [-i*e^{-i\\phi}, 1]]
    It is driven by Rabi laser.
    https://ionq.com/best-practices
    """,
)


class MSGate(cirq.Gate):
    """The MS gate is a 2 qubit gate.
    The unitary of this gate is
         MS(\\phi_1 - \\phi_0) =
         MS(t) =
            \\frac{1}{\\sqrt{2}}
            [[\\cos(t), 0, 0, -i*\\sin(t)],
             [0, \\cos(t), -i*\\sin(t), 0],
             [0, -i*\\sin(t), \\cos(t), 0],
             [-i*\\sin(t), 0, 0, \\cos(t)]]
    """

    def __init__(self, *, phi1, phi2):
        self.phi1 = phi1
        self.phi2 = phi2

    def __repr__(self) -> str:
        return 'cirq_ionq.MS'

    def _unitary_(self) -> np.ndarray:
        tee = self.phi2 - self.phi1
        diag = math.cos(tee)
        adiag = -1j * math.sin(tee)
        return np.array(
            [[diag, 0, 0, adiag], [0, diag, adiag, 0], [0, adiag, diag, 0], [adiag, 0, 0, diag]]
        )

    @property
    def phases(self) -> Sequence[float]:
        return [self.phi1, self.phi2]

    def __str__(self) -> str:
        return 'MS'

    def _num_qubits_(self) -> int:
        return 2

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('MS',), exponent=self.phases
        )

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(self, [])


MS = MSGate(phi1=0, phi2=0)
document(
    MS,
    """The MS gate is a 2 qubit gate.
    The unitary of this gate is
         MS(\\phi_1 - \\phi_0) =
         MS(t) =
            \\frac{1}{\\sqrt{2}}
            [[\\cos(t), 0, 0, -i*\\sin(t)],
             [0, \\cos(t), -i*\\sin(t), 0],
             [0, -i*\\sin(t), \\cos(t), 0],
             [-i*\\sin(t), 0, 0, \\cos(t)]]

    https://ionq.com/best-practices
    """,
)
