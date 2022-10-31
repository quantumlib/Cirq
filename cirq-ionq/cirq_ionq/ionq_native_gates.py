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

from typing import Any, Dict, Sequence, Union

import cmath
import math
import cirq
from cirq import protocols
from cirq._doc import document
import numpy as np


@cirq.value.value_equality
class GPIGate(cirq.Gate):
    r"""The GPI gate is a single qubit gate representing a pi pulse.

    The unitary matrix of this gate is:
    $$
    \begin{bmatrix}
      0 & e^{-i 2\pi\phi} \\
      e^{-i2\pi\phi} & 0
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """

    def __init__(self, *, phi):
        self.phi = phi

    def _unitary_(self) -> np.ndarray:
        top = cmath.exp(-self.phi * 2 * math.pi * 1j)
        bot = cmath.exp(self.phi * 2 * math.pi * 1j)
        return np.array([[0, top], [bot, 0]])

    def __str__(self) -> str:
        return 'GPI'

    def _num_qubits_(self) -> int:
        return 1

    @property
    def phase(self) -> float:
        return self.phi

    def __repr__(self) -> str:
        return f'cirq_ionq.GPIGate(phi={self.phi!r})'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['phi'])

    def _value_equality_values_(self) -> Any:
        return self.phi

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(wire_symbols=(f'GPI({self.phase!r})',))

    def __pow__(self, power):
        if power == 1:
            return self

        if power == -1:
            return self

        return NotImplemented


GPI = GPIGate(phi=0)
document(
    GPI,
    r"""An instance of the single qubit GPI gate with no phase.

    The unitary matrix of this gate is:
    $$
    \begin{bmatrix}
      0 & 1 \\
      1 & 0
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """,
)


@cirq.value.value_equality
class GPI2Gate(cirq.Gate):
    r"""The GPI2 gate is a single qubit gate representing a pi/2 pulse.

    The unitary matrix of this gate is
    $$
    \frac{1}{\sqrt{2}}
    \begin{bmatrix}
        1 & -i e^{-i \phi} \\
        -i e^{-i \phi} & 1
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """

    def __init__(self, *, phi):
        self.phi = phi

    def _unitary_(self) -> np.ndarray:
        top = -1j * cmath.exp(self.phase * 2 * math.pi * -1j)
        bot = -1j * cmath.exp(self.phase * 2 * math.pi * 1j)
        return np.array([[1, top], [bot, 1]]) / math.sqrt(2)

    @property
    def phase(self) -> float:
        return self.phi

    def __str__(self) -> str:
        return 'GPI2'

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(wire_symbols=(f'GPI2({self.phase!r})',))

    def _num_qubits_(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f'cirq_ionq.GPI2Gate(phi={self.phi!r})'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['phi'])

    def _value_equality_values_(self) -> Any:
        return self.phi

    def __pow__(self, power):
        if power == 1:
            return self

        if power == -1:
            return GPI2Gate(phi=self.phi + 0.5)

        return NotImplemented


GPI2 = GPI2Gate(phi=0)
document(
    GPI2,
    r"""An instance of the single qubit GPI2 gate with no phase.

    The unitary matrix of this gate is
    $$
    \frac{1}{\sqrt{2}}
    \begin{bmatrix}
        1 & -i \\
        -i & 1
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """,
)


@cirq.value.value_equality
class MSGate(cirq.Gate):
    r"""The Mølmer–Sørensen (MS) gate is a two qubit gate native to trapped ions.

    The unitary matrix of this gate for parameters $\phi_0$, $\phi_1$ and $\theta is

    $$
    \begin{bmatrix}
        cos{\theta/2} & 0 & 0 & -i*e^{-i*2*\pi(\phi_0+\phi_1)}*sin{\theta/2} \\
        0 & cos{\theta/2} & -i*e^{-i*2*\pi(\phi_0-\phi_1)}*sin{\theta/2} & 0 \\
        0 & -i*e^{i*2*\pi(\phi_0-\phi_1)}*sin(\theta/2) & cos{\theta/2} & 0 \\
        -i*e^{i*2*\pi(\phi_0+\phi_1)}*sin{\theta/2} & 0 & 0 & cos{\theta/2}
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """

    def __init__(self, *, phi0, phi1, theta=0.25):
        self.phi0 = phi0
        self.phi1 = phi1
        self.theta = theta

    def _unitary_(self) -> np.ndarray:
        theta = self.theta
        phi0 = self.phi0
        phi1 = self.phi1
        diag = np.cos(math.pi * theta)
        sin = np.sin(math.pi * theta)

        return np.array(
            [
                [diag, 0, 0, sin * -1j * cmath.exp(-1j * 2 * math.pi * (phi0 + phi1))],
                [0, diag, sin * -1j * cmath.exp(-1j * 2 * math.pi * (phi0 - phi1)), 0],
                [0, sin * -1j * cmath.exp(1j * 2 * math.pi * (phi0 - phi1)), diag, 0],
                [sin * -1j * cmath.exp(1j * 2 * math.pi * (phi0 + phi1)), 0, 0, diag],
            ]
        )

    @property
    def phases(self) -> Sequence[float]:
        return [self.phi0, self.phi1]

    def __str__(self) -> str:
        return 'MS'

    def _num_qubits_(self) -> int:
        return 2

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(
            wire_symbols=(f'MS({self.phi0!r})', f'MS({self.phi1!r})')
        )

    def __repr__(self) -> str:
        return f'cirq_ionq.MSGate(phi0={self.phi0!r}, phi1={self.phi1!r})'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['phi0', 'phi1', 'theta'])

    def _value_equality_values_(self) -> Any:
        return (self.phi0, self.phi1)

    def __pow__(self, power):
        if power == 1:
            return self

        if power == -1:
            return MSGate(phi0=self.phi0 + 0.5, phi1=self.phi1)

        return NotImplemented


MS = MSGate(phi0=0, phi1=0)
document(
    MS,
    r"""An instance of the two qubit Mølmer–Sørensen (MS) gate with no phases.

    The unitary matrix of this gate for parameters $\phi_0$ and $\phi_1$ is

    $$
    \frac{1}{\sqrt{2}}
    \begin{bmatrix}
        1 & 0 &  0 & -i \\
        0 & 1 & -i & 0 \\
        0 & -i & 1 & 0 \\
        -i & 0 & 0 & 1 \\
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """,
)
