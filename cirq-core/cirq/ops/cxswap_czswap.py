# Copyright 2024 The Cirq Developers
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

from typing import (
    Optional,
    Tuple,
    Dict,
    Any
)

import numpy as np

import cirq
from cirq import protocols
from cirq.ops import gate_features, raw_types

class CZSWAPGate(gate_features.InterchangeableQubitsGate, raw_types.Gate):
    r"""CZSWAP gate.

    This gate combines the CZ and SWAP gates.

    The unitary matrix of this gate is:

    $$
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 0 & -1
    \end{bmatrix}
    $$

    Up to a global phase, CZSWAP(0,1) is equivalent to applying the following:
    H(0)
    CX(0,1)
    CX(1,0)
    H(1)

    """
    def __init__(self) -> None:
        super(CZSWAPGate, self)

    def _num_qubits_(self) -> int:
        return 2

    def _unitary_(self) -> Optional[np.ndarray]:
 
        # fmt: off
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1],
            ]
        )
        # fmt: on
    
    def _has_unitary_(self):
        return True

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.H(a)
        yield cirq.CNOT(a, b)
        yield cirq.CNOT(b, a)
        yield cirq.H(b)

    def __repr__(self) -> str:
        return 'cirq.CZSWAPGate()'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self)


class CXSWAPGate(raw_types.Gate):
    r"""CXSWAP gate.

    This gate combines the CX and SWAP gates.

    The unitary matrix of this gate is:

    $$
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1 \\
        0 & 1 & 0 & 0
    \end{bmatrix}
    $$

    Up to a global phase, `CXSWAP(0,1)` is equivalent to applying the following:
    CX(0,1)
    CX(1,0)

    """
    def __init__(self) -> None:
        super(CXSWAPGate, self)

    def _num_qubits_(self) -> int:
        return 2

    def _unitary_(self) -> Optional[np.ndarray]:
 
        # fmt: off
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
            ]
        )
        # fmt: on

    def _has_unitary_(self):
        return True

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.CNOT(a, b)
        yield cirq.CNOT(b, a)

    def _circuit_diagram_info_(self, args) -> Tuple[str, ...]:
        return 'cirq.CXSWAPGate', 'cirq.CXSWAPGate'

    def __repr__(self) -> str:
        return 'cirq.CXSWAPGate()'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self)