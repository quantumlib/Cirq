# Copyright 2020 The Cirq Developers
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

"""Creates the gate instance for a two qubit diagonal gate.

The gate is used to create a 4x4 matrix with the diagonal elements
passed as a list.
"""

from typing import AbstractSet, Any, Dict, Iterator, Tuple, Optional, Sequence, TYPE_CHECKING
import numpy as np
import sympy

from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import raw_types, common_gates

if TYPE_CHECKING:
    import cirq


@value.value_equality()
class TwoQubitDiagonalGate(raw_types.Gate):
    r"""A two qubit gate whose unitary is a diagonal $4 \times 4$ matrix.

    This gate's off-diagonal elements are zero and its on-diagonal
    elements are all phases.

    For example, `cirq.TwoQubitDiagonalGate([0, 1, -1, 0])` has the
    unitary matrix

    $$
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & e^i & 0 & 0 \\
        0 & 0 & e^{-i} & 0 \\
        0 & 0 & 0 & 1
    \end{bmatrix}
    $$
    """

    def __init__(self, diag_angles_radians: Sequence[value.TParamVal]) -> None:
        r"""A two qubit gate with only diagonal elements.

        This gate's off-diagonal elements are zero and its on-diagonal
        elements are all phases.

        Args:
            diag_angles_radians: The list of angles on the diagonal in radians.
                If these values are $(x_0, x_1, \ldots , x_3)$ then the unitary
                has diagonal values $(e^{i x_0}, e^{i x_1}, \ldots, e^{i x_3})$.
        """
        self._diag_angles_radians: Tuple[value.TParamVal, ...] = tuple(diag_angles_radians)

    @property
    def diag_angles_radians(self) -> Tuple[value.TParamVal, ...]:
        return self._diag_angles_radians

    def _num_qubits_(self) -> int:
        return 2

    def _is_parameterized_(self) -> bool:
        return any(protocols.is_parameterized(angle) for angle in self._diag_angles_radians)

    def _parameter_names_(self) -> AbstractSet[str]:
        return {
            name for angle in self._diag_angles_radians for name in protocols.parameter_names(angle)
        }

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'TwoQubitDiagonalGate':
        return TwoQubitDiagonalGate(
            protocols.resolve_parameters(self._diag_angles_radians, resolver, recursive)
        )

    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_()

    def _unitary_(self) -> Optional[np.ndarray]:
        if self._is_parameterized_():
            return None
        return np.diag([np.exp(1j * angle) for angle in self._diag_angles_radians])

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> Iterator['cirq.OP_TREE']:
        x0, x1, x2, x3 = self._diag_angles_radians
        q0, q1 = qubits
        yield common_gates.ZPowGate(exponent=x2 / np.pi).on(q0)
        yield common_gates.ZPowGate(exponent=x1 / np.pi).on(q1)
        yield common_gates.CZPowGate(exponent=(x3 - (x1 + x2)) / np.pi).on(q0, q1)
        yield common_gates.XPowGate().on_each(q0, q1)
        yield common_gates.CZPowGate(exponent=x0 / np.pi).on(q0, q1)
        yield common_gates.XPowGate().on_each(q0, q1)

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        if self._is_parameterized_():
            return NotImplemented
        for index, angle in enumerate(self._diag_angles_radians):
            subspace_index = args.subspace_index(big_endian_bits_int=index)
            args.target_tensor[subspace_index] *= np.exp(1j * angle)
        return args.target_tensor

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        rounded_angles = np.array(self._diag_angles_radians)
        if args.precision is not None:
            rounded_angles = rounded_angles.round(args.precision)
        diag_str = f"diag({', '.join(proper_repr(angle) for angle in rounded_angles)})"
        return protocols.CircuitDiagramInfo((diag_str, '#2'))

    def __pow__(self, exponent: Any) -> 'TwoQubitDiagonalGate':
        if not isinstance(exponent, (int, float, sympy.Basic)):
            return NotImplemented
        angles = []
        for angle in self._diag_angles_radians:
            mulAngle = protocols.mul(angle, exponent, NotImplemented)
            if mulAngle == NotImplemented:
                return NotImplemented
            angles.append(mulAngle)
        return TwoQubitDiagonalGate(angles)

    def _value_equality_values_(self) -> Any:
        return tuple(self._diag_angles_radians)

    def __repr__(self) -> str:
        angles = ','.join(proper_repr(angle) for angle in self._diag_angles_radians)
        return f'cirq.TwoQubitDiagonalGate([{angles}])'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, attribute_names=["diag_angles_radians"])
