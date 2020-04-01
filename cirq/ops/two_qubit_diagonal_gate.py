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
"""Common quantum gates that target two qubits."""

from typing import Any, List, TYPE_CHECKING, cast, Collection, Sequence, Tuple, Union, Optional
# from typing import Any, cast, Collection, Optional, Sequence, Tuple, Union
import numpy as np
import sympy

from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import gate_features

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import cirq


@value.value_equality()
class TwoQubitDiagonalGate(gate_features.TwoQubitGate):
    """A gate given by a diagonal 4x4 matrix."""

    def __init__(self, diag_angles_radians: List[value.TParamVal]) -> None:
        """A two qubit gate with only diagonal elements.

        This gate's off-diagonal elements are zero and it's on diagonal
        elements are all phases.

        Args:
            diag_angles_radians: The list of angles on the diagonal in radians.
                If these values are $(x_0, x_1, \ldots , x_3)$ then the unitary
                has diagonal values $(e^{i x_0}, e^{i x_1}, \ldots, e^{i x_3})$.
        """
        self._diag_angles_radians: List[value.TParamVal] = np.array(diag_angles_radians, dtype=np.float64)

    def _is_parameterized_(self):
        return any(
            isinstance(angle, sympy.Basic)
            for angle in self._diag_angles_radians)

    def _resolve_parameters_(self, param_resolver: 'cirq.ParamResolver'
                            ) -> 'TwoQubitDiagonalGate':
        return TwoQubitDiagonalGate([
            protocols.resolve_parameters(tuple(self._diag_angles_radians), param_resolver)
            for angle in self._diag_angles_radians
        ])

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        rounded_angles = np.array(self._diag_angles_radians)
        if args.precision is not None:
            rounded_angles = rounded_angles.round(args.precision)
        diag_str = 'diag({})'.format(', '.join(
            proper_repr(angle) for angle in rounded_angles))
        return protocols.CircuitDiagramInfo((diag_str, '#2'))

    def __pow__(self, exponent: Any) -> 'TwoQubitDiagonalGate':
        if not isinstance(exponent, (int, float, sympy.Basic)):
            raise ValueError('No angle')
            return None
        return TwoQubitDiagonalGate([
            protocols.mul(angle, exponent, NotImplemented)
            for angle in self._diag_angles_radians
        ])

    def _value_equality_values_(self):
        return tuple(self._diag_angles_radians)

    def __repr__(self) -> str:
        return 'cirq.TwoQubitDiagonalGate([{}])'.format(','.join(
            proper_repr(angle) for angle in self._diag_angles_radians))

    def _quil_(self, qubits: Tuple['cirq.Qid', ...],
               formatter: 'cirq.QuilFormatter') -> Optional[str]:
        if qubits[0] == 0 and qubits[1] == 0:
            return formatter.format('CPHASE00 {0} {1} {2}\n', self._exponent, qubits[0], qubits[1])
        if qubits[0] == 1 and qubits[1] == 0:
            return formatter.format('CPHASE01 {0} {1} {2}\n', self._exponent, qubits[0], qubits[1])
        if qubits[0] == 0 and qubits[1] == 1:
            return formatter.format('CPHASE10 {0} {1} {2}\n', self._exponent, qubits[0], qubits[1])   
        return None 

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        if self._is_parameterized_():
            return NotImplemented
        for index, angle in enumerate(self._diag_angles_radians):
            subspace_index = args.subspace_index(big_endian_bits_int=index)
            args.target_tensor[subspace_index] *= np.exp(1j * angle)
        return args.target_tensor

    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_()

    # def _unitary_(self) -> np.ndarray:
    #     if self._is_parameterized_():
    #         return None
    #     return np.diag(np.diag(np.exp(1j * self._diag_angles_radians))