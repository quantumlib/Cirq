# Copyright 2022 The Cirq Developers
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

from typing import AbstractSet, Any, Dict, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numbers

import numpy as np
import sympy

from cirq import value, ops, protocols, linalg
from cirq.ops import raw_types
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class PhasedXZGate(raw_types.Gate):
    r"""A single qubit gate equivalent to the circuit $Z^{-a} X^x Z^{a} Z^z$ (in time order).

    The unitary matrix of `cirq.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)` is:
    $$
        \begin{bmatrix}
            e^{i \pi x / 2} \cos(\pi x /2) & -i e^{i \pi (x/2 - a)} \sin(\pi x / 2) \\
             -i e^{i \pi (x/2 + z + a)} \sin(\pi x / 2) &  e^{i \pi (x / 2 + z)} \cos(\pi x /2)
        \end{bmatrix}
    $$

    This gate can be thought of as a `cirq.PhasedXPowGate` followed by a `cirq.ZPowGate`.

    The axis phase exponent ($a$) decides which axis in the XY plane to rotate
    around. The amount of rotation around that axis is decided by the x
    exponent ($x$). Then the z exponent ($z$) decides how much to finally phase the qubit.

    Every single qubit gate can be written as a single `cirq.PhasedXZGate`.
    """

    def __init__(
        self,
        *,
        x_exponent: Union[float, sympy.Expr],
        z_exponent: Union[float, sympy.Expr],
        axis_phase_exponent: Union[float, sympy.Expr],
    ) -> None:
        """Inits PhasedXZGate.

        Args:
            x_exponent: Determines how much to rotate during the
                axis-in-XY-plane rotation. The $x$ in $Z^z Z^a X^x Z^{-a}$.
            z_exponent: The amount of phasing to apply after the
                axis-in-XY-plane rotation. The $z$ in $Z^z Z^a X^x Z^{-a}$.
            axis_phase_exponent: Determines which axis to rotate around during
                the axis-in-XY-plane rotation. The $a$ in $Z^z Z^a X^x Z^{-a}$.
        """
        self._x_exponent = x_exponent
        self._z_exponent = z_exponent
        self._axis_phase_exponent = axis_phase_exponent

    @classmethod
    def from_zyz_angles(cls, z0_rad: float, y_rad: float, z1_rad: float) -> 'cirq.PhasedXZGate':
        r"""Create a PhasedXZGate from ZYZ angles.

        The returned gate is equivalent to $Rz(z0\_rad) Ry(y\_rad) Rz(z1\_rad)$ (in time order).
        """
        return cls.from_zyz_exponents(z0=z0_rad / np.pi, y=y_rad / np.pi, z1=z1_rad / np.pi)

    @classmethod
    def from_zyz_exponents(cls, z0: float, y: float, z1: float) -> 'cirq.PhasedXZGate':
        """Create a PhasedXZGate from ZYZ exponents.

        The returned gate is equivalent to $Z^{z0} Y^y Z^{z1}$ (in time order).
        """
        return PhasedXZGate(axis_phase_exponent=-z0 + 0.5, x_exponent=y, z_exponent=z0 + z1)

    def _canonical(self) -> 'cirq.PhasedXZGate':
        x = self.x_exponent
        z = self.z_exponent
        a = self.axis_phase_exponent

        # Canonicalize X exponent into (-1, +1].
        if not isinstance(x, sympy.Expr):
            x %= 2
            if x > 1.0:
                x -= 2

        # Axis phase exponent is irrelevant if there is no X exponent.
        if x == 0:
            a = 0.0
        # For 180 degree X rotations, the axis phase and z exponent overlap.
        if x == 1 and z != 0:
            a += z / 2
            z = 0.0

        # Canonicalize Z exponent into (-1, +1].
        if not isinstance(z, sympy.Expr):
            z %= 2
            if z > 1.0:
                z -= 2

        # Canonicalize axis phase exponent into (-0.5, +0.5].
        if not isinstance(a, sympy.Expr):
            a %= 2
            if a > 1.0:
                a -= 2
            if a <= -0.5:
                a += 1
                if x != 1:
                    x = -x
            elif a > 0.5:
                a -= 1
                if x != 1:
                    x = -x

        return PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)

    @property
    def x_exponent(self) -> Union[float, sympy.Expr]:
        return self._x_exponent

    @property
    def z_exponent(self) -> Union[float, sympy.Expr]:
        return self._z_exponent

    @property
    def axis_phase_exponent(self) -> Union[float, sympy.Expr]:
        return self._axis_phase_exponent

    def _value_equality_values_(self):
        c = self._canonical()
        return (
            value.PeriodicValue(c._x_exponent, 2),
            value.PeriodicValue(c._z_exponent, 2),
            value.PeriodicValue(c._axis_phase_exponent, 2),
        )

    @staticmethod
    def from_matrix(mat: np.ndarray) -> 'cirq.PhasedXZGate':
        pre_phase, rotation, post_phase = linalg.deconstruct_single_qubit_matrix_into_angles(mat)
        pre_phase /= np.pi
        post_phase /= np.pi
        rotation /= np.pi
        pre_phase -= 0.5
        post_phase += 0.5
        return PhasedXZGate(
            x_exponent=rotation, axis_phase_exponent=-pre_phase, z_exponent=post_phase + pre_phase
        )._canonical()

    def with_z_exponent(self, z_exponent: Union[float, sympy.Expr]) -> 'cirq.PhasedXZGate':
        return PhasedXZGate(
            axis_phase_exponent=self._axis_phase_exponent,
            x_exponent=self._x_exponent,
            z_exponent=z_exponent,
        )

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        from cirq.circuits import qasm_output

        qasm_gate = qasm_output.QasmUGate(
            lmda=0.5 - self._axis_phase_exponent,
            theta=self._x_exponent,
            phi=self._z_exponent + self._axis_phase_exponent - 0.5,
        )
        return protocols.qasm(qasm_gate, args=args, qubits=qubits)

    def _num_qubits_(self) -> int:
        return 1

    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_()

    def _unitary_(self) -> Optional[np.ndarray]:
        """See `cirq.SupportsUnitary`."""
        if self._is_parameterized_():
            return None
        z_pre = protocols.unitary(ops.Z**-self._axis_phase_exponent)
        x = protocols.unitary(ops.X**self._x_exponent)
        z_post = protocols.unitary(ops.Z ** (self._axis_phase_exponent + self._z_exponent))
        return z_post @ x @ z_pre

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> Iterator['cirq.OP_TREE']:
        q = qubits[0]
        yield ops.Z(q) ** -self._axis_phase_exponent
        yield ops.X(q) ** self._x_exponent
        yield ops.Z(q) ** (self._axis_phase_exponent + self._z_exponent)

    def __pow__(self, exponent: float) -> 'PhasedXZGate':
        if exponent == 1:
            return self
        if exponent == -1:
            return PhasedXZGate(
                x_exponent=-self._x_exponent,
                z_exponent=-self._z_exponent,
                axis_phase_exponent=self._z_exponent + self.axis_phase_exponent,
            )
        return NotImplemented

    def _is_parameterized_(self) -> bool:
        """See `cirq.SupportsParameterization`."""
        return (
            protocols.is_parameterized(self._x_exponent)
            or protocols.is_parameterized(self._z_exponent)
            or protocols.is_parameterized(self._axis_phase_exponent)
        )

    def _parameter_names_(self) -> AbstractSet[str]:
        """See `cirq.SupportsParameterization`."""
        return (
            protocols.parameter_names(self._x_exponent)
            | protocols.parameter_names(self._z_exponent)
            | protocols.parameter_names(self._axis_phase_exponent)
        )

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'cirq.PhasedXZGate':
        """See `cirq.SupportsParameterization`."""
        z_exponent = resolver.value_of(self._z_exponent, recursive)
        x_exponent = resolver.value_of(self._x_exponent, recursive)
        axis_phase_exponent = resolver.value_of(self._axis_phase_exponent, recursive)
        if isinstance(z_exponent, numbers.Complex):
            if isinstance(z_exponent, numbers.Real):
                z_exponent = float(z_exponent)
            else:
                raise ValueError(f'Complex exponent {z_exponent} not allowed in cirq.PhasedXZGate')
        if isinstance(x_exponent, numbers.Complex):
            if isinstance(x_exponent, numbers.Real):
                x_exponent = float(x_exponent)
            else:
                raise ValueError(f'Complex exponent {x_exponent} not allowed in cirq.PhasedXZGate')
        if isinstance(axis_phase_exponent, numbers.Complex):
            if isinstance(axis_phase_exponent, numbers.Real):
                axis_phase_exponent = float(axis_phase_exponent)
            else:
                raise ValueError(
                    f'Complex exponent {axis_phase_exponent} not allowed in cirq.PhasedXZGate'
                )
        return PhasedXZGate(
            z_exponent=z_exponent, x_exponent=x_exponent, axis_phase_exponent=axis_phase_exponent
        )

    def _phase_by_(self, phase_turns, qubit_index) -> 'cirq.PhasedXZGate':
        """See `cirq.SupportsPhase`."""
        assert qubit_index == 0
        return PhasedXZGate(
            x_exponent=self._x_exponent,
            z_exponent=self._z_exponent,
            axis_phase_exponent=self._axis_phase_exponent + phase_turns * 2,
        )

    def _pauli_expansion_(self) -> 'cirq.LinearDict[str]':
        if protocols.is_parameterized(self):
            return NotImplemented
        x_angle = np.pi * self._x_exponent / 2
        z_angle = np.pi * self._z_exponent / 2
        axis_angle = np.pi * self._axis_phase_exponent
        phase = np.exp(1j * (x_angle + z_angle))

        cx = np.cos(x_angle)
        sx = np.sin(x_angle)
        return value.LinearDict(
            {
                'I': phase * cx * np.cos(z_angle),
                'X': -1j * phase * sx * np.cos(z_angle + axis_angle),
                'Y': -1j * phase * sx * np.sin(z_angle + axis_angle),
                'Z': -1j * phase * cx * np.sin(z_angle),
            }
        )  # yapf: disable

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> str:
        """See `cirq.SupportsCircuitDiagramInfo`."""
        return (
            f'PhXZ('
            f'a={args.format_real(self._axis_phase_exponent)},'
            f'x={args.format_real(self._x_exponent)},'
            f'z={args.format_real(self._z_exponent)})'
        )

    def __str__(self) -> str:
        return protocols.circuit_diagram_info(self).wire_symbols[0]

    def __repr__(self) -> str:
        return (
            f'cirq.PhasedXZGate('
            f'axis_phase_exponent={proper_repr(self._axis_phase_exponent)},'
            f' x_exponent={proper_repr(self._x_exponent)}, '
            f'z_exponent={proper_repr(self._z_exponent)})'
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(
            self, ['axis_phase_exponent', 'x_exponent', 'z_exponent']
        )
