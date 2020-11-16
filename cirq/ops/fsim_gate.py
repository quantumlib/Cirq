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
"""Defines the fermionic simulation gate family.

This is the family of two-qubit gates that preserve excitations (number of ON
qubits), ignoring single-qubit gates and global phase. For example, when using
the second quantized representation of electrons to simulate chemistry, this is
a natural gateset because each ON qubit corresponds to an electron and in the
context of chemistry the electron count is conserved over time. This property
applies more generally to fermions, thus the name of the gate.
"""

import cmath
import math
from typing import AbstractSet, Any, Dict, Optional, Tuple, Union

import numpy as np
import sympy

import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import gate_features


@value.value_equality(approximate=True)
class FSimGate(gate_features.TwoQubitGate,
               gate_features.InterchangeableQubitsGate):
    """Fermionic simulation gate family.

    Contains all two qubit interactions that preserve excitations, up to
    single-qubit rotations and global phase.

    The unitary matrix of this gate is:

        [[1, 0, 0, 0],
         [0, a, b, 0],
         [0, b, a, 0],
         [0, 0, 0, c]]

    where:

        a = cos(theta)
        b = -i·sin(theta)
        c = exp(-i·phi)

    Note the difference in sign conventions between FSimGate and the
    ISWAP and CZPowGate:

        FSimGate(θ, φ) = ISWAP**(-2θ/π) CZPowGate(exponent=-φ/π)
    """

    def __init__(self, theta: float, phi: float) -> None:
        """
        Args:
            theta: Swap angle on the ``|01⟩`` ``|10⟩`` subspace, in radians.
                Determined by the strength and duration of the XX+YY
                interaction. Note: uses opposite sign convention to the
                iSWAP gate. Maximum strength (full iswap) is at pi/2.
            phi: Controlled phase angle, in radians. Determines how much the
                ``|11⟩`` state is phased. Note: uses opposite sign convention to
                the CZPowGate. Maximum strength (full cz) is at pi/2.
        """
        self.theta = theta
        self.phi = phi

    def _value_equality_values_(self) -> Any:
        return self.theta, self.phi

    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self.theta) or cirq.is_parameterized(
            self.phi)

    def _parameter_names_(self) -> AbstractSet[str]:
        return cirq.parameter_names(self.theta) | cirq.parameter_names(self.phi)

    def _has_unitary_(self):
        return not self._is_parameterized_()

    def _unitary_(self) -> Optional[np.ndarray]:
        if self._is_parameterized_():
            return None
        a = math.cos(self.theta)
        b = -1j * math.sin(self.theta)
        c = cmath.exp(-1j * self.phi)
        return np.array([
            [1, 0, 0, 0],
            [0, a, b, 0],
            [0, b, a, 0],
            [0, 0, 0, c],
        ])

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        a = math.cos(self.theta)
        b = -1j * math.sin(self.theta)
        c = cmath.exp(-1j * self.phi)
        return value.LinearDict({
            'II': (1 + c) / 4 + a / 2,
            'IZ': (1 - c) / 4,
            'ZI': (1 - c) / 4,
            'ZZ': (1 + c) / 4 - a / 2,
            'XX': b / 2,
            'YY': b / 2,
        })

    def _resolve_parameters_(self, param_resolver: 'cirq.ParamResolver'
                            ) -> 'cirq.FSimGate':
        return FSimGate(
            protocols.resolve_parameters(self.theta, param_resolver),
            protocols.resolve_parameters(self.phi, param_resolver))

    def _apply_unitary_(self,
                        args: 'cirq.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if cirq.is_parameterized(self):
            return None
        if self.theta != 0:
            inner_matrix = protocols.unitary(cirq.rx(2 * self.theta))
            oi = args.subspace_index(0b01)
            io = args.subspace_index(0b10)
            out = cirq.apply_matrix_to_slices(args.target_tensor,
                                              inner_matrix,
                                              slices=[oi, io],
                                              out=args.available_buffer)
        else:
            out = args.target_tensor
        if self.phi != 0:
            ii = args.subspace_index(0b11)
            out[ii] *= cmath.exp(-1j * self.phi)
        return out

    def _decompose_(self, qubits) -> 'cirq.OP_TREE':
        a, b = qubits
        xx = cirq.XXPowGate(exponent=self.theta / np.pi, global_shift=-0.5)
        yy = cirq.YYPowGate(exponent=self.theta / np.pi, global_shift=-0.5)
        yield xx(a, b)
        yield yy(a, b)
        yield cirq.CZ(a, b)**(-self.phi / np.pi)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                              ) -> Tuple[str, ...]:
        t = args.format_radians(self.theta)
        p = args.format_radians(self.phi)
        return f'FSim({t}, {p})', f'FSim({t}, {p})'

    def __pow__(self, power) -> 'FSimGate':
        return FSimGate(cirq.mul(self.theta, power), cirq.mul(self.phi, power))

    def __repr__(self) -> str:
        t = proper_repr(self.theta)
        p = proper_repr(self.phi)
        return f'cirq.FSimGate(theta={t}, phi={p})'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['theta', 'phi'])


@value.value_equality(approximate=True)
class PhasedFSimGate(gate_features.TwoQubitGate,
                     gate_features.InterchangeableQubitsGate):
    """General excitation-preserving two-qubit gate.

    The unitary matrix of this gate is:

        [[1,           0,             0,           0],
         [0,    f1 cos(θ), -i f2 sin(θ),           0],
         [0, -i f3 sin(θ),    f4 cos(θ),           0],
         [0,           0,             0, f5 exp(-iφ)]]

    where the phase factors are

        f1 = exp(i∆p + i∆m)
        f2 = exp(i∆p - i∆mod)
        f3 = exp(i∆p + i∆mod)
        f4 = exp(i∆p - i∆m)
        f5 = exp(2i∆p)

    and the correspondence between the symbols above and the parameters
    to __init__ is as follows:

        θ:    theta,
        φ:    phi,
        ∆p:   delta_plus,
        ∆m:   delta_minus,
        ∆mod: delta_minus_off_diagonal.

    See also eq (43) in https://arxiv.org/abs/1910.11333.

    There is another useful parametrization of PhasedFSimGate based on
    the fact that the gate is equivalent to the following circuit:

        0: ───Rz(b0)───FSim(theta, phi)───Rz(a0)───
                       │
        1: ───Rz(b1)───FSim(theta, phi)───Rz(a1)───

    where b0 and b1 are phase angles to be applied before the core FSimGate,
    a0 and a1 are phase angles to be applied after FSimGate and theta and
    phi specify the core FSimGate. Use the static factory function
    PhasedFSimGate.from_phase_angles_and_fsim to instantiate the gate
    using this parametrization.

    Note that the theta and phi parameters in the two parametrizations are
    the same.

    The matrix above is block diagonal where the middle block may be any
    element of U(2) and the bottom right block may be any element of U(1).
    Consequently, five real parameters are required to specify an instance
    of PhasedFSimGate. Therefore, the second parametrization is not injective.
    Specifically, for any d

        cirq.PhasedFSimGate.from_phase_angles_and_fsim(
            theta, phi, (b0, b1), (a0, a1))

    and

        cirq.PhasedFSimGate.from_phase_angles_and_fsim(
            theta, phi, (b0 + d, b1 + d), (a0 - d, a1 - d))

    specify the same gate and therefore the two instances will compare as
    equal up to numerical error. Another consequence of the non-injective
    character of the second parametrization is the fact that the properties
    phase_angles_before and phase_angles_after may return different phase
    angles than the ones used in the call to from_phase_angles_and_fsim.
    """

    def __init__(self,
                 theta: Union[float, sympy.Basic],
                 phi: Union[float, sympy.Basic],
                 delta_plus: Union[float, sympy.Basic] = 0.0,
                 delta_minus: Union[float, sympy.Basic] = 0.0,
                 delta_minus_off_diagonal: Union[float, sympy.Basic] = 0.0
                ) -> None:
        """
        Args:
            theta: Swap angle on the ``|01⟩`` ``|10⟩`` subspace, in radians.
                See class docstring above for details.
            phi: Controlled phase angle, in radians. See class docstring
                above for details.
            delta_plus: One of the phase angles, in radians. See class
                docstring above for details.
            delta_minus: One of the phase angles, in radians. See class
                docstring above for details.
            delta_minus_off_diagonal: One of the phase angles, in radians.
                See class docstring above for details.
        """

        def canonicalize(value: Union[float, sympy.Basic]
                        ) -> Union[float, sympy.Basic]:
            """Assumes value is 2π-periodic and shifts it into [-π, π]."""
            if protocols.is_parameterized(value):
                return value
            period = 2 * np.pi
            return value - period * np.round(value / period)

        self.theta = canonicalize(theta)
        self.phi = canonicalize(phi)
        self.delta_plus = canonicalize(delta_plus)
        self.delta_minus = canonicalize(delta_minus)
        self.delta_minus_off_diagonal = canonicalize(delta_minus_off_diagonal)

    @staticmethod
    def from_phase_angles_and_fsim(
            theta: Union[float, sympy.Basic], phi: Union[float, sympy.Basic],
            phase_angles_before: Tuple[Union[float, sympy.
                                             Basic], Union[float, sympy.Basic]],
            phase_angles_after: Tuple[Union[float, sympy.
                                            Basic], Union[float, sympy.Basic]]
    ) -> 'PhasedFSimGate':
        """Creates PhasedFSimGate using an alternate parametrization.

        Args:
            theta: Swap angle on the ``|01⟩`` ``|10⟩`` subspace, in radians.
                See class docstring above for details.
            phi: Controlled phase angle, in radians. See class docstring
                above for details.
            phase_angles_before: 2-tuple of phase angles to apply to each qubit
                before the core FSimGate. See class docstring for details.
            phase_angles_after: 2-tuple of phase angles to apply to each qubit
                after the core FSimGate. See class docstring for details.
        """
        b0, b1 = phase_angles_before
        a0, a1 = phase_angles_after
        delta_plus = (b0 + b1 + a0 + a1) / 2.0
        delta_minus = (-b0 + b1 - a0 + a1) / 2.0
        delta_minus_off_diagonal = (-b0 + b1 + a0 - a1) / 2.0
        return PhasedFSimGate(theta, phi, delta_plus, delta_minus,
                              delta_minus_off_diagonal)

    @property
    def phase_angles_before(
            self
    ) -> Tuple[Union[float, sympy.Basic], Union[float, sympy.Basic]]:
        """Returns 2-tuple of phase angles applied to qubits before FSimGate."""
        b0 = (self.delta_plus - self.delta_minus -
              self.delta_minus_off_diagonal) / 2.0
        b1 = (self.delta_plus + self.delta_minus +
              self.delta_minus_off_diagonal) / 2.0
        return b0, b1

    @property
    def phase_angles_after(
            self
    ) -> Tuple[Union[float, sympy.Basic], Union[float, sympy.Basic]]:
        """Returns 2-tuple of phase angles applied to qubits after FSimGate."""
        a0 = (self.delta_plus - self.delta_minus +
              self.delta_minus_off_diagonal) / 2.0
        a1 = (self.delta_plus + self.delta_minus -
              self.delta_minus_off_diagonal) / 2.0
        return a0, a1

    def _value_equality_values_(self) -> Any:
        return (self.theta, self.phi, self.delta_plus, self.delta_minus,
                self.delta_minus_off_diagonal)

    def _is_parameterized_(self) -> bool:
        return (cirq.is_parameterized(self.theta) or
                cirq.is_parameterized(self.phi) or
                cirq.is_parameterized(self.delta_plus) or
                cirq.is_parameterized(self.delta_minus) or
                cirq.is_parameterized(self.delta_minus_off_diagonal))

    def _has_unitary_(self):
        return not self._is_parameterized_()

    def _unitary_(self) -> Optional[np.ndarray]:
        if self._is_parameterized_():
            return None
        a = math.cos(self.theta)
        b = -1j * math.sin(self.theta)
        c = cmath.exp(-1j * self.phi)
        f1 = cmath.exp(1j * self.delta_plus + 1j * self.delta_minus)
        f2 = cmath.exp(1j * self.delta_plus -
                       1j * self.delta_minus_off_diagonal)
        f3 = cmath.exp(1j * self.delta_plus +
                       1j * self.delta_minus_off_diagonal)
        f4 = cmath.exp(1j * self.delta_plus - 1j * self.delta_minus)
        f5 = cmath.exp(2j * self.delta_plus)
        return np.array([
            [1, 0, 0, 0],
            [0, f1 * a, f2 * b, 0],
            [0, f3 * b, f4 * a, 0],
            [0, 0, 0, f5 * c],
        ])

    def _resolve_parameters_(self, param_resolver: 'cirq.ParamResolver'
                            ) -> 'cirq.PhasedFSimGate':
        return PhasedFSimGate(
            protocols.resolve_parameters(self.theta, param_resolver),
            protocols.resolve_parameters(self.phi, param_resolver),
            protocols.resolve_parameters(self.delta_plus, param_resolver),
            protocols.resolve_parameters(self.delta_minus, param_resolver),
            protocols.resolve_parameters(self.delta_minus_off_diagonal,
                                         param_resolver))

    def _apply_unitary_(self,
                        args: 'cirq.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if cirq.is_parameterized(self):
            return None
        oi = args.subspace_index(0b01)
        io = args.subspace_index(0b10)
        ii = args.subspace_index(0b11)
        if (self.theta != 0 or self.delta_minus != 0 or
                self.delta_minus_off_diagonal != 0):
            rx = protocols.unitary(cirq.rx(2 * self.theta))
            rz1 = protocols.unitary(
                cirq.rz(self.delta_minus - self.delta_minus_off_diagonal))
            rz2 = protocols.unitary(
                cirq.rz(self.delta_minus + self.delta_minus_off_diagonal))
            inner_matrix = rz1 @ rx @ rz2
            out = cirq.apply_matrix_to_slices(args.target_tensor,
                                              inner_matrix,
                                              slices=[oi, io],
                                              out=args.available_buffer)
        else:
            out = args.target_tensor
        if self.phi != 0:
            out[ii] *= cmath.exp(-1j * self.phi)
        if self.delta_plus != 0:
            f = cmath.exp(1j * self.delta_plus)
            out[oi] *= f
            out[io] *= f
            out[ii] *= f * f
        return out

    def _decompose_(self, qubits) -> 'cirq.OP_TREE':
        q0, q1 = qubits
        q0_z_exponent_before = self.phase_angles_before[0] / np.pi
        q1_z_exponent_before = self.phase_angles_before[1] / np.pi
        q0_z_exponent_after = self.phase_angles_after[0] / np.pi
        q1_z_exponent_after = self.phase_angles_after[1] / np.pi
        yield cirq.Z(q0)**q0_z_exponent_before
        yield cirq.Z(q1)**q1_z_exponent_before
        yield FSimGate(self.theta, self.phi).on(q0, q1)
        yield cirq.Z(q0)**q0_z_exponent_after
        yield cirq.Z(q1)**q1_z_exponent_after

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                              ) -> Tuple[str, ...]:
        t = args.format_radians(self.theta)
        p = args.format_radians(self.phi)
        dp = args.format_radians(self.delta_plus)
        dm = args.format_radians(self.delta_minus)
        dmod = args.format_radians(self.delta_minus_off_diagonal)
        return (f'PhFSim({t}, {p}, {dp}, {dm}, {dmod})',
                f'PhFSim({t}, {p}, {dp}, {dm}, {dmod})')

    def __repr__(self) -> str:
        t = proper_repr(self.theta)
        p = proper_repr(self.phi)
        dp = proper_repr(self.delta_plus)
        dm = proper_repr(self.delta_minus)
        dmod = proper_repr(self.delta_minus_off_diagonal)
        return (f'cirq.PhasedFSimGate(theta={t}, phi={p}, delta_plus={dp}, '
                f'delta_minus={dm}, delta_minus_off_diagonal={dmod})')

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, [
            'theta', 'phi', 'delta_plus', 'delta_minus',
            'delta_minus_off_diagonal'
        ])
