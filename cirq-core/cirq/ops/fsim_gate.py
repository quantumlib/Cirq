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


def _canonicalize(value: Union[float, sympy.Basic]) -> Union[float, sympy.Basic]:
    """Assumes value is 2π-periodic and shifts it into [-π, π]."""
    if protocols.is_parameterized(value):
        return value
    period = 2 * np.pi
    return value - period * np.round(value / period)


def _zero_mod_pi(param: Union[float, sympy.Basic]) -> bool:
    """Returns True iff param, assumed to be in [-pi, pi], is 0 (mod pi)."""
    return param in (-np.pi, 0.0, np.pi, -sympy.pi, sympy.pi)


def _half_pi_mod_pi(param: Union[float, sympy.Basic]) -> bool:
    """Returns True iff param, assumed to be in [-pi, pi], is pi/2 (mod pi)."""
    return param in (-np.pi / 2, np.pi / 2, -sympy.pi / 2, sympy.pi / 2)


@value.value_equality(approximate=True)
class FSimGate(gate_features.TwoQubitGate, gate_features.InterchangeableQubitsGate):
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
        """Inits FSimGate.

        Args:
            theta: Swap angle on the ``|01⟩`` ``|10⟩`` subspace, in radians.
                Determined by the strength and duration of the XX+YY
                interaction. Note: uses opposite sign convention to the
                iSWAP gate. Maximum strength (full iswap) is at pi/2.
            phi: Controlled phase angle, in radians. Determines how much the
                ``|11⟩`` state is phased. Note: uses opposite sign convention to
                the CZPowGate. Maximum strength (full cz) is at pi/2.
        """
        self.theta = _canonicalize(theta)
        self.phi = _canonicalize(phi)

    def _value_equality_values_(self) -> Any:
        return self.theta, self.phi

    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self.theta) or cirq.is_parameterized(self.phi)

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
        return np.array(
            [
                [1, 0, 0, 0],
                [0, a, b, 0],
                [0, b, a, 0],
                [0, 0, 0, c],
            ]
        )

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        a = math.cos(self.theta)
        b = -1j * math.sin(self.theta)
        c = cmath.exp(-1j * self.phi)
        return value.LinearDict(
            {
                'II': (1 + c) / 4 + a / 2,
                'IZ': (1 - c) / 4,
                'ZI': (1 - c) / 4,
                'ZZ': (1 + c) / 4 - a / 2,
                'XX': b / 2,
                'YY': b / 2,
            }
        )

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'cirq.FSimGate':
        return FSimGate(
            protocols.resolve_parameters(self.theta, resolver, recursive),
            protocols.resolve_parameters(self.phi, resolver, recursive),
        )

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if cirq.is_parameterized(self):
            return None
        if self.theta != 0:
            inner_matrix = protocols.unitary(cirq.rx(2 * self.theta))
            oi = args.subspace_index(0b01)
            io = args.subspace_index(0b10)
            out = cirq.apply_matrix_to_slices(
                args.target_tensor, inner_matrix, slices=[oi, io], out=args.available_buffer
            )
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
        yield cirq.CZ(a, b) ** (-self.phi / np.pi)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Tuple[str, ...]:
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
class PhasedFSimGate(gate_features.TwoQubitGate, gate_features.InterchangeableQubitsGate):
    """General excitation-preserving two-qubit gate.

    The unitary matrix of PhasedFSimGate(θ, ζ, χ, γ, φ) is:

        [[1,                       0,                       0,            0],
         [0,    exp(-iγ - iζ) cos(θ), -i exp(-iγ + iχ) sin(θ),            0],
         [0, -i exp(-iγ - iχ) sin(θ),    exp(-iγ + iζ) cos(θ),            0],
         [0,                       0,                       0, exp(-2iγ-iφ)]].

    This parametrization follows eq (18) in https://arxiv.org/abs/2010.07965.
    See also eq (43) in https://arxiv.org/abs/1910.11333 for an older variant
    which uses the same θ and φ parameters, but its three phase angles have
    different names and opposite sign. Specifically, ∆+ angle corresponds to
    -γ, ∆- corresponds to -ζ and ∆-,off corresponds to -χ.

    Another useful parametrization of PhasedFSimGate is based on the fact that
    the gate is equivalent up to global phase to the following circuit:

        0: ───Rz(α0)───FSim(θ, φ)───Rz(β0)───
                       │
        1: ───Rz(α1)───FSim(θ, φ)───Rz(β1)───

    where α0 and α1 are Rz angles to be applied before the core FSimGate,
    β0 and β1 are Rz angles to be applied after FSimGate and θ and φ specify
    the core FSimGate. Use the static factory function from_fsim_rz to
    instantiate the gate using this parametrization.

    Note that the θ and φ parameters in the two parametrizations are the same.

    The matrix above is block diagonal where the middle block may be any
    element of U(2) and the bottom right block may be any element of U(1).
    Consequently, five real parameters are required to specify an instance
    of PhasedFSimGate. Therefore, the second parametrization is not injective.
    Indeed, for any angle δ

        cirq.PhasedFSimGate.from_fsim_rz(θ, φ, (α0, α1), (β0, β1))

    and

        cirq.PhasedFSimGate.from_fsim_rz(θ, φ,
                                         (α0 + δ, α1 + δ),
                                         (β0 - δ, β1 - δ))

    specify the same gate and therefore the two instances will compare as
    equal up to numerical error. Another consequence of the non-injective
    character of the second parametrization is the fact that the properties
    rz_angles_before and rz_angles_after may return different Rz angles
    than the ones used in the call to from_fsim_rz.

    This gate is generally not symmetric under exchange of qubits. It becomes
    symmetric if both of the following conditions are satisfied:
     * ζ = kπ or θ = π/2 + lπ for k and l integers,
     * χ = kπ or θ = lπ for k and l integers.
    """

    def __init__(
        self,
        theta: Union[float, sympy.Basic],
        zeta: Union[float, sympy.Basic] = 0.0,
        chi: Union[float, sympy.Basic] = 0.0,
        gamma: Union[float, sympy.Basic] = 0.0,
        phi: Union[float, sympy.Basic] = 0.0,
    ) -> None:
        """Inits PhasedFSimGate.

        Args:
            theta: Swap angle on the ``|01⟩`` ``|10⟩`` subspace, in radians.
                See class docstring above for details.
            zeta: One of the phase angles, in radians. See class
                docstring above for details.
            chi: One of the phase angles, in radians.
                See class docstring above for details.
            gamma: One of the phase angles, in radians. See class
                docstring above for details.
            phi: Controlled phase angle, in radians. See class docstring
                above for details.
        """
        self.theta = _canonicalize(theta)
        self.zeta = _canonicalize(zeta)
        self.chi = _canonicalize(chi)
        self.gamma = _canonicalize(gamma)
        self.phi = _canonicalize(phi)

    @staticmethod
    def from_fsim_rz(
        theta: Union[float, sympy.Basic],
        phi: Union[float, sympy.Basic],
        rz_angles_before: Tuple[Union[float, sympy.Basic], Union[float, sympy.Basic]],
        rz_angles_after: Tuple[Union[float, sympy.Basic], Union[float, sympy.Basic]],
    ) -> 'PhasedFSimGate':
        """Creates PhasedFSimGate using an alternate parametrization.

        Args:
            theta: Swap angle on the ``|01⟩`` ``|10⟩`` subspace, in radians.
                See class docstring above for details.
            phi: Controlled phase angle, in radians. See class docstring
                above for details.
            rz_angles_before: 2-tuple of phase angles to apply to each qubit
                before the core FSimGate. See class docstring for details.
            rz_angles_after: 2-tuple of phase angles to apply to each qubit
                after the core FSimGate. See class docstring for details.
        """
        b0, b1 = rz_angles_before
        a0, a1 = rz_angles_after
        gamma = (-b0 - b1 - a0 - a1) / 2.0
        zeta = (b0 - b1 + a0 - a1) / 2.0
        chi = (b0 - b1 - a0 + a1) / 2.0
        return PhasedFSimGate(theta, zeta, chi, gamma, phi)

    @property
    def rz_angles_before(self) -> Tuple[Union[float, sympy.Basic], Union[float, sympy.Basic]]:
        """Returns 2-tuple of phase angles applied to qubits before FSimGate."""
        b0 = (-self.gamma + self.zeta + self.chi) / 2.0
        b1 = (-self.gamma - self.zeta - self.chi) / 2.0
        return b0, b1

    @property
    def rz_angles_after(self) -> Tuple[Union[float, sympy.Basic], Union[float, sympy.Basic]]:
        """Returns 2-tuple of phase angles applied to qubits after FSimGate."""
        a0 = (-self.gamma + self.zeta - self.chi) / 2.0
        a1 = (-self.gamma - self.zeta + self.chi) / 2.0
        return a0, a1

    def _zeta_insensitive(self) -> bool:
        return _half_pi_mod_pi(self.theta)

    def _chi_insensitive(self) -> bool:
        return _zero_mod_pi(self.theta)

    def qubit_index_to_equivalence_group_key(self, index: int) -> int:
        """Returns a key that differs between non-interchangeable qubits."""
        if (_zero_mod_pi(self.zeta) or self._zeta_insensitive()) and (
            _zero_mod_pi(self.chi) or self._chi_insensitive()
        ):
            return 0
        return index

    def _value_equality_values_(self) -> Any:
        if self._zeta_insensitive():
            return (self.theta, 0.0, self.chi, self.gamma, self.phi)
        if self._chi_insensitive():
            return (self.theta, self.zeta, 0.0, self.gamma, self.phi)
        return (self.theta, self.zeta, self.chi, self.gamma, self.phi)

    def _is_parameterized_(self) -> bool:
        return (
            cirq.is_parameterized(self.theta)
            or cirq.is_parameterized(self.zeta)
            or cirq.is_parameterized(self.chi)
            or cirq.is_parameterized(self.gamma)
            or cirq.is_parameterized(self.phi)
        )

    def _has_unitary_(self):
        return not self._is_parameterized_()

    def _unitary_(self) -> Optional[np.ndarray]:
        if self._is_parameterized_():
            return None
        a = math.cos(self.theta)
        b = -1j * math.sin(self.theta)
        c = cmath.exp(-1j * self.phi)
        f1 = cmath.exp(-1j * self.gamma - 1j * self.zeta)
        f2 = cmath.exp(-1j * self.gamma + 1j * self.chi)
        f3 = cmath.exp(-1j * self.gamma - 1j * self.chi)
        f4 = cmath.exp(-1j * self.gamma + 1j * self.zeta)
        f5 = cmath.exp(-2j * self.gamma)
        return np.array(
            [
                [1, 0, 0, 0],
                [0, f1 * a, f2 * b, 0],
                [0, f3 * b, f4 * a, 0],
                [0, 0, 0, f5 * c],
            ]
        )

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'cirq.PhasedFSimGate':
        return PhasedFSimGate(
            protocols.resolve_parameters(self.theta, resolver, recursive),
            protocols.resolve_parameters(self.zeta, resolver, recursive),
            protocols.resolve_parameters(self.chi, resolver, recursive),
            protocols.resolve_parameters(self.gamma, resolver, recursive),
            protocols.resolve_parameters(self.phi, resolver, recursive),
        )

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if cirq.is_parameterized(self):
            return None
        oi = args.subspace_index(0b01)
        io = args.subspace_index(0b10)
        ii = args.subspace_index(0b11)
        if self.theta != 0 or self.zeta != 0 or self.chi != 0:
            rx = protocols.unitary(cirq.rx(2 * self.theta))
            rz1 = protocols.unitary(cirq.rz(-self.zeta + self.chi))
            rz2 = protocols.unitary(cirq.rz(-self.zeta - self.chi))
            inner_matrix = rz1 @ rx @ rz2
            out = cirq.apply_matrix_to_slices(
                args.target_tensor, inner_matrix, slices=[oi, io], out=args.available_buffer
            )
        else:
            out = args.target_tensor
        if self.phi != 0:
            out[ii] *= cmath.exp(-1j * self.phi)
        if self.gamma != 0:
            f = cmath.exp(-1j * self.gamma)
            out[oi] *= f
            out[io] *= f
            out[ii] *= f * f
        return out

    def _decompose_(self, qubits) -> 'cirq.OP_TREE':
        """Decomposes self into Z rotations and FSimGate.

        Note that Z rotations returned by this method have unusual global phase
        in that one of their eigenvalues is 1. This ensures the decomposition
        agrees with the matrix specified in class docstring. In particular, it
        makes the top left element of the matrix equal to 1.
        """

        def to_exponent(angle_rads: Union[float, sympy.Basic]) -> Union[float, sympy.Basic]:
            """Divides angle_rads by symbolic or numerical pi."""
            pi = sympy.pi if protocols.is_parameterized(angle_rads) else np.pi
            return angle_rads / pi

        q0, q1 = qubits
        before = self.rz_angles_before
        after = self.rz_angles_after
        yield cirq.Z(q0) ** to_exponent(before[0])
        yield cirq.Z(q1) ** to_exponent(before[1])
        yield FSimGate(self.theta, self.phi).on(q0, q1)
        yield cirq.Z(q0) ** to_exponent(after[0])
        yield cirq.Z(q1) ** to_exponent(after[1])

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Tuple[str, ...]:
        theta = args.format_radians(self.theta)
        zeta = args.format_radians(self.zeta)
        chi = args.format_radians(self.chi)
        gamma = args.format_radians(self.gamma)
        phi = args.format_radians(self.phi)
        return (
            f'PhFSim({theta}, {zeta}, {chi}, {gamma}, {phi})',
            f'PhFSim({theta}, {zeta}, {chi}, {gamma}, {phi})',
        )

    def __repr__(self) -> str:
        theta = proper_repr(self.theta)
        zeta = proper_repr(self.zeta)
        chi = proper_repr(self.chi)
        gamma = proper_repr(self.gamma)
        phi = proper_repr(self.phi)
        return (
            f'cirq.PhasedFSimGate(theta={theta}, zeta={zeta}, chi={chi}, '
            f'gamma={gamma}, phi={phi})'
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['theta', 'zeta', 'chi', 'gamma', 'phi'])
