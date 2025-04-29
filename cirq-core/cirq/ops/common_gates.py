# Copyright 2018 The Cirq Developers
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

"""Quantum gates that are commonly used in the literature.

This module creates Gate instances for the following gates:
    X,Y,Z: Pauli gates.
    H,S: Clifford gates.
    T: A non-Clifford gate.
    CZ: Controlled phase gate.
    CNOT: Controlled not gate.

Each of these are implemented as EigenGates, which means that they can be
raised to a power (i.e. cirq.H**0.5). See the definition in EigenGate.
"""

from __future__ import annotations

from types import NotImplementedType
from typing import Any, cast, Collection, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sympy

import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import control_values as cv, controlled_gate, eigen_gate, gate_features, raw_types
from cirq.ops.measurement_gate import MeasurementGate
from cirq.ops.swap_gates import ISWAP, ISwapPowGate, SWAP, SwapPowGate

assert all(
    [ISWAP, SWAP, ISwapPowGate, SwapPowGate, MeasurementGate]
), """
Included for compatibility. Please continue to use top-level cirq.{thing}
imports.
"""


def _pi(rads):
    return sympy.pi if protocols.is_parameterized(rads) else np.pi


@value.value_equality
class XPowGate(eigen_gate.EigenGate):
    r"""A gate that rotates around the X axis of the Bloch sphere.

    The unitary matrix of `cirq.XPowGate(exponent=t, global_shift=s)` is:
    $$
    e^{i \pi t (s + 1/2)}
    \begin{bmatrix}
      \cos(\pi t /2) & -i \sin(\pi t /2) \\
      -i \sin(\pi t /2) & \cos(\pi t /2)
    \end{bmatrix}
    $$

    Note in particular that this gate has a global phase factor of
    $e^{i \pi t / 2}$ vs the traditionally defined rotation matrices
    about the Pauli X axis. See `cirq.Rx` for rotations without the global
    phase. The global phase factor can be adjusted by using the `global_shift`
    parameter when initializing.

    `cirq.X`, the Pauli X gate, is an instance of this gate at `exponent=1`.
    """

    _eigencomponents: Dict[int, List[Tuple[float, np.ndarray]]] = {}

    def __init__(
        self, *, exponent: value.TParamVal = 1.0, global_shift: float = 0.0, dimension: int = 2
    ):
        """Initialize an XPowGate.

        Args:
            exponent: The t in gate**t. Determines how much the eigenvalues of
                the gate are phased by. For example, eigenvectors phased by -1
                when `gate**1` is applied will gain a relative phase of
                e^{i pi exponent} when `gate**exponent` is applied (relative to
                eigenvectors unaffected by `gate**1`).
            global_shift: Offsets the eigenvalues of the gate at exponent=1.
                In effect, this controls a global phase factor on the gate's
                unitary matrix. The factor for global_shift=s is:

                    exp(i * pi * s * t)

                For example, `cirq.X**t` uses a `global_shift` of 0 but
                `cirq.rx(t)` uses a `global_shift` of -0.5, which is why
                `cirq.unitary(cirq.rx(pi))` equals -iX instead of X.
            dimension: Qudit dimension of this gate. For qu*b*its (the default),
                this is set to 2.

        Raises:
            ValueError: If the supplied exponent is a complex number with an
                imaginary component.
        """
        super().__init__(exponent=exponent, global_shift=global_shift)
        self._dimension = dimension

    @property
    def dimension(self) -> value.TParamVal:
        return self._dimension

    def _num_qubits_(self) -> int:
        return 1

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs) -> Optional[np.ndarray]:
        if self._exponent != 1 or self._dimension != 2:
            return NotImplemented
        zero = args.subspace_index(0)
        one = args.subspace_index(1)
        args.available_buffer[zero] = args.target_tensor[one]
        args.available_buffer[one] = args.target_tensor[zero]
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.available_buffer *= p
        return args.available_buffer

    def in_su2(self) -> Rx:
        """Returns an equal-up-global-phase gate from the group SU2."""
        return Rx(rads=self._exponent * _pi(self._exponent))

    def with_canonical_global_phase(self) -> XPowGate:
        """Returns an equal-up-global-phase standardized form of the gate."""
        return XPowGate(exponent=self._exponent, dimension=self._dimension)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self._dimension,)

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        if self._dimension not in XPowGate._eigencomponents:
            components = []
            root = 1j ** (4 / self._dimension)
            for i in range(self._dimension):
                half_turns = i * 2 / self._dimension
                v = np.array([root ** (i * j) / self._dimension for j in range(self._dimension)])
                m = np.array([np.roll(v, j) for j in range(self._dimension)])
                components.append((half_turns, m))
            XPowGate._eigencomponents[self._dimension] = components
        return XPowGate._eigencomponents[self._dimension]

    def _with_exponent(self, exponent: cirq.TParamVal) -> cirq.XPowGate:
        return XPowGate(
            exponent=exponent, global_shift=self._global_shift, dimension=self._dimension
        )

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.clifford_gate import SingleQubitCliffordGate

        if self.exponent % 2 == 0:
            return []
        if self.exponent % 2 == 0.5:
            return SingleQubitCliffordGate.X_sqrt.on(*qubits)
        if self.exponent % 2 == 1:
            return SingleQubitCliffordGate.X.on(*qubits)
        if self.exponent % 2 == 1.5:
            return SingleQubitCliffordGate.X_nsqrt.on(*qubits)
        return NotImplemented  # pragma: no cover

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_() or self._dimension != 2:
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> raw_types.Gate:
        """Returns a controlled `XPowGate`, using a `CXPowGate` where possible.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate`. This method overrides this behavior
        to return a `CXPowGate` or a `ControlledGate` of a `CXPowGate`, when
        this is possible.

        The conditions for the override to occur are:

        * The `global_shift` of the `XPowGate` is 0.
        * The `control_values` and `control_qid_shape` are compatible with
            the `CXPowGate`:
            * The last value of `control_qid_shape` is a qubit.
            * The last value of `control_values` corresponds to the
                control being satisfied if that last qubit is 1 and
                not satisfied if the last qubit is 0.

        If these conditions are met, then the returned object is a `CXPowGate`
        or, in the case that there is more than one controlled qudit, a
        `ControlledGate` with the `Gate` being a `CXPowGate`. In the
        latter case the `ControlledGate` is controlled by one less qudit
        than specified in `control_values` and `control_qid_shape` (since
        one of these, the last qubit, is used as the control for the
        `CXPowGate`).

        If the above conditions are not met, a `ControlledGate` of this
        gate will be returned.

        Args:
            num_controls: Total number of control qubits.
            control_values: Which control computational basis state to apply the
                sub gate.  A sequence of length `num_controls` where each
                entry is an integer (or set of integers) corresponding to the
                computational basis state (or set of possible values) where that
                control is enabled.  When all controls are enabled, the sub gate is
                applied.  If unspecified, control values default to 1.
            control_qid_shape: The qid shape of the controls.  A tuple of the
                expected dimension of each control qid.  Defaults to
                `(2,) * num_controls`.  Specify this argument when using qudits.

        Returns:
            A `cirq.ControlledGate` (or `cirq.CXPowGate` if possible) representing
                `self` controlled by the given control values and qubits.
        """
        if control_values and not isinstance(control_values, cv.AbstractControlValues):
            control_values = cv.ProductOfSums(
                tuple(
                    (val,) if isinstance(val, int) else tuple(sorted(val)) for val in control_values
                )
            )
        result = super().controlled(num_controls, control_values, control_qid_shape)
        if (
            self._global_shift == 0
            and isinstance(result, controlled_gate.ControlledGate)
            and isinstance(result.control_values, cv.ProductOfSums)
            and result.control_values[-1] == (1,)
            and result.control_qid_shape[-1] == 2
        ):
            return cirq.CXPowGate(
                exponent=self._exponent, global_shift=self._global_shift
            ).controlled(
                result.num_controls() - 1, result.control_values[:-1], result.control_qid_shape[:-1]
            )
        return result

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if self._dimension != 2:
            return NotImplemented  # pragma: no cover
        phase = 1j ** (2 * self._exponent * (self._global_shift + 0.5))
        lib = sympy if protocols.is_parameterized(self) else np
        angle = lib.pi * self._exponent / 2
        return value.LinearDict({'I': phase * lib.cos(angle), 'X': -1j * phase * lib.sin(angle)})

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> Union[str, protocols.CircuitDiagramInfo]:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('X',), exponent=self._diagram_exponent(args)
        )

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> Optional[str]:
        args.validate_version('2.0', '3.0')
        if self._global_shift == 0:
            if self._exponent == 1:
                return args.format('x {0};\n', qubits[0])
            elif self._exponent == 0.5:
                return args.format('sx {0};\n', qubits[0])
            elif self._exponent == -0.5:
                return args.format('sxdg {0};\n', qubits[0])
        return args.format('rx({0:half_turns}) {1};\n', self._exponent, qubits[0])

    @property
    def phase_exponent(self):
        return 0.0

    def _phase_by_(self, phase_turns, qubit_index):
        """See `cirq.SupportsPhase`."""
        return _phased_x_or_pauli_gate(exponent=self._exponent, phase_exponent=phase_turns * 2)

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_() or self._dimension != 2:
            return None
        return self.exponent % 0.5 == 0

    def __str__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'X'
            return f'X**{self._exponent}'
        return f'XPowGate(exponent={self._exponent}, global_shift={self._global_shift!r})'

    def __repr__(self) -> str:
        if self._global_shift == 0 and self._dimension == 2:
            if self._exponent == 1:
                return 'cirq.X'
            return f'(cirq.X**{proper_repr(self._exponent)})'
        args = []
        if self._exponent != 1:
            args.append(f'exponent={proper_repr(self._exponent)}')
        if self._global_shift != 0:
            args.append(f'global_shift={self._global_shift}')
        if self._dimension != 2:
            args.append(f'dimension={self._dimension}')
        all_args = ', '.join(args)
        return f'cirq.XPowGate({all_args})'

    def _json_dict_(self) -> Dict[str, Any]:
        d = protocols.obj_to_dict_helper(self, ['exponent', 'global_shift'])
        if self.dimension != 2:
            d['dimension'] = self.dimension
        return d


class Rx(XPowGate):
    r"""A gate with matrix $e^{-i X t/2}$ that rotates around the X axis of the Bloch sphere by $t$.

    The unitary matrix of `cirq.Rx(rads=t)` is:
    $$
    e^{-i X t /2} =
        \begin{bmatrix}
            \cos(t/2) & -i \sin(t/2) \\
            -i \sin(t/2) & \cos(t/2)
        \end{bmatrix}
    $$

    This gate corresponds to the traditionally defined rotation matrices about the Pauli X axis.
    """

    def __init__(self, *, rads: value.TParamVal):
        """Initialize an Rx (`cirq.XPowGate`).

        Args:
            rads: Radians to rotate about the X axis of the Bloch sphere.
        """
        self._rads = rads
        super().__init__(exponent=rads / _pi(rads), global_shift=-0.5)

    def _with_exponent(self, exponent: value.TParamVal) -> Rx:
        return Rx(rads=exponent * _pi(exponent))

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> Union[str, protocols.CircuitDiagramInfo]:
        angle_str = self._format_exponent_as_angle(args)
        return f'Rx({angle_str})'

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'Rx(π)'
        return f'Rx({self._exponent}π)'

    def __repr__(self) -> str:
        return f'cirq.Rx(rads={proper_repr(self._rads)})'

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> Optional[str]:
        args.validate_version('2.0', '3.0')
        return args.format('rx({0:half_turns}) {1};\n', self._exponent, qubits[0])

    def _json_dict_(self) -> Dict[str, Any]:
        return {'rads': self._rads}

    @classmethod
    def _from_json_dict_(cls, rads, **kwargs) -> Rx:
        return cls(rads=rads)


@value.value_equality
class YPowGate(eigen_gate.EigenGate):
    r"""A gate that rotates around the Y axis of the Bloch sphere.

    The unitary matrix of `cirq.YPowGate(exponent=t)` is:
    $$
        \begin{bmatrix}
            e^{i \pi t /2} \cos(\pi t /2) & - e^{i \pi t /2} \sin(\pi t /2) \\
            e^{i \pi t /2} \sin(\pi t /2) & e^{i \pi t /2} \cos(\pi t /2)
        \end{bmatrix}
    $$

    Note in particular that this gate has a global phase factor of
    $e^{i \pi t / 2}$ vs the traditionally defined rotation matrices
    about the Pauli Y axis. See `cirq.Ry` for rotations without the global
    phase. The global phase factor can be adjusted by using the `global_shift`
    parameter when initializing.

    `cirq.Y`, the Pauli Y gate, is an instance of this gate at `exponent=1`.

    Unlike `cirq.XPowGate` and `cirq.ZPowGate`, this gate has no generalization
    to qudits and hence does not take the dimension argument. Ignoring the
    global phase all generalized Pauli operators on a d-level system may be
    written as X**a Z**b for a,b=0,1,...,d-1. For a qubit, there is only one
    "mixed" operator: XZ, conventionally denoted -iY. However, when d > 2 there
    are (d-1)*(d-1) > 1 such "mixed" operators (still ignoring the global phase).
    Due to this ambiguity, qudit Y gate is not well defined. The "mixed" operators
    for qudits are generally not referred to by name, but instead are specified in
    terms of X and Z.
    """

    def _num_qubits_(self) -> int:
        return 1

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs) -> Optional[np.ndarray]:
        if self._exponent != 1:
            return NotImplemented
        zero = args.subspace_index(0)
        one = args.subspace_index(1)
        args.available_buffer[zero] = -1j * args.target_tensor[one]
        args.available_buffer[one] = 1j * args.target_tensor[zero]
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.available_buffer *= p
        return args.available_buffer

    def in_su2(self) -> Ry:
        """Returns an equal-up-global-phase gate from the group SU2."""
        return Ry(rads=self._exponent * _pi(self._exponent))

    def with_canonical_global_phase(self) -> YPowGate:
        """Returns an equal-up-global-phase standardized form of the gate."""
        return YPowGate(exponent=self._exponent)

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.clifford_gate import SingleQubitCliffordGate

        if self.exponent % 2 == 0:
            return []
        if self.exponent % 2 == 0.5:
            return SingleQubitCliffordGate.Y_sqrt.on(*qubits)
        if self.exponent % 2 == 1:
            return SingleQubitCliffordGate.Y.on(*qubits)
        if self.exponent % 2 == 1.5:
            return SingleQubitCliffordGate.Y_nsqrt.on(*qubits)
        return NotImplemented  # pragma: no cover

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [
            (0, np.array([[0.5, -0.5j], [0.5j, 0.5]])),
            (1, np.array([[0.5, 0.5j], [-0.5j, 0.5]])),
        ]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        phase = 1j ** (2 * self._exponent * (self._global_shift + 0.5))
        lib = sympy if protocols.is_parameterized(self) else np
        angle = lib.pi * self._exponent / 2
        return value.LinearDict({'I': phase * lib.cos(angle), 'Y': -1j * phase * lib.sin(angle)})

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> Union[str, protocols.CircuitDiagramInfo]:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('Y',), exponent=self._diagram_exponent(args)
        )

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> Optional[str]:
        args.validate_version('2.0', '3.0')
        if self._exponent == 1 and self.global_shift != -0.5:
            return args.format('y {0};\n', qubits[0])

        return args.format('ry({0:half_turns}) {1};\n', self._exponent, qubits[0])

    @property
    def phase_exponent(self):
        return 0.5

    def _phase_by_(self, phase_turns, qubit_index):
        """See `cirq.SupportsPhase`."""
        return _phased_x_or_pauli_gate(
            exponent=self._exponent, phase_exponent=0.5 + phase_turns * 2
        )

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_():
            return None
        return self.exponent % 0.5 == 0

    def __str__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'Y'
            return f'Y**{self._exponent}'
        return f'YPowGate(exponent={self._exponent}, global_shift={self._global_shift!r})'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.Y'
            return f'(cirq.Y**{proper_repr(self._exponent)})'
        return (
            'cirq.YPowGate('
            f'exponent={proper_repr(self._exponent)}, '
            f'global_shift={self._global_shift!r})'
        )


class Ry(YPowGate):
    r"""A gate with matrix $e^{-i Y t/2}$ that rotates around the Y axis of the Bloch sphere by $t$.

    The unitary matrix of `cirq.Ry(rads=t)` is:
    $$
    e^{-i Y t / 2} =
        \begin{bmatrix}
            \cos(t/2) & -\sin(t/2) \\
            \sin(t/2) & \cos(t/2)
        \end{bmatrix}
    $$

    This gate corresponds to the traditionally defined rotation matrices about the Pauli Y axis.
    """

    def __init__(self, *, rads: value.TParamVal):
        """Initialize an Ry (`cirq.YPowGate`).

        Args:
            rads: Radians to rotate about the Y axis of the Bloch sphere.
        """
        self._rads = rads
        super().__init__(exponent=rads / _pi(rads), global_shift=-0.5)

    def _with_exponent(self, exponent: value.TParamVal) -> Ry:
        return Ry(rads=exponent * _pi(exponent))

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> Union[str, protocols.CircuitDiagramInfo]:
        angle_str = self._format_exponent_as_angle(args)
        return f'Ry({angle_str})'

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'Ry(π)'
        return f'Ry({self._exponent}π)'

    def __repr__(self) -> str:
        return f'cirq.Ry(rads={proper_repr(self._rads)})'

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> Optional[str]:
        args.validate_version('2.0', '3.0')
        return args.format('ry({0:half_turns}) {1};\n', self._exponent, qubits[0])

    def _json_dict_(self) -> Dict[str, Any]:
        return {'rads': self._rads}

    @classmethod
    def _from_json_dict_(cls, rads, **kwargs) -> Ry:
        return cls(rads=rads)


@value.value_equality
class ZPowGate(eigen_gate.EigenGate):
    r"""A gate that rotates around the Z axis of the Bloch sphere.

    The unitary matrix of `cirq.ZPowGate(exponent=t, global_shift=s)` is:
    $$
        e^{i \pi s t}
        \begin{bmatrix}
            1 & 0 \\
            0 & e^{i \pi t}
        \end{bmatrix}
    $$

    Note in particular that this gate has a global phase factor of
    $e^{i\pi t/2}$ vs the traditionally defined rotation matrices
    about the Pauli Z axis. See `cirq.Rz` for rotations without the global
    phase. The global phase factor can be adjusted by using the `global_shift`
    parameter when initializing.

    `cirq.Z`, the Pauli Z gate, is an instance of this gate at `exponent=1`.
    """

    _eigencomponents: Dict[int, List[Tuple[float, np.ndarray]]] = {}

    def __init__(
        self, *, exponent: value.TParamVal = 1.0, global_shift: float = 0.0, dimension: int = 2
    ):
        """Initialize a ZPowGate.

        Args:
            exponent: The t in gate**t. Determines how much the eigenvalues of
                the gate are phased by. For example, eigenvectors phased by -1
                when `gate**1` is applied will gain a relative phase of
                e^{i pi exponent} when `gate**exponent` is applied (relative to
                eigenvectors unaffected by `gate**1`).
            global_shift: Offsets the eigenvalues of the gate at exponent=1.
                In effect, this controls a global phase factor on the gate's
                unitary matrix. The factor for global_shift=s is:

                    exp(i * pi * s * t)

                For example, `cirq.X**t` uses a `global_shift` of 0 but
                `cirq.rx(t)` uses a `global_shift` of -0.5, which is why
                `cirq.unitary(cirq.rx(pi))` equals -iX instead of X.
            dimension: Qudit dimension of this gate. For qu*b*its (the default),
                this is set to 2.

        Raises:
            ValueError: If the supplied exponent is a complex number with an
                imaginary component.
        """
        super().__init__(exponent=exponent, global_shift=global_shift)
        self._dimension = dimension

    @property
    def dimension(self) -> value.TParamVal:
        return self._dimension

    def _num_qubits_(self) -> int:
        return 1

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs) -> Optional[np.ndarray]:
        if protocols.is_parameterized(self):
            return None

        for i in range(1, self._dimension):
            subspace = args.subspace_index(i)
            c = 1j ** (self._exponent * 4 * i / self._dimension)
            args.target_tensor[subspace] *= c
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return args.target_tensor

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.clifford_gate import SingleQubitCliffordGate

        if self.exponent % 2 == 0:
            return []
        if self.exponent % 2 == 0.5:
            return SingleQubitCliffordGate.Z_sqrt.on(*qubits)
        if self.exponent % 2 == 1:
            return SingleQubitCliffordGate.Z.on(*qubits)
        if self.exponent % 2 == 1.5:
            return SingleQubitCliffordGate.Z_nsqrt.on(*qubits)
        return NotImplemented  # pragma: no cover

    def in_su2(self) -> Rz:
        """Returns an equal-up-global-phase gate from the group SU2."""
        return Rz(rads=self._exponent * _pi(self._exponent))

    def with_canonical_global_phase(self) -> ZPowGate:
        """Returns an equal-up-global-phase standardized form of the gate."""
        return ZPowGate(exponent=self._exponent, dimension=self._dimension)

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> raw_types.Gate:
        """Returns a controlled `ZPowGate`, using a `CZPowGate` where possible.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate`. This method overrides this behavior
        to return a `CZPowGate` or a `ControlledGate` of a `CZPowGate`, when
        this is possible.

        The conditions for the override to occur are:

        * The `global_shift` of the `ZPowGate` is 0.
        * The `control_values` and `control_qid_shape` are compatible with
            the `CZPowGate`:
            * The last value of `control_qid_shape` is a qubit.
            * The last value of `control_values` corresponds to the
                control being satisfied if that last qubit is 1 and
                not satisfied if the last qubit is 0.

        If these conditions are met, then the returned object is a `CZPowGate`
        or, in the case that there is more than one controlled qudit, a
        `ControlledGate` with the `Gate` being a `CZPowGate`. In the
        latter case the `ControlledGate` is controlled by one less qudit
        than specified in `control_values` and `control_qid_shape` (since
        one of these, the last qubit, is used as the control for the
        `CZPowGate`).

        If the above conditions are not met, a `ControlledGate` of this
        gate will be returned.

        Args:
            num_controls: Total number of control qubits.
            control_values: Which control computational basis state to apply the
                sub gate.  A sequence of length `num_controls` where each
                entry is an integer (or set of integers) corresponding to the
                computational basis state (or set of possible values) where that
                control is enabled.  When all controls are enabled, the sub gate is
                applied.  If unspecified, control values default to 1.
            control_qid_shape: The qid shape of the controls.  A tuple of the
                expected dimension of each control qid.  Defaults to
                `(2,) * num_controls`.  Specify this argument when using qudits.

        Returns:
            A `cirq.ControlledGate` (or `cirq.CZPowGate` if possible) representing
                `self` controlled by the given control values and qubits.
        """
        if control_values and not isinstance(control_values, cv.AbstractControlValues):
            control_values = cv.ProductOfSums(
                tuple(
                    (val,) if isinstance(val, int) else tuple(sorted(val)) for val in control_values
                )
            )
        result = super().controlled(num_controls, control_values, control_qid_shape)
        if (
            self._global_shift == 0
            and isinstance(result, controlled_gate.ControlledGate)
            and isinstance(result.control_values, cv.ProductOfSums)
            and result.control_values[-1] == (1,)
            and result.control_qid_shape[-1] == 2
        ):
            return cirq.CZPowGate(
                exponent=self._exponent, global_shift=self._global_shift
            ).controlled(
                result.num_controls() - 1, result.control_values[:-1], result.control_qid_shape[:-1]
            )
        return result

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self._dimension,)

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        if self._dimension not in ZPowGate._eigencomponents:
            components = []
            for i in range(self._dimension):
                half_turns = i * 2 / self._dimension
                m = np.zeros((self._dimension, self._dimension))
                m[i][i] = 1
                components.append((half_turns, m))
            ZPowGate._eigencomponents[self._dimension] = components
        return ZPowGate._eigencomponents[self._dimension]

    def _with_exponent(self, exponent: cirq.TParamVal) -> cirq.ZPowGate:
        return ZPowGate(
            exponent=exponent, global_shift=self._global_shift, dimension=self._dimension
        )

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_() or self._dimension != 2:
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if self._dimension != 2:
            return NotImplemented  # pragma: no cover
        phase = 1j ** (2 * self._exponent * (self._global_shift + 0.5))
        lib = sympy if protocols.is_parameterized(self) else np
        angle = lib.pi * self._exponent / 2
        return value.LinearDict({'I': phase * lib.cos(angle), 'Z': -1j * phase * lib.sin(angle)})

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        return self

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_() or self._dimension != 2:
            return None
        return self.exponent % 0.5 == 0

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> Union[str, protocols.CircuitDiagramInfo]:
        e = self._diagram_exponent(args)
        if e in [-0.25, 0.25]:
            return protocols.CircuitDiagramInfo(wire_symbols=('T',), exponent=cast(float, e) * 4)

        if e in [-0.5, 0.5]:
            return protocols.CircuitDiagramInfo(wire_symbols=('S',), exponent=cast(float, e) * 2)

        return protocols.CircuitDiagramInfo(wire_symbols=('Z',), exponent=e)

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> Optional[str]:
        args.validate_version('2.0', '3.0')

        if self.global_shift == 0:
            if self._exponent == 1:
                return args.format('z {0};\n', qubits[0])
            elif self._exponent == 0.5:
                return args.format('s {0};\n', qubits[0])
            elif self._exponent == -0.5:
                return args.format('sdg {0};\n', qubits[0])
            elif self._exponent == 0.25:
                return args.format('t {0};\n', qubits[0])
            elif self._exponent == -0.25:
                return args.format('tdg {0};\n', qubits[0])
        return args.format('rz({0:half_turns}) {1};\n', self._exponent, qubits[0])

    def __str__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 0.25:
                return 'T'
            if self._exponent == -0.25:
                return 'T**-1'
            if self._exponent == 0.5:
                return 'S'
            if self._exponent == -0.5:
                return 'S**-1'
            if self._exponent == 1:
                return 'Z'
            return f'Z**{self._exponent}'
        return f'ZPowGate(exponent={self._exponent}, global_shift={self._global_shift!r})'

    def __repr__(self) -> str:
        if self._global_shift == 0 and self._dimension == 2:
            if self._exponent == 0.25:
                return 'cirq.T'
            if self._exponent == -0.25:
                return '(cirq.T**-1)'
            if self._exponent == 0.5:
                return 'cirq.S'
            if self._exponent == -0.5:
                return '(cirq.S**-1)'
            if self._exponent == 1:
                return 'cirq.Z'
            return f'(cirq.Z**{proper_repr(self._exponent)})'
        args = []
        if self._exponent != 1:
            args.append(f'exponent={proper_repr(self._exponent)}')
        if self._global_shift != 0:
            args.append(f'global_shift={self._global_shift}')
        if self._dimension != 2:
            args.append(f'dimension={self._dimension}')
        all_args = ', '.join(args)
        return f'cirq.ZPowGate({all_args})'

    def _commutes_on_qids_(
        self, qids: Sequence[cirq.Qid], other: Any, *, atol: float = 1e-8
    ) -> Union[bool, NotImplementedType, None]:
        from cirq.ops.parity_gates import ZZPowGate

        if not isinstance(other, raw_types.Operation):
            return NotImplemented
        if not isinstance(other.gate, (ZPowGate, CZPowGate, ZZPowGate)):
            return NotImplemented
        return True

    def _json_dict_(self) -> Dict[str, Any]:
        d = protocols.obj_to_dict_helper(self, ['exponent', 'global_shift'])
        if self.dimension != 2:
            d['dimension'] = self.dimension
        return d


class Rz(ZPowGate):
    r"""A gate with matrix $e^{-i Z t/2}$ that rotates around the Z axis of the Bloch sphere by $t$.

    The unitary matrix of `cirq.Rz(rads=t)` is:
    $$
    e^{-i Z t /2} =
        \begin{bmatrix}
            e^{-it/2} & 0 \\
            0 & e^{it/2}
        \end{bmatrix}
    $$

    This gate corresponds to the traditionally defined rotation matrices about the Pauli Z axis.
    """

    def __init__(self, *, rads: value.TParamVal):
        """Initialize an Rz (`cirq.ZPowGate`).

        Args:
            rads: Radians to rotate about the Z axis of the Bloch sphere.
        """
        self._rads = rads
        super().__init__(exponent=rads / _pi(rads), global_shift=-0.5)

    def _with_exponent(self, exponent: value.TParamVal) -> Rz:
        return Rz(rads=exponent * _pi(exponent))

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> Union[str, protocols.CircuitDiagramInfo]:
        angle_str = self._format_exponent_as_angle(args)
        return f'Rz({angle_str})'

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'Rz(π)'
        return f'Rz({self._exponent}π)'

    def __repr__(self) -> str:
        return f'cirq.Rz(rads={proper_repr(self._rads)})'

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> Optional[str]:
        args.validate_version('2.0', '3.0')
        return args.format('rz({0:half_turns}) {1};\n', self._exponent, qubits[0])

    def _json_dict_(self) -> Dict[str, Any]:
        return {'rads': self._rads}

    @classmethod
    def _from_json_dict_(cls, rads, **kwargs) -> Rz:
        return cls(rads=rads)


class HPowGate(eigen_gate.EigenGate):
    r"""A Gate that performs a rotation around the X+Z axis of the Bloch sphere.

    The unitary matrix of `cirq.HPowGate(exponent=t)` is:
    $$
        \begin{bmatrix}
            e^{i\pi t/2} \left(\cos(\pi t/2) - i \frac{\sin (\pi t /2)}{\sqrt{2}}\right)
                && -i e^{i\pi t/2} \frac{\sin(\pi t /2)}{\sqrt{2}} \\
            -i e^{i\pi t/2} \frac{\sin(\pi t /2)}{\sqrt{2}}
                && e^{i\pi t/2} \left(\cos(\pi t/2) + i \frac{\sin (\pi t /2)}{\sqrt{2}}\right)
        \end{bmatrix}
    $$
    Note in particular that for $t=1$, this gives the Hadamard matrix
    $$
        \begin{bmatrix}
            \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
            \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}
    $$

    `cirq.H`, the Hadamard gate, is an instance of this gate at `exponent=1`.
    """

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        s = np.sqrt(2)

        component0 = np.array([[3 + 2 * s, 1 + s], [1 + s, 1]]) / (4 + 2 * s)

        component1 = np.array([[3 - 2 * s, 1 - s], [1 - s, 1]]) / (4 - 2 * s)

        return [(0, component0), (1, component1)]

    def _num_qubits_(self) -> int:
        return 1

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        phase = 1j ** (2 * self._exponent * (self._global_shift + 0.5))
        angle = np.pi * self._exponent / 2
        return value.LinearDict(
            {
                'I': phase * np.cos(angle),
                'X': -1j * phase * np.sin(angle) / np.sqrt(2),
                'Z': -1j * phase * np.sin(angle) / np.sqrt(2),
            }
        )

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.clifford_gate import SingleQubitCliffordGate

        if self.exponent % 2 == 1:
            return SingleQubitCliffordGate.H.on(*qubits)
        if self.exponent % 2 == 0:
            return []
        return NotImplemented  # pragma: no cover

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs) -> Optional[np.ndarray]:
        if self._exponent != 1:
            return NotImplemented

        zero = args.subspace_index(0)
        one = args.subspace_index(1)
        args.target_tensor[one] -= args.target_tensor[zero]
        args.target_tensor[one] *= -0.5
        args.target_tensor[zero] -= args.target_tensor[one]
        p = 1j ** (2 * self._exponent * self._global_shift)
        args.target_tensor *= np.sqrt(2) * p
        return args.target_tensor

    def _decompose_(self, qubits):
        q = qubits[0]

        if self._exponent == 1:
            yield cirq.Y(q) ** 0.5
            yield cirq.XPowGate(global_shift=-0.25 + self.global_shift).on(q)
            return

        yield YPowGate(exponent=0.25).on(q)
        yield XPowGate(exponent=self._exponent, global_shift=self.global_shift).on(q)
        yield YPowGate(exponent=-0.25).on(q)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('H',), exponent=self._diagram_exponent(args)
        )

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> Optional[str]:
        args.validate_version('2.0', '3.0')
        if self._exponent == 0:
            return args.format('id {0};\n', qubits[0])
        elif self._exponent == 1 and self._global_shift == 0:
            return args.format('h {0};\n', qubits[0])

        return args.format(
            'ry({0:half_turns}) {3};\nrx({1:half_turns}) {3};\nry({2:half_turns}) {3};\n',
            0.25,
            self._exponent,
            -0.25,
            qubits[0],
        )

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_():
            return None
        return self.exponent % 1 == 0

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'H'
        return f'H**{self._exponent}'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.H'
            return f'(cirq.H**{proper_repr(self._exponent)})'
        return (
            f'cirq.HPowGate(exponent={proper_repr(self._exponent)}, '
            f'global_shift={self._global_shift!r})'
        )


class CZPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    r"""A gate that applies a phase to the |11⟩ state of two qubits.

    The unitary matrix of `CZPowGate(exponent=t)` is:

    $$
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & e^{i \pi t} \\
    \end{bmatrix}
    $$

    `cirq.CZ`, the controlled Z gate, is an instance of this gate at
    `exponent=1`.
    """

    def _num_qubits_(self) -> int:
        return 2

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.pauli_interaction_gate import PauliInteractionGate

        if self.exponent % 2 == 1:
            return PauliInteractionGate.CZ.on(*qubits)
        if self.exponent % 2 == 0:
            return []
        return NotImplemented  # pragma: no cover

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.diag([1, 1, 1, 0])), (1, np.diag([0, 0, 0, 1]))]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _apply_unitary_(
        self, args: protocols.ApplyUnitaryArgs
    ) -> Union[np.ndarray, NotImplementedType]:
        if protocols.is_parameterized(self):
            return NotImplemented

        c = 1j ** (2 * self._exponent)
        one_one = args.subspace_index(0b11)
        args.target_tensor[one_one] *= c
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return args.target_tensor

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        z_phase = 1j**self._exponent
        c = -1j * z_phase * np.sin(np.pi * self._exponent / 2) / 2
        return value.LinearDict(
            {
                'II': global_phase * (1 - c),
                'IZ': global_phase * c,
                'ZI': global_phase * c,
                'ZZ': global_phase * -c,
            }
        )

    def _phase_by_(self, phase_turns, qubit_index):
        return self

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> raw_types.Gate:
        """Returns a controlled `CZPowGate`, using a `CCZPowGate` where possible.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate`. This method overrides this behavior
        to return a `CCZPowGate` or a `ControlledGate` of a `CCZPowGate`, when
        this is possible.

        The conditions for the override to occur are:

        * The `global_shift` of the `CZPowGate` is 0.
        * The `control_values` and `control_qid_shape` are compatible with
            the `CCZPowGate`:
            * The last value of `control_qid_shape` is a qubit.
            * The last value of `control_values` corresponds to the
                control being satisfied if that last qubit is 1 and
                not satisfied if the last qubit is 0.

        If these conditions are met, then the returned object is a `CCZPowGate`
        or, in the case that there is more than one controlled qudit, a
        `ControlledGate` with the `Gate` being a `CCZPowGate`. In the
        latter case the `ControlledGate` is controlled by one less qudit
        than specified in `control_values` and `control_qid_shape` (since
        one of these, the last qubit, is used as the control for the
        `CCZPowGate`).

        If the above conditions are not met, a `ControlledGate` of this
        gate will be returned.

        Args:
            num_controls: Total number of control qubits.
            control_values: Which control computational basis state to apply the
                sub gate.  A sequence of length `num_controls` where each
                entry is an integer (or set of integers) corresponding to the
                computational basis state (or set of possible values) where that
                control is enabled.  When all controls are enabled, the sub gate is
                applied.  If unspecified, control values default to 1.
            control_qid_shape: The qid shape of the controls.  A tuple of the
                expected dimension of each control qid.  Defaults to
                `(2,) * num_controls`.  Specify this argument when using qudits.

        Returns:
            A `cirq.ControlledGate` (or `cirq.CCZPowGate` if possible) representing
                `self` controlled by the given control values and qubits.
        """
        if control_values and not isinstance(control_values, cv.AbstractControlValues):
            control_values = cv.ProductOfSums(
                tuple(
                    (val,) if isinstance(val, int) else tuple(sorted(val)) for val in control_values
                )
            )
        result = super().controlled(num_controls, control_values, control_qid_shape)
        if (
            self._global_shift == 0
            and isinstance(result, controlled_gate.ControlledGate)
            and isinstance(result.control_values, cv.ProductOfSums)
            and result.control_values[-1] == (1,)
            and result.control_qid_shape[-1] == 2
        ):
            return cirq.CCZPowGate(
                exponent=self._exponent, global_shift=self._global_shift
            ).controlled(
                result.num_controls() - 1, result.control_values[:-1], result.control_qid_shape[:-1]
            )
        return result

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('@', '@'), exponent=self._diagram_exponent(args)
        )

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> Optional[str]:
        if self._exponent != 1:
            return None  # Don't have an equivalent gate in QASM
        args.validate_version('2.0', '3.0')
        return args.format('cz {0},{1};\n', qubits[0], qubits[1])

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_():
            return None
        return self.exponent % 1 == 0

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'CZ'
        return f'CZ**{self._exponent!r}'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.CZ'
            return f'(cirq.CZ**{proper_repr(self._exponent)})'
        return (
            'cirq.CZPowGate('
            f'exponent={proper_repr(self._exponent)}, '
            f'global_shift={self._global_shift!r})'
        )


class CXPowGate(eigen_gate.EigenGate):
    r"""A gate that applies a controlled power of an X gate.

    When applying CNOT (controlled-not) to qubits, you can either use
    positional arguments CNOT(q1, q2), where q2 is toggled when q1 is on,
    or named arguments CNOT(control=q1, target=q2).
    (Mixing the two is not permitted.)

    The unitary matrix of `cirq.CXPowGate(exponent=t)` is:

    $$
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & g c & -i g s \\
        0 & 0 & -i g s & g c
    \end{bmatrix}
    $$

    where:

    $$
    c = \cos\left(\frac{\pi t}{2}\right)
    $$
    $$
    s = \sin\left(\frac{\pi t}{2}\right)
    $$
    $$
    g = e^{\frac{i \pi t}{2}}
    $$

    `cirq.CNOT`, the controlled NOT gate, is an instance of this gate at
    `exponent=1`.
    """

    def _num_qubits_(self) -> int:
        return 2

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.pauli_interaction_gate import PauliInteractionGate

        if self.exponent % 2 == 1:
            return PauliInteractionGate.CNOT.on(*qubits)
        if self.exponent % 2 == 0:
            return []
        return NotImplemented  # pragma: no cover

    def _decompose_(self, qubits):
        c, t = qubits
        yield YPowGate(exponent=-0.5).on(t)
        yield cirq.CZPowGate(exponent=self._exponent, global_shift=self.global_shift).on(c, t)
        yield YPowGate(exponent=0.5).on(t)

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [
            (0, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]])),
            (1, np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.5, -0.5], [0, 0, -0.5, 0.5]])),
        ]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('@', 'X'), exponent=self._diagram_exponent(args), exponent_qubit_index=1
        )

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs) -> Optional[np.ndarray]:
        if self._exponent != 1:
            return NotImplemented

        oo = args.subspace_index(0b11)
        zo = args.subspace_index(0b01)
        args.available_buffer[oo] = args.target_tensor[oo]
        args.target_tensor[oo] = args.target_tensor[zo]
        args.target_tensor[zo] = args.available_buffer[oo]
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return args.target_tensor

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        cnot_phase = 1j**self._exponent
        c = -1j * cnot_phase * np.sin(np.pi * self._exponent / 2) / 2
        return value.LinearDict(
            {
                'II': global_phase * (1 - c),
                'IX': global_phase * c,
                'ZI': global_phase * c,
                'ZX': global_phase * -c,
            }
        )

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> raw_types.Gate:
        """Returns a controlled `CXPowGate`, using a `CCXPowGate` where possible.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate`. This method overrides this behavior
        to return a `CCXPowGate` or a `ControlledGate` of a `CCXPowGate`, when
        this is possible.

        The conditions for the override to occur are:

        * The `global_shift` of the `CXPowGate` is 0.
        * The `control_values` and `control_qid_shape` are compatible with
            the `CCXPowGate`:
            * The last value of `control_qid_shape` is a qubit.
            * The last value of `control_values` corresponds to the
                control being satisfied if that last qubit is 1 and
                not satisfied if the last qubit is 0.

        If these conditions are met, then the returned object is a `CCXPowGate`
        or, in the case that there is more than one controlled qudit, a
        `ControlledGate` with the `Gate` being a `CCXPowGate`. In the
        latter case the `ControlledGate` is controlled by one less qudit
        than specified in `control_values` and `control_qid_shape` (since
        one of these, the last qubit, is used as the control for the
        `CCXPowGate`).

        If the above conditions are not met, a `ControlledGate` of this
        gate will be returned.

        Args:
            num_controls: Total number of control qubits.
            control_values: Which control computational basis state to apply the
                sub gate.  A sequence of length `num_controls` where each
                entry is an integer (or set of integers) corresponding to the
                computational basis state (or set of possible values) where that
                control is enabled.  When all controls are enabled, the sub gate is
                applied.  If unspecified, control values default to 1.
            control_qid_shape: The qid shape of the controls.  A tuple of the
                expected dimension of each control qid.  Defaults to
                `(2,) * num_controls`.  Specify this argument when using qudits.

        Returns:
            A `cirq.ControlledGate` (or `cirq.CCXPowGate` if possible) representing
                `self` controlled by the given control values and qubits.
        """
        if control_values and not isinstance(control_values, cv.AbstractControlValues):
            control_values = cv.ProductOfSums(
                tuple(
                    (val,) if isinstance(val, int) else tuple(sorted(val)) for val in control_values
                )
            )
        result = super().controlled(num_controls, control_values, control_qid_shape)
        if (
            self._global_shift == 0
            and isinstance(result, controlled_gate.ControlledGate)
            and isinstance(result.control_values, cv.ProductOfSums)
            and result.control_values[-1] == (1,)
            and result.control_qid_shape[-1] == 2
        ):
            return cirq.CCXPowGate(
                exponent=self._exponent, global_shift=self._global_shift
            ).controlled(
                result.num_controls() - 1, result.control_values[:-1], result.control_qid_shape[:-1]
            )
        return result

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> Optional[str]:
        if self._exponent != 1:
            return None  # Don't have an equivalent gate in QASM
        args.validate_version('2.0', '3.0')
        return args.format('cx {0},{1};\n', qubits[0], qubits[1])

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_():
            return None
        return self.exponent % 1 == 0

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'CNOT'
        return f'CNOT**{self._exponent!r}'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.CNOT'
            return f'(cirq.CNOT**{proper_repr(self._exponent)})'
        return (
            f'cirq.CXPowGate(exponent={proper_repr(self._exponent)}, '
            f'global_shift={self._global_shift!r})'
        )


def rx(rads: value.TParamVal) -> Rx:
    """Returns a gate with the matrix $e^{-i X t / 2}$ where $t=rads$."""
    return Rx(rads=rads)


def ry(rads: value.TParamVal) -> Ry:
    """Returns a gate with the matrix $e^{-i Y t / 2}$ where $t=rads$."""
    return Ry(rads=rads)


def rz(rads: value.TParamVal) -> Rz:
    """Returns a gate with the matrix $e^{-i Z t / 2}$ where $t=rads$."""
    return Rz(rads=rads)


def cphase(rads: value.TParamVal) -> CZPowGate:
    r"""Returns a cphase gate with phase of `rad` radians.

    Returns a gate with the unitary matrix:

    $$
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & e^{i rads} \\
    \end{bmatrix}
    $$

    """
    return CZPowGate(exponent=rads / _pi(rads))


H = HPowGate()
document(
    H,
    r"""The Hadamard gate.

    The `exponent=1` instance of `cirq.HPowGate`.

    The unitary matrix of `cirq.H` is:
    $$
    \begin{bmatrix}
        \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
    \end{bmatrix}
    $$
    """,
)

S = ZPowGate(exponent=0.5)
document(
    S,
    r"""The Clifford S gate.

    The `exponent=0.5` instance of `cirq.ZPowGate`.

    The unitary matrix of `cirq.S` is:
    $$
    \begin{bmatrix}
        1 & 0 \\
        0 & i
    \end{bmatrix}
    $$
    """,
)

T = ZPowGate(exponent=0.25)
document(
    T,
    r"""The non-Clifford T gate.

    The `exponent=0.25` instance of `cirq.ZPowGate`.

    The unitary matrix of `cirq.T` is
    $$
    \begin{bmatrix}
        1 & 0 \\
        0 & e^{i \pi /4}
    \end{bmatrix}
    $$
    """,
)

CZ = CZPowGate()
document(
    CZ,
    r"""The controlled Z gate.

    This is the `exponent=1` instance of `cirq.CZPowGate`.

    The unitary matrix of this gate is (empty elements are $0$):
    $$
        \begin{bmatrix}
            1 & & & \\
            & 1 & & \\
            & & 1 & \\
            & & & -1
        \end{bmatrix}
    $$
    """,
)

CNotPowGate = CXPowGate
CNOT = CX = CNotPowGate()
document(
    CNOT,
    r"""The controlled NOT gate.

    This is the `exponent=1` instance of `cirq.CXPowGate`.

    Alternative name: `cirq.CNOT`.

    The unitary matrix of this gate is (empty elements are $0$):
    $$
        \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0
        \end{bmatrix}
    $$
    """,
)


def _phased_x_or_pauli_gate(
    exponent: Union[float, sympy.Expr], phase_exponent: Union[float, sympy.Expr]
) -> Union[cirq.PhasedXPowGate, cirq.XPowGate, cirq.YPowGate]:
    """Return PhasedXPowGate or X or Y gate if equivalent at the given phase_exponent."""
    if not isinstance(phase_exponent, sympy.Expr) or phase_exponent.is_constant():
        half_turns = value.canonicalize_half_turns(float(phase_exponent))
        match half_turns:
            case 0.0:
                return XPowGate(exponent=exponent)
            case 0.5:
                return YPowGate(exponent=exponent)
    return cirq.ops.PhasedXPowGate(exponent=exponent, phase_exponent=phase_exponent)
