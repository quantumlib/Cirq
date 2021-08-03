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

from typing import (
    Any,
    cast,
    Collection,
    List,
    Dict,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import numpy as np
import sympy

import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import controlled_gate, eigen_gate, gate_features, raw_types

from cirq.type_workarounds import NotImplementedType

from cirq.ops.swap_gates import ISWAP, SWAP, ISwapPowGate, SwapPowGate
from cirq.ops.measurement_gate import MeasurementGate

if TYPE_CHECKING:
    import cirq

assert all(
    [ISWAP, SWAP, ISwapPowGate, SwapPowGate, MeasurementGate]
), """
Included for compatibility. Please continue to use top-level cirq.{thing}
imports.
"""


def _act_with_gates(args, qubits, *gates: 'cirq.SupportsActOnQubits') -> None:
    """Act on the given args with the given gates in order."""
    for gate in gates:
        assert gate._act_on_(args, qubits)


def _pi(rads):
    return sympy.pi if protocols.is_parameterized(rads) else np.pi


@value.value_equality
class XPowGate(eigen_gate.EigenGate, gate_features.SingleQubitGate):
    """A gate that rotates around the X axis of the Bloch sphere.

    The unitary matrix of ``XPowGate(exponent=t)`` is:

        [[g·c, -i·g·s],
         [-i·g·s, g·c]]

    where:

        c = cos(π·t/2)
        s = sin(π·t/2)
        g = exp(i·π·t/2).

    Note in particular that this gate has a global phase factor of
    e^{i·π·t/2} vs the traditionally defined rotation matrices
    about the Pauli X axis. See `cirq.rx` for rotations without the global
    phase. The global phase factor can be adjusted by using the `global_shift`
    parameter when initializing.

    `cirq.X`, the Pauli X gate, is an instance of this gate at exponent=1.
    """

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if self._exponent != 1:
            return NotImplemented
        zero = args.subspace_index(0)
        one = args.subspace_index(1)
        args.available_buffer[zero] = args.target_tensor[one]
        args.available_buffer[one] = args.target_tensor[zero]
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.available_buffer *= p
        return args.available_buffer

    def _act_on_(self, args: 'cirq.ActOnArgs', qubits: Sequence['cirq.Qid']):
        from cirq.sim import clifford

        if isinstance(args, clifford.ActOnCliffordTableauArgs):
            if not protocols.has_stabilizer_effect(self):
                return NotImplemented
            tableau = args.tableau
            q = args.qubit_map[qubits[0]]
            effective_exponent = self._exponent % 2
            if effective_exponent == 0.5:
                tableau.xs[:, q] ^= tableau.zs[:, q]
                tableau.rs[:] ^= tableau.xs[:, q] & tableau.zs[:, q]
            elif effective_exponent == 1:
                tableau.rs[:] ^= tableau.zs[:, q]
            elif effective_exponent == 1.5:
                tableau.rs[:] ^= tableau.xs[:, q] & tableau.zs[:, q]
                tableau.xs[:, q] ^= tableau.zs[:, q]
            return True

        if isinstance(args, clifford.ActOnStabilizerCHFormArgs):
            if not protocols.has_stabilizer_effect(self):
                return NotImplemented
            _act_with_gates(args, qubits, H, ZPowGate(exponent=self._exponent), H)
            # Adjust the global phase based on the global_shift parameter.
            args.state.omega *= np.exp(1j * np.pi * self.global_shift * self.exponent)
            return True

        return NotImplemented

    def in_su2(self) -> 'Rx':
        """Returns an equal-up-global-phase gate from the group SU2."""
        return Rx(rads=self._exponent * _pi(self._exponent))

    def with_canonical_global_phase(self) -> 'XPowGate':
        """Returns an equal-up-global-phase standardized form of the gate."""
        return XPowGate(exponent=self._exponent)

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [
            (0, np.array([[0.5, 0.5], [0.5, 0.5]])),
            (1, np.array([[0.5, -0.5], [-0.5, 0.5]])),
        ]

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
        return NotImplemented

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def controlled(
        self,
        num_controls: int = None,
        control_values: Optional[Sequence[Union[int, Collection[int]]]] = None,
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
        """
        result = super().controlled(num_controls, control_values, control_qid_shape)
        if (
            self._global_shift == 0
            and isinstance(result, controlled_gate.ControlledGate)
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
        if protocols.is_parameterized(self):
            return NotImplemented
        phase = 1j ** (2 * self._exponent * (self._global_shift + 0.5))
        angle = np.pi * self._exponent / 2
        return value.LinearDict(
            {
                'I': phase * np.cos(angle),
                'X': -1j * phase * np.sin(angle),
            }
        )

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('X',), exponent=self._diagram_exponent(args)
        )

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        if self._exponent == 1 and self._global_shift != -0.5:
            return args.format('x {0};\n', qubits[0])
        elif self._exponent == 0.5:
            return args.format('sx {0};\n', qubits[0])
        return args.format('rx({0:half_turns}) {1};\n', self._exponent, qubits[0])

    def _quil_(
        self, qubits: Tuple['cirq.Qid', ...], formatter: 'cirq.QuilFormatter'
    ) -> Optional[str]:
        if self._exponent == 1 and self._global_shift != -0.5:
            return formatter.format('X {0}\n', qubits[0])
        return formatter.format('RX({0}) {1}\n', self._exponent * np.pi, qubits[0])

    @property
    def phase_exponent(self):
        return 0.0

    def _phase_by_(self, phase_turns, qubit_index):
        """See `cirq.SupportsPhase`."""
        return cirq.ops.phased_x_gate.PhasedXPowGate(
            exponent=self._exponent, phase_exponent=phase_turns * 2
        )

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_():
            return None
        return self.exponent % 0.5 == 0

    def __str__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'X'
            return f'X**{self._exponent}'
        return f'XPowGate(exponent={self._exponent}, global_shift={self._global_shift!r})'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.X'
            return f'(cirq.X**{proper_repr(self._exponent)})'
        return 'cirq.XPowGate(exponent={}, global_shift={!r})'.format(
            proper_repr(self._exponent), self._global_shift
        )


class Rx(XPowGate):
    """A gate, with matrix e^{-i X rads/2}, that rotates around the X axis of the Bloch sphere.

    The unitary matrix of ``Rx(rads=t)`` is:

    exp(-i X t/2) =  [ cos(t/2)  -isin(t/2)]
                     [-isin(t/2)  cos(t/2) ]

    The gate corresponds to the traditionally defined rotation matrices about the Pauli X axis.
    """

    def __init__(self, *, rads: value.TParamVal):
        self._rads = rads
        super().__init__(exponent=rads / _pi(rads), global_shift=-0.5)

    def _with_exponent(self: 'Rx', exponent: value.TParamVal) -> 'Rx':
        return Rx(rads=exponent * _pi(exponent))

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        angle_str = self._format_exponent_as_angle(args)
        return f'Rx({angle_str})'

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'Rx(π)'
        return f'Rx({self._exponent}π)'

    def __repr__(self) -> str:
        return f'cirq.Rx(rads={proper_repr(self._rads)})'

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'rads': self._rads,
        }

    @classmethod
    def _from_json_dict_(cls, rads, **kwargs) -> 'Rx':
        return cls(rads=rads)


@value.value_equality
class YPowGate(eigen_gate.EigenGate, gate_features.SingleQubitGate):
    """A gate that rotates around the Y axis of the Bloch sphere.

    The unitary matrix of ``YPowGate(exponent=t)`` is:

        [[g·c, -g·s],
         [g·s, g·c]]

    where:

        c = cos(π·t/2)
        s = sin(π·t/2)
        g = exp(i·π·t/2).

    Note in particular that this gate has a global phase factor of
    e^{i·π·t/2} vs the traditionally defined rotation matrices
    about the Pauli Y axis. See `cirq.Ry` for rotations without the global
    phase. The global phase factor can be adjusted by using the `global_shift`
    parameter when initializing.

    `cirq.Y`, the Pauli Y gate, is an instance of this gate at exponent=1.
    """

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
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

    def _act_on_(self, args: 'cirq.ActOnArgs', qubits: Sequence['cirq.Qid']):
        from cirq.sim import clifford

        if isinstance(args, clifford.ActOnCliffordTableauArgs):
            if not protocols.has_stabilizer_effect(self):
                return NotImplemented
            tableau = args.tableau
            q = args.qubit_map[qubits[0]]
            effective_exponent = self._exponent % 2
            if effective_exponent == 0.5:
                tableau.rs[:] ^= tableau.xs[:, q] & (~tableau.zs[:, q])
                (tableau.xs[:, q], tableau.zs[:, q]) = (
                    tableau.zs[:, q].copy(),
                    tableau.xs[:, q].copy(),
                )
            elif effective_exponent == 1:
                tableau.rs[:] ^= tableau.xs[:, q] ^ tableau.zs[:, q]
            elif effective_exponent == 1.5:
                tableau.rs[:] ^= ~(tableau.xs[:, q]) & tableau.zs[:, q]
                (tableau.xs[:, q], tableau.zs[:, q]) = (
                    tableau.zs[:, q].copy(),
                    tableau.xs[:, q].copy(),
                )
            return True

        if isinstance(args, clifford.ActOnStabilizerCHFormArgs):
            if not protocols.has_stabilizer_effect(self):
                return NotImplemented
            effective_exponent = self._exponent % 2
            state = args.state
            Z = ZPowGate()
            if effective_exponent == 0.5:
                _act_with_gates(args, qubits, Z, H)
                state.omega *= (1 + 1j) / (2 ** 0.5)
            elif effective_exponent == 1:
                _act_with_gates(args, qubits, Z, H, Z, H)
                state.omega *= 1j
            elif effective_exponent == 1.5:
                _act_with_gates(args, qubits, H, Z)
                state.omega *= (1 - 1j) / (2 ** 0.5)
            # Adjust the global phase based on the global_shift parameter.
            args.state.omega *= np.exp(1j * np.pi * self.global_shift * self.exponent)
            return True
        return NotImplemented

    def in_su2(self) -> 'Ry':
        """Returns an equal-up-global-phase gate from the group SU2."""
        return Ry(rads=self._exponent * _pi(self._exponent))

    def with_canonical_global_phase(self) -> 'YPowGate':
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
        return NotImplemented

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
        if protocols.is_parameterized(self):
            return NotImplemented
        phase = 1j ** (2 * self._exponent * (self._global_shift + 0.5))
        angle = np.pi * self._exponent / 2
        return value.LinearDict(
            {
                'I': phase * np.cos(angle),
                'Y': -1j * phase * np.sin(angle),
            }
        )

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('Y',), exponent=self._diagram_exponent(args)
        )

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        if self._exponent == 1 and self.global_shift != -0.5:
            return args.format('y {0};\n', qubits[0])

        return args.format('ry({0:half_turns}) {1};\n', self._exponent, qubits[0])

    def _quil_(
        self, qubits: Tuple['cirq.Qid', ...], formatter: 'cirq.QuilFormatter'
    ) -> Optional[str]:
        if self._exponent == 1 and self.global_shift != -0.5:
            return formatter.format('Y {0}\n', qubits[0])
        return formatter.format('RY({0}) {1}\n', self._exponent * np.pi, qubits[0])

    @property
    def phase_exponent(self):
        return 0.5

    def _phase_by_(self, phase_turns, qubit_index):
        """See `cirq.SupportsPhase`."""
        return cirq.ops.phased_x_gate.PhasedXPowGate(
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
        return 'cirq.YPowGate(exponent={}, global_shift={!r})'.format(
            proper_repr(self._exponent), self._global_shift
        )


class Ry(YPowGate):
    """A gate, with matrix e^{-i Y rads/2}, that rotates around the Y axis of the Bloch sphere.

    The unitary matrix of ``Ry(rads=t)`` is:

    exp(-i Y t/2) =  [cos(t/2)  -sin(t/2)]
                     [sin(t/2)  cos(t/2) ]

    The gate corresponds to the traditionally defined rotation matrices about the Pauli Y axis.
    """

    def __init__(self, *, rads: value.TParamVal):
        self._rads = rads
        super().__init__(exponent=rads / _pi(rads), global_shift=-0.5)

    def _with_exponent(self: 'Ry', exponent: value.TParamVal) -> 'Ry':
        return Ry(rads=exponent * _pi(exponent))

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        angle_str = self._format_exponent_as_angle(args)
        return f'Ry({angle_str})'

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'Ry(π)'
        return f'Ry({self._exponent}π)'

    def __repr__(self) -> str:
        return f'cirq.Ry(rads={proper_repr(self._rads)})'

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'rads': self._rads,
        }

    @classmethod
    def _from_json_dict_(cls, rads, **kwargs) -> 'Ry':
        return cls(rads=rads)


@value.value_equality
class ZPowGate(eigen_gate.EigenGate, gate_features.SingleQubitGate):
    """A gate that rotates around the Z axis of the Bloch sphere.

    The unitary matrix of ``ZPowGate(exponent=t)`` is:

        [[1, 0],
         [0, g]]

    where:

        g = exp(i·π·t).

    Note in particular that this gate has a global phase factor of
    e^{i·π·t/2} vs the traditionally defined rotation matrices
    about the Pauli Z axis. See `cirq.Rz` for rotations without the global
    phase. The global phase factor can be adjusted by using the `global_shift`
    parameter when initializing.

    `cirq.Z`, the Pauli Z gate, is an instance of this gate at exponent=1.
    """

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if protocols.is_parameterized(self):
            return None

        one = args.subspace_index(1)
        c = 1j ** (self._exponent * 2)
        args.target_tensor[one] *= c
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return args.target_tensor

    def _act_on_(self, args: 'cirq.ActOnArgs', qubits: Sequence['cirq.Qid']):
        from cirq.sim import clifford

        if isinstance(args, clifford.ActOnCliffordTableauArgs):
            if not protocols.has_stabilizer_effect(self):
                return NotImplemented
            tableau = args.tableau
            q = args.qubit_map[qubits[0]]
            effective_exponent = self._exponent % 2
            if effective_exponent == 0.5:
                tableau.rs[:] ^= tableau.xs[:, q] & tableau.zs[:, q]
                tableau.zs[:, q] ^= tableau.xs[:, q]
            elif effective_exponent == 1:
                tableau.rs[:] ^= tableau.xs[:, q]
            elif effective_exponent == 1.5:
                tableau.rs[:] ^= tableau.xs[:, q] & (~tableau.zs[:, q])
                tableau.zs[:, q] ^= tableau.xs[:, q]
            return True

        if isinstance(args, clifford.ActOnStabilizerCHFormArgs):
            if not protocols.has_stabilizer_effect(self):
                return NotImplemented
            q = args.qubit_map[qubits[0]]
            effective_exponent = self._exponent % 2
            state = args.state
            for _ in range(int(effective_exponent * 2)):
                # Prescription for S left multiplication.
                # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
                state.M[q, :] ^= state.G[q, :]
                state.gamma[q] = (state.gamma[q] - 1) % 4
            # Adjust the global phase based on the global_shift parameter.
            args.state.omega *= np.exp(1j * np.pi * self.global_shift * self.exponent)
            return True

        return NotImplemented

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
        return NotImplemented

    def in_su2(self) -> 'Rz':
        """Returns an equal-up-global-phase gate from the group SU2."""
        return Rz(rads=self._exponent * _pi(self._exponent))

    def with_canonical_global_phase(self) -> 'ZPowGate':
        """Returns an equal-up-global-phase standardized form of the gate."""
        return ZPowGate(exponent=self._exponent)

    def controlled(
        self,
        num_controls: int = None,
        control_values: Optional[Sequence[Union[int, Collection[int]]]] = None,
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
        """
        result = super().controlled(num_controls, control_values, control_qid_shape)
        if (
            self._global_shift == 0
            and isinstance(result, controlled_gate.ControlledGate)
            and result.control_values[-1] == (1,)
            and result.control_qid_shape[-1] == 2
        ):
            return cirq.CZPowGate(
                exponent=self._exponent, global_shift=self._global_shift
            ).controlled(
                result.num_controls() - 1, result.control_values[:-1], result.control_qid_shape[:-1]
            )
        return result

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [
            (0, np.diag([1, 0])),
            (1, np.diag([0, 1])),
        ]

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
                'Z': -1j * phase * np.sin(angle),
            }
        )

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        return self

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_():
            return None
        return self.exponent % 0.5 == 0

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        e = self._diagram_exponent(args)
        if e in [-0.25, 0.25]:
            return protocols.CircuitDiagramInfo(wire_symbols=('T',), exponent=cast(float, e) * 4)

        if e in [-0.5, 0.5]:
            return protocols.CircuitDiagramInfo(wire_symbols=('S',), exponent=cast(float, e) * 2)

        return protocols.CircuitDiagramInfo(wire_symbols=('Z',), exponent=e)

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        if self._exponent == 1 and self.global_shift != -0.5:
            return args.format('z {0};\n', qubits[0])
        elif self._exponent == 0.5:
            return args.format('s {0};\n', qubits[0])
        elif self._exponent == -0.5:
            return args.format('sdg {0};\n', qubits[0])

        return args.format('rz({0:half_turns}) {1};\n', self._exponent, qubits[0])

    def _quil_(
        self, qubits: Tuple['cirq.Qid', ...], formatter: 'cirq.QuilFormatter'
    ) -> Optional[str]:
        if self._exponent == 1 and self.global_shift != -0.5:
            return formatter.format('Z {0}\n', qubits[0])
        return formatter.format('RZ({0}) {1}\n', self._exponent * np.pi, qubits[0])

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
        if self._global_shift == 0:
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
        return 'cirq.ZPowGate(exponent={}, global_shift={!r})'.format(
            proper_repr(self._exponent), self._global_shift
        )

    def _commutes_on_qids_(
        self, qids: 'Sequence[cirq.Qid]', other: Any, atol: float
    ) -> Union[bool, NotImplementedType, None]:
        from cirq.ops.parity_gates import ZZPowGate

        if not isinstance(other, raw_types.Operation):
            return NotImplemented
        if not isinstance(other.gate, (ZPowGate, CZPowGate, ZZPowGate)):
            return NotImplemented
        return True


class Rz(ZPowGate):
    """A gate, with matrix e^{-i Z rads/2}, that rotates around the Z axis of the Bloch sphere.

    The unitary matrix of ``Rz(rads=t)`` is:

    exp(-i Z t/2) =  [ e^(-it/2)     0   ]
                     [    0      e^(it/2)]

    The gate corresponds to the traditionally defined rotation matrices about the Pauli Z axis.
    """

    def __init__(self, *, rads: value.TParamVal):
        self._rads = rads
        super().__init__(exponent=rads / _pi(rads), global_shift=-0.5)

    def _with_exponent(self: 'Rz', exponent: value.TParamVal) -> 'Rz':
        return Rz(rads=exponent * _pi(exponent))

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        angle_str = self._format_exponent_as_angle(args)
        return f'Rz({angle_str})'

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'Rz(π)'
        return f'Rz({self._exponent}π)'

    def __repr__(self) -> str:
        return f'cirq.Rz(rads={proper_repr(self._rads)})'

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'rads': self._rads,
        }

    @classmethod
    def _from_json_dict_(cls, rads, **kwargs) -> 'Rz':
        return cls(rads=rads)


class HPowGate(eigen_gate.EigenGate, gate_features.SingleQubitGate):
    """A Gate that performs a rotation around the X+Z axis of the Bloch sphere.

    The unitary matrix of ``HPowGate(exponent=t)`` is:

        [[g·(c-i·s/sqrt(2)), -i·g·s/sqrt(2)],
        [-i·g·s/sqrt(2)], g·(c+i·s/sqrt(2))]]

    where

        c = cos(π·t/2)
        s = sin(π·t/2)
        g = exp(i·π·t/2).

    Note in particular that for `t=1`, this gives the Hadamard matrix.

    `cirq.H`, the Hadamard gate, is an instance of this gate at `exponent=1`.
    """

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        s = np.sqrt(2)

        component0 = np.array([[3 + 2 * s, 1 + s], [1 + s, 1]]) / (4 + 2 * s)

        component1 = np.array([[3 - 2 * s, 1 - s], [1 - s, 1]]) / (4 - 2 * s)

        return [(0, component0), (1, component1)]

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
        return NotImplemented

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
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

    def _act_on_(self, args: 'cirq.ActOnArgs', qubits: Sequence['cirq.Qid']):
        from cirq.sim import clifford

        if isinstance(args, clifford.ActOnCliffordTableauArgs):
            if not protocols.has_stabilizer_effect(self):
                return NotImplemented
            tableau = args.tableau
            q = args.qubit_map[qubits[0]]
            if self._exponent % 2 == 1:
                (tableau.xs[:, q], tableau.zs[:, q]) = (
                    tableau.zs[:, q].copy(),
                    tableau.xs[:, q].copy(),
                )
                tableau.rs[:] ^= tableau.xs[:, q] & tableau.zs[:, q]
            return True

        if isinstance(args, clifford.ActOnStabilizerCHFormArgs):
            if not protocols.has_stabilizer_effect(self):
                return NotImplemented
            q = args.qubit_map[qubits[0]]
            state = args.state
            if self._exponent % 2 == 1:
                # Prescription for H left multiplication
                # Reference: https://arxiv.org/abs/1808.00128
                # Equations 48, 49 and Proposition 4
                t = state.s ^ (state.G[q, :] & state.v)
                u = state.s ^ (state.F[q, :] & (~state.v)) ^ (state.M[q, :] & state.v)

                alpha = sum(state.G[q, :] & (~state.v) & state.s) % 2
                beta = sum(state.M[q, :] & (~state.v) & state.s)
                beta += sum(state.F[q, :] & state.v & state.M[q, :])
                beta += sum(state.F[q, :] & state.v & state.s)
                beta %= 2

                delta = (state.gamma[q] + 2 * (alpha + beta)) % 4

                state.update_sum(t, u, delta=delta, alpha=alpha)
            # Adjust the global phase based on the global_shift parameter.
            args.state.omega *= np.exp(1j * np.pi * self.global_shift * self.exponent)
            return True

        return NotImplemented

    def _decompose_(self, qubits):
        q = qubits[0]

        if self._exponent == 1:
            yield cirq.Y(q) ** 0.5
            yield cirq.XPowGate(global_shift=-0.25).on(q)
            return

        yield YPowGate(exponent=0.25).on(q)
        yield XPowGate(exponent=self._exponent).on(q)
        yield YPowGate(exponent=-0.25).on(q)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('H',), exponent=self._diagram_exponent(args)
        )

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        if self._exponent == 1:
            return args.format('h {0};\n', qubits[0])

        return args.format(
            'ry({0:half_turns}) {3};\nrx({1:half_turns}) {3};\nry({2:half_turns}) {3};\n',
            0.25,
            self._exponent,
            -0.25,
            qubits[0],
        )

    def _quil_(
        self, qubits: Tuple['cirq.Qid', ...], formatter: 'cirq.QuilFormatter'
    ) -> Optional[str]:
        if self._exponent == 1:
            return formatter.format('H {0}\n', qubits[0])
        return formatter.format(
            'RY({0}) {3}\nRX({1}) {3}\nRY({2}) {3}\n',
            0.25 * np.pi,
            self._exponent * np.pi,
            -0.25 * np.pi,
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


class CZPowGate(
    eigen_gate.EigenGate, gate_features.TwoQubitGate, gate_features.InterchangeableQubitsGate
):
    """A gate that applies a phase to the |11⟩ state of two qubits.

    The unitary matrix of `CZPowGate(exponent=t)` is:

        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, g]]

    where:

        g = exp(i·π·t).

    `cirq.CZ`, the controlled Z gate, is an instance of this gate at
    `exponent=1`.
    """

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.pauli_interaction_gate import PauliInteractionGate

        if self.exponent % 2 == 1:
            return PauliInteractionGate.CZ.on(*qubits)
        if self.exponent % 2 == 0:
            return []
        return NotImplemented

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [
            (0, np.diag([1, 1, 1, 0])),
            (1, np.diag([0, 0, 0, 1])),
        ]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _apply_unitary_(
        self, args: 'protocols.ApplyUnitaryArgs'
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

    def _act_on_(self, args: 'cirq.ActOnArgs', qubits: Sequence['cirq.Qid']):
        from cirq.sim import clifford

        if isinstance(args, clifford.ActOnCliffordTableauArgs):
            if not protocols.has_stabilizer_effect(self):
                return NotImplemented
            tableau = args.tableau
            q1 = args.qubit_map[qubits[0]]
            q2 = args.qubit_map[qubits[1]]
            if self._exponent % 2 == 1:
                (tableau.xs[:, q2], tableau.zs[:, q2]) = (
                    tableau.zs[:, q2].copy(),
                    tableau.xs[:, q2].copy(),
                )
                tableau.rs[:] ^= tableau.xs[:, q2] & tableau.zs[:, q2]
                tableau.rs[:] ^= (
                    tableau.xs[:, q1]
                    & tableau.zs[:, q2]
                    & (~(tableau.xs[:, q2] ^ tableau.zs[:, q1]))
                )
                tableau.xs[:, q2] ^= tableau.xs[:, q1]
                tableau.zs[:, q1] ^= tableau.zs[:, q2]
                (tableau.xs[:, q2], tableau.zs[:, q2]) = (
                    tableau.zs[:, q2].copy(),
                    tableau.xs[:, q2].copy(),
                )
                tableau.rs[:] ^= tableau.xs[:, q2] & tableau.zs[:, q2]
            return True

        if isinstance(args, clifford.ActOnStabilizerCHFormArgs):
            if not protocols.has_stabilizer_effect(self):
                return NotImplemented
            q1 = args.qubit_map[qubits[0]]
            q2 = args.qubit_map[qubits[1]]
            state = args.state
            if self._exponent % 2 == 1:
                # Prescription for CZ left multiplication.
                # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
                state.M[q1, :] ^= state.G[q2, :]
                state.M[q2, :] ^= state.G[q1, :]
            # Adjust the global phase based on the global_shift parameter.
            args.state.omega *= np.exp(1j * np.pi * self.global_shift * self.exponent)
            return True

        return NotImplemented

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        z_phase = 1j ** self._exponent
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
        num_controls: int = None,
        control_values: Optional[Sequence[Union[int, Collection[int]]]] = None,
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
        """
        result = super().controlled(num_controls, control_values, control_qid_shape)
        if (
            self._global_shift == 0
            and isinstance(result, controlled_gate.ControlledGate)
            and result.control_values[-1] == (1,)
            and result.control_qid_shape[-1] == 2
        ):
            return cirq.CCZPowGate(
                exponent=self._exponent, global_shift=self._global_shift
            ).controlled(
                result.num_controls() - 1, result.control_values[:-1], result.control_qid_shape[:-1]
            )
        return result

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('@', '@'), exponent=self._diagram_exponent(args)
        )

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        if self._exponent != 1:
            return None  # Don't have an equivalent gate in QASM
        args.validate_version('2.0')
        return args.format('cz {0},{1};\n', qubits[0], qubits[1])

    def _quil_(
        self, qubits: Tuple['cirq.Qid', ...], formatter: 'cirq.QuilFormatter'
    ) -> Optional[str]:
        if self._exponent == 1:
            return formatter.format('CZ {0} {1}\n', qubits[0], qubits[1])
        return formatter.format(
            'CPHASE({0}) {1} {2}\n', self._exponent * np.pi, qubits[0], qubits[1]
        )

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
        return 'cirq.CZPowGate(exponent={}, global_shift={!r})'.format(
            proper_repr(self._exponent), self._global_shift
        )


class CXPowGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A gate that applies a controlled power of an X gate.

    When applying CNOT (controlled-not) to qubits, you can either use
    positional arguments CNOT(q1, q2), where q2 is toggled when q1 is on,
    or named arguments CNOT(control=q1, target=q2).
    (Mixing the two is not permitted.)

    The unitary matrix of `CXPowGate(exponent=t)` is:

        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, g·c, -i·g·s],
         [0, 0, -i·g·s, g·c]]

    where:

        c = cos(π·t/2)
        s = sin(π·t/2)
        g = exp(i·π·t/2).

    `cirq.CNOT`, the controlled NOT gate, is an instance of this gate at
    `exponent=1`.
    """

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.pauli_interaction_gate import PauliInteractionGate

        if self.exponent % 2 == 1:
            return PauliInteractionGate.CNOT.on(*qubits)
        if self.exponent % 2 == 0:
            return []
        return NotImplemented

    def _decompose_(self, qubits):
        c, t = qubits
        yield YPowGate(exponent=-0.5).on(t)
        yield CZ(c, t) ** self._exponent
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

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('@', 'X'), exponent=self._diagram_exponent(args)
        )

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
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

    def _act_on_(self, args: 'cirq.ActOnArgs', qubits: Sequence['cirq.Qid']):
        from cirq.sim import clifford

        if isinstance(args, clifford.ActOnCliffordTableauArgs):
            if not protocols.has_stabilizer_effect(self):
                return NotImplemented
            tableau = args.tableau
            q1 = args.qubit_map[qubits[0]]
            q2 = args.qubit_map[qubits[1]]
            if self._exponent % 2 == 1:
                tableau.rs[:] ^= (
                    tableau.xs[:, q1]
                    & tableau.zs[:, q2]
                    & (~(tableau.xs[:, q2] ^ tableau.zs[:, q1]))
                )
                tableau.xs[:, q2] ^= tableau.xs[:, q1]
                tableau.zs[:, q1] ^= tableau.zs[:, q2]
            return True

        if isinstance(args, clifford.ActOnStabilizerCHFormArgs):
            if not protocols.has_stabilizer_effect(self):
                return NotImplemented
            q1 = args.qubit_map[qubits[0]]
            q2 = args.qubit_map[qubits[1]]
            state = args.state
            if self._exponent % 2 == 1:
                # Prescription for CX left multiplication.
                # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
                state.gamma[q1] = (
                    state.gamma[q1]
                    + state.gamma[q2]
                    + 2 * (sum(state.M[q1, :] & state.F[q2, :]) % 2)
                ) % 4
                state.G[q2, :] ^= state.G[q1, :]
                state.F[q1, :] ^= state.F[q2, :]
                state.M[q1, :] ^= state.M[q2, :]
            # Adjust the global phase based on the global_shift parameter.
            args.state.omega *= np.exp(1j * np.pi * self.global_shift * self.exponent)
            return True

        return NotImplemented

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        cnot_phase = 1j ** self._exponent
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
        num_controls: int = None,
        control_values: Optional[Sequence[Union[int, Collection[int]]]] = None,
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
        """
        result = super().controlled(num_controls, control_values, control_qid_shape)
        if (
            self._global_shift == 0
            and isinstance(result, controlled_gate.ControlledGate)
            and result.control_values[-1] == (1,)
            and result.control_qid_shape[-1] == 2
        ):
            return cirq.CCXPowGate(
                exponent=self._exponent, global_shift=self._global_shift
            ).controlled(
                result.num_controls() - 1, result.control_values[:-1], result.control_qid_shape[:-1]
            )
        return result

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        if self._exponent != 1:
            return None  # Don't have an equivalent gate in QASM
        args.validate_version('2.0')
        return args.format('cx {0},{1};\n', qubits[0], qubits[1])

    def _quil_(
        self, qubits: Tuple['cirq.Qid', ...], formatter: 'cirq.QuilFormatter'
    ) -> Optional[str]:
        if self._exponent == 1:
            return formatter.format('CNOT {0} {1}\n', qubits[0], qubits[1])
        return None

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

    def on(self, *args: 'cirq.Qid', **kwargs: 'cirq.Qid') -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(args, kwargs)
        )


def rx(rads: value.TParamVal) -> Rx:
    """Returns a gate with the matrix e^{-i X rads / 2}."""
    return Rx(rads=rads)


def ry(rads: value.TParamVal) -> Ry:
    """Returns a gate with the matrix e^{-i Y rads / 2}."""
    return Ry(rads=rads)


def rz(rads: value.TParamVal) -> Rz:
    """Returns a gate with the matrix e^{-i Z rads / 2}."""
    return Rz(rads=rads)


def cphase(rads: value.TParamVal) -> CZPowGate:
    """Returns a gate with the matrix diag(1, 1, 1, e^{i rads}."""
    return CZPowGate(exponent=rads / _pi(rads))


H = HPowGate()
document(
    H,
    """The Hadamard gate.

    The `exponent=1` instance of `cirq.HPowGate`.

    Matrix:
    ```
        [[s, s],
         [s, -s]]
    ```
        where s = sqrt(0.5).
    """,
)

S = ZPowGate(exponent=0.5)
document(
    S,
    """The Clifford S gate.

    The `exponent=0.5` instance of `cirq.ZPowGate`.

    Matrix:
    ```
        [[1, 0],
         [0, i]]
    ```
    """,
)

T = ZPowGate(exponent=0.25)
document(
    T,
    """The non-Clifford T gate.

    The `exponent=0.25` instance of `cirq.ZPowGate`.

    Matrix:
    ```
        [[1, 0]
         [0, exp(i pi / 4)]]
    ```
    """,
)

CZ = CZPowGate()
document(
    CZ,
    """The controlled Z gate.

    The `exponent=1` instance of `cirq.CZPowGate`.

    Matrix:
    ```
        [[1 . . .],
         [. 1 . .],
         [. . 1 .],
         [. . . -1]]
    ```
    """,
)

CNotPowGate = CXPowGate
CNOT = CX = CNotPowGate()
document(
    CNOT,
    """The controlled NOT gate.

    The `exponent=1` instance of `cirq.CXPowGate`.

    Matrix:
    ```
        [[1 . . .],
         [. 1 . .],
         [. . . 1],
         [. . 1 .]]
    ```
    """,
)
