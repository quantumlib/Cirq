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

from typing import AbstractSet, Sequence, Union, List, Tuple

import pytest

import numpy as np
import sympy

import cirq
from cirq._compat import proper_repr
from cirq.type_workarounds import NotImplementedType


class GoodGate(cirq.SingleQubitGate):
    def __init__(
        self,
        *,
        phase_exponent: Union[float, sympy.Symbol],
        exponent: Union[float, sympy.Symbol] = 1.0,
    ) -> None:
        self.phase_exponent = cirq.canonicalize_half_turns(phase_exponent)
        self.exponent = exponent

    def _has_unitary_(self):
        return not cirq.is_parameterized(self)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        if cirq.is_parameterized(self):
            return NotImplemented
        z = cirq.unitary(cirq.Z ** self.phase_exponent)
        x = cirq.unitary(cirq.X ** self.exponent)
        return np.dot(np.dot(z, x), np.conj(z))

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> Union[np.ndarray, NotImplementedType]:
        if self.exponent != 1 or cirq.is_parameterized(self):
            return NotImplemented

        zero = cirq.slice_for_qubits_equal_to(args.axes, 0)
        one = cirq.slice_for_qubits_equal_to(args.axes, 1)
        c = np.exp(1j * np.pi * self.phase_exponent)

        args.target_tensor[one] *= c.conj()
        args.available_buffer[zero] = args.target_tensor[one]
        args.available_buffer[one] = args.target_tensor[zero]
        args.available_buffer[one] *= c

        return args.available_buffer

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        assert len(qubits) == 1
        q = qubits[0]
        z = cirq.Z(q) ** self.phase_exponent
        x = cirq.X(q) ** self.exponent
        if cirq.is_parameterized(z):
            # coverage: ignore
            return NotImplemented
        return z ** -1, x, z

    def _pauli_expansion_(self) -> cirq.LinearDict[str]:
        if self._is_parameterized_():
            return NotImplemented
        phase_angle = np.pi * self.phase_exponent / 2
        angle = np.pi * self.exponent / 2
        global_phase = np.exp(1j * angle)
        return cirq.LinearDict(
            {
                'I': global_phase * np.cos(angle),
                'X': -1j * global_phase * np.sin(angle) * np.cos(2 * phase_angle),
                'Y': -1j * global_phase * np.sin(angle) * np.sin(2 * phase_angle),
            }
        )

    def _phase_by_(self, phase_turns, qubit_index):
        assert qubit_index == 0
        return GoodGate(
            exponent=self.exponent, phase_exponent=self.phase_exponent + phase_turns * 2
        )

    def __pow__(self, exponent: Union[float, sympy.Symbol]) -> 'GoodGate':
        new_exponent = cirq.mul(self.exponent, exponent, NotImplemented)
        if new_exponent is NotImplemented:
            # coverage: ignore
            return NotImplemented
        return GoodGate(phase_exponent=self.phase_exponent, exponent=new_exponent)

    def __repr__(self):
        args = [f'phase_exponent={proper_repr(self.phase_exponent)}']
        if self.exponent != 1:
            args.append(f'exponent={proper_repr(self.exponent)}')
        return f"GoodGate({', '.join(args)})"

    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self.exponent) or cirq.is_parameterized(self.phase_exponent)

    def _parameter_names_(self) -> AbstractSet[str]:
        return cirq.parameter_names(self.exponent) | cirq.parameter_names(self.phase_exponent)

    def _resolve_parameters_(self, resolver, recursive) -> 'GoodGate':
        return GoodGate(
            phase_exponent=resolver.value_of(self.phase_exponent, recursive),
            exponent=resolver.value_of(self.exponent, recursive),
        )

    def _identity_tuple(self):
        return (GoodGate, self.phase_exponent, self.exponent)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            # coverage: ignore
            return NotImplemented
        return self._identity_tuple() == other._identity_tuple()


class BadGateIsParameterized(GoodGate):
    def _is_parameterized_(self) -> bool:
        return not super()._is_parameterized_()


class BadGateParameterNames(GoodGate):
    def _parameter_names_(self) -> AbstractSet[str]:
        return super()._parameter_names_() | {'not_a_param'}


class BadGateApplyUnitaryToTensor(GoodGate):
    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> Union[np.ndarray, NotImplementedType]:
        if self.exponent != 1 or cirq.is_parameterized(self):
            # coverage: ignore
            return NotImplemented

        zero = cirq.slice_for_qubits_equal_to(args.axes, 0)
        one = cirq.slice_for_qubits_equal_to(args.axes, 1)
        c = np.exp(1j * np.pi * self.phase_exponent)

        args.target_tensor[one] *= c
        args.available_buffer[zero] = args.target_tensor[one]
        args.available_buffer[one] = args.target_tensor[zero]
        args.available_buffer[one] *= c

        return args.available_buffer


class BadGateDecompose(GoodGate):
    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        assert len(qubits) == 1
        q = qubits[0]
        z = cirq.Z(q) ** self.phase_exponent
        x = cirq.X(q) ** (2 * self.exponent)
        if cirq.is_parameterized(z):
            # coverage: ignore
            return NotImplemented
        return z ** -1, x, z


class BadGatePauliExpansion(GoodGate):
    def _pauli_expansion_(self) -> cirq.LinearDict[str]:
        return cirq.LinearDict({'I': 10})


class BadGatePhaseBy(GoodGate):
    def _phase_by_(self, phase_turns, qubit_index):
        assert qubit_index == 0
        return BadGatePhaseBy(
            exponent=self.exponent, phase_exponent=self.phase_exponent + phase_turns * 4
        )


class BadGateRepr(GoodGate):
    def __repr__(self):
        args = [f'phase_exponent={2 * self.phase_exponent!r}']
        if self.exponent != 1:
            # coverage: ignore
            args.append(f'exponent={proper_repr(self.exponent)}')
        return f"BadGateRepr({', '.join(args)})"


class GoodEigenGate(cirq.EigenGate, cirq.SingleQubitGate):
    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [
            (0, np.diag([1, 0])),
            (1, np.diag([0, 1])),
        ]

    def __repr__(self):
        return 'GoodEigenGate(exponent={}, global_shift={!r})'.format(
            proper_repr(self._exponent), self._global_shift
        )


class BadEigenGate(GoodEigenGate):
    def _eigen_shifts(self):
        return [0, 0]

    def __repr__(self):
        return 'BadEigenGate(exponent={}, global_shift={!r})'.format(
            proper_repr(self._exponent), self._global_shift
        )


def test_assert_implements_consistent_protocols():
    cirq.testing.assert_implements_consistent_protocols(
        GoodGate(phase_exponent=0.0), global_vals={'GoodGate': GoodGate}
    )

    cirq.testing.assert_implements_consistent_protocols(
        GoodGate(phase_exponent=0.25), global_vals={'GoodGate': GoodGate}
    )

    cirq.testing.assert_implements_consistent_protocols(
        GoodGate(phase_exponent=sympy.Symbol('t')), global_vals={'GoodGate': GoodGate}
    )

    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(
            BadGateIsParameterized(phase_exponent=0.25)
        )

    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(
            BadGateParameterNames(phase_exponent=0.25)
        )

    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(
            BadGateApplyUnitaryToTensor(phase_exponent=0.25)
        )

    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(BadGateDecompose(phase_exponent=0.25))

    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(
            BadGatePauliExpansion(phase_exponent=0.25)
        )

    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(BadGatePhaseBy(phase_exponent=0.25))

    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(
            BadGateRepr(phase_exponent=0.25), global_vals={'BadGateRepr': BadGateRepr}
        )


def test_assert_eigengate_implements_consistent_protocols():
    cirq.testing.assert_eigengate_implements_consistent_protocols(
        GoodEigenGate, global_vals={'GoodEigenGate': GoodEigenGate}
    )

    with pytest.raises(AssertionError):
        cirq.testing.assert_eigengate_implements_consistent_protocols(
            BadEigenGate, global_vals={'BadEigenGate': BadEigenGate}
        )


def test_assert_commutes_magic_method_consistent_with_unitaries():
    gate_op = cirq.CNOT(*cirq.LineQubit.range(2))
    with pytest.raises(TypeError):
        cirq.testing.assert_commutes_magic_method_consistent_with_unitaries(gate_op)

    exponents = [sympy.Symbol('s'), 0.1, 0.2]
    gates = [cirq.ZPowGate(exponent=e) for e in exponents]
    cirq.testing.assert_commutes_magic_method_consistent_with_unitaries(*gates)

    cirq.testing.assert_commutes_magic_method_consistent_with_unitaries(cirq.Z, cirq.CNOT)
