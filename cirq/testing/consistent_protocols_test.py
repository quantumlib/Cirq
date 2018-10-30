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

from typing import Sequence, Union

import pytest

import numpy as np

import cirq
from cirq.type_workarounds import NotImplementedType


class GoodGate(cirq.SingleQubitGate):

    def __init__(self,
                 *,
                 phase_exponent: Union[float, cirq.Symbol],
                 exponent: Union[float, cirq.Symbol] = 1.0) -> None:
        self.phase_exponent = cirq.canonicalize_half_turns(phase_exponent)
        self.exponent = exponent

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        if self._is_parameterized_():
            return NotImplemented
        z = cirq.unitary(cirq.ops.common_gates.Z**self.phase_exponent)
        x = cirq.unitary(cirq.ops.common_gates.X**self.exponent)
        return np.dot(np.dot(z, x), np.conj(z))

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplementedType]:
        if self.exponent != 1 or self._is_parameterized_():
            return NotImplemented

        zero = cirq.slice_for_qubits_equal_to(axes, 0)
        one = cirq.slice_for_qubits_equal_to(axes, 1)
        c = np.exp(1j * np.pi * self.phase_exponent)

        target_tensor[one] *= c.conj()
        available_buffer[zero] = target_tensor[one]
        available_buffer[one] = target_tensor[zero]
        available_buffer[one] *= c

        return available_buffer

    def _decompose_(self, qubits: Sequence[cirq.QubitId]) -> cirq.OP_TREE:
        assert len(qubits) == 1
        q = qubits[0]
        z = cirq.ops.common_gates.Z(q)**self.phase_exponent
        x = cirq.ops.common_gates.X(q)**self.exponent
        if cirq.is_parameterized(z):
            return NotImplemented
        return z**-1, x, z

    def _phase_by_(self, phase_turns, qubit_index):
        assert qubit_index == 0
        return GoodGate(
            exponent=self.exponent,
            phase_exponent=self.phase_exponent + phase_turns * 2)

    def __pow__(self, exponent: Union[float, cirq.Symbol]) -> 'GoodGate':
        new_exponent = cirq.mul(self.exponent, exponent, NotImplemented)
        if new_exponent is NotImplemented:
            return NotImplemented
        return GoodGate(phase_exponent=self.phase_exponent,
                        exponent=new_exponent)

    def __repr__(self):
        args = ['phase_exponent={!r}'.format(self.phase_exponent)]
        if self.exponent != 1:
            args.append('exponent={!r}'.format(self.exponent))
        return 'cirq.testing.consistent_protocols_test.GoodGate({})'.format(
                ', '.join(args))

    def _is_parameterized_(self) -> bool:
        return (isinstance(self.exponent, cirq.Symbol) or
                isinstance(self.phase_exponent, cirq.Symbol))

    def _resolve_parameters_(self, param_resolver) -> 'GoodGate':
        return GoodGate(
            phase_exponent=param_resolver.value_of(self.phase_exponent),
            exponent=param_resolver.value_of(self.exponent))

    def _identity_tuple(self):
        return (GoodGate,
                self.phase_exponent,
                self.exponent)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._identity_tuple() == other._identity_tuple()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._identity_tuple())


class BadGateApplyUnitaryToTensor(GoodGate):

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplementedType]:
        if self.exponent != 1 or self._is_parameterized_():
            return NotImplemented

        zero = cirq.slice_for_qubits_equal_to(axes, 0)
        one = cirq.slice_for_qubits_equal_to(axes, 1)
        c = np.exp(1j * np.pi * self.phase_exponent)

        target_tensor[one] *= c
        available_buffer[zero] = target_tensor[one]
        available_buffer[one] = target_tensor[zero]
        available_buffer[one] *= c

        return available_buffer


class BadGateDecompose(GoodGate):

    def _decompose_(self, qubits: Sequence[cirq.QubitId]) -> cirq.OP_TREE:
        assert len(qubits) == 1
        q = qubits[0]
        z = cirq.ops.common_gates.Z(q)**self.phase_exponent
        x = cirq.ops.common_gates.X(q)**(2*self.exponent)
        if cirq.is_parameterized(z):
            return NotImplemented
        return z**-1, x, z


class BadGatePhaseBy(GoodGate):

    def _phase_by_(self, phase_turns, qubit_index):
        assert qubit_index == 0
        return BadGatePhaseBy(
            exponent=self.exponent,
            phase_exponent=self.phase_exponent + phase_turns * 4)


class BadGateRepr(GoodGate):

    def __repr__(self):
        args = ['phase_exponent={!r}'.format(2*self.phase_exponent)]
        if self.exponent != 1:
            args.append('exponent={!r}'.format(self.exponent))
        return 'cirq.testing.consistent_protocols_test.BadGateRepr({})'.format(
                ', '.join(args))


def test_assert_implements_consistent_protocols():
    cirq.testing.assert_implements_consistent_protocols(
            GoodGate(phase_exponent=0.0)
    )

    cirq.testing.assert_implements_consistent_protocols(
            GoodGate(phase_exponent=0.25)
    )

    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(
                BadGateApplyUnitaryToTensor(phase_exponent=0.25)
        )

    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(
                BadGateDecompose(phase_exponent=0.25)
        )

    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(
                BadGatePhaseBy(phase_exponent=0.25)
        )

    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(
                BadGateRepr(phase_exponent=0.25)
        )
