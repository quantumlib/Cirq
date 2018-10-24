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

"""An `XPowGate` conjugated by `ZPowGate`s."""

from typing import Union, Sequence

import numpy as np

from cirq import value, protocols
from cirq.ops import gate_features, raw_types, common_gates, op_tree
from cirq.type_workarounds import NotImplementedType


class PhasedXPowGate(gate_features.SingleQubitGate,
                     gate_features.CompositeGate):
    """A gate equivalent to the circuit ───Z^-p───X^t───Z^p───.

    Attributes:
        phase_exponent: The exponent on the Z gates conjugating the X gate.
        exponent: The exponent on the X gate conjugated by Zs.
    """

    def __new__(cls,
                phase_exponent: Union[float, value.Symbol] = 0,
                exponent: Union[float, value.Symbol] = 1):
        """Substitutes a raw X or raw Y if possible.

        Args:
            phase_exponent: The exponent on the Z gates conjugating the X gate.
            exponent: The exponent on the X gate conjugated by Zs.
        """
        p = value.canonicalize_half_turns(phase_exponent)
        if p == 0:
            return common_gates.X**exponent
        if p == 0.5:
            return common_gates.Y**exponent
        if p == 1:
            return common_gates.X**-exponent
        if p == -0.5:
            return common_gates.Y**-exponent
        return super().__new__(cls)

    def __init__(self,
                 phase_exponent: Union[float, value.Symbol] = 0,
                 exponent: Union[float, value.Symbol] = 1) -> None:
        """
        Args:
            phase_exponent: The exponent on the Z gates conjugating the X gate.
            exponent: The exponent on the X gate conjugated by Zs.
        """
        self._phase_exponent = value.canonicalize_half_turns(phase_exponent)
        self._exponent = exponent

    def default_decompose(self, qubits: Sequence[raw_types.QubitId]
                          ) -> op_tree.OP_TREE:
        assert len(qubits) == 1
        q = qubits[0]
        yield common_gates.Z(q)**-self._phase_exponent
        yield common_gates.X(q)**self._exponent
        yield common_gates.Z(q)**self._phase_exponent

    @property
    def exponent(self) -> Union[float, value.Symbol]:
        return self._exponent

    @property
    def phase_exponent(self) -> Union[float, value.Symbol]:
        return self._phase_exponent

    def __pow__(self, exponent: Union[float, value.Symbol]) -> 'PhasedXPowGate':
        new_exponent = protocols.mul(self._exponent, exponent, NotImplemented)
        if new_exponent is NotImplemented:
            return NotImplemented
        return PhasedXPowGate(phase_exponent=self._phase_exponent,
                              exponent=new_exponent)

    def _trace_distance_bound_(self):
        """See `cirq.SupportsTraceDistanceBound`."""
        return protocols.trace_distance_bound(common_gates.X**self._exponent)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        """See `cirq.SupportsUnitary`."""
        if self._is_parameterized_():
            return NotImplemented
        z = protocols.unitary(common_gates.Z**self._phase_exponent)
        x = protocols.unitary(common_gates.X**self._exponent)
        return np.dot(np.dot(z, x), np.conj(z))

    def _is_parameterized_(self) -> bool:
        """See `cirq.SupportsParameterization`."""
        return (isinstance(self._exponent, value.Symbol) or
                isinstance(self._phase_exponent, value.Symbol))

    def _resolve_parameters_(self, param_resolver) -> 'PhasedXPowGate':
        """See `cirq.SupportsParameterization`."""
        return PhasedXPowGate(
            phase_exponent=param_resolver.value_of(self._phase_exponent),
            exponent=param_resolver.value_of(self._exponent))

    def _phase_by_(self, phase_turns, qubit_index):
        """See `cirq.SupportsPhase`."""
        assert qubit_index == 0
        return PhasedXPowGate(
            exponent=self._exponent,
            phase_exponent=self._phase_exponent + phase_turns * 2)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        """See `cirq.SupportsCircuitDiagramInfo`."""
        return protocols.CircuitDiagramInfo(
            wire_symbols=('PhasedX({})'.format(self._phase_exponent),),
            exponent=self._exponent)

    def __str__(self):
        info = protocols.circuit_diagram_info(self)
        if info.exponent == 1:
            return info.wire_symbols[0]
        return '{}^{}'.format(info.wire_symbols[0], info.exponent)

    def __repr__(self):
        return 'cirq.PhasedXPowGate({!r}, {!r})'.format(self._phase_exponent,
                                                        self._exponent)

    def _identity_tuple(self):
        return (PhasedXPowGate,
                value.canonicalize_half_turns(self._exponent),
                self._phase_exponent)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._identity_tuple() == other._identity_tuple()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._identity_tuple())
