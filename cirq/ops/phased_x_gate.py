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
from typing import (AbstractSet, Any, cast, Dict, Optional, Sequence, Tuple,
                    Union)

import math
import numpy as np
import sympy

import cirq
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq.ops import gate_features, common_gates
from cirq.type_workarounds import NotImplementedType


@value.value_equality(manual_cls=True, approximate=True)
class PhasedXPowGate(gate_features.SingleQubitGate):
    """A gate equivalent to the circuit ───Z^-p───X^t───Z^p───."""

    def __init__(self,
                 *,
                 phase_exponent: Union[float, sympy.Symbol],
                 exponent: Union[float, sympy.Symbol] = 1.0,
                 global_shift: float = 0.0) -> None:
        """
        Args:
            phase_exponent: The exponent on the Z gates conjugating the X gate.
            exponent: The exponent on the X gate conjugated by Zs.
            global_shift: How much to shift the operation's eigenvalues at
                exponent=1.
        """
        self._phase_exponent = value.canonicalize_half_turns(phase_exponent)
        self._exponent = exponent
        self._global_shift = global_shift

    def _qasm_(self, args: 'cirq.QasmArgs',
               qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        if cirq.is_parameterized(self):
            return None

        args.validate_version('2.0')

        e = cast(float, value.canonicalize_half_turns(self._exponent))
        p = cast(float, self.phase_exponent)
        epsilon = 10**-args.precision

        if abs(e + 0.5) <= epsilon:
            return args.format('u2({0:half_turns}, {1:half_turns}) {2};\n',
                               p + 0.5, -p - 0.5, qubits[0])

        if abs(e - 0.5) <= epsilon:
            return args.format('u2({0:half_turns}, {1:half_turns}) {2};\n',
                               p - 0.5, -p + 0.5, qubits[0])

        return args.format(
            'u3({0:half_turns}, {1:half_turns}, {2:half_turns}) {3};\n',
            -e, p + 0.5, -p - 0.5, qubits[0])

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        assert len(qubits) == 1
        q = qubits[0]
        z = cirq.Z(q)**self._phase_exponent
        x = cirq.X(q)**self._exponent
        if protocols.is_parameterized(z):
            return NotImplemented
        return z**-1, x, z

    @property
    def exponent(self) -> Union[float, sympy.Symbol]:
        """The exponent on the central X gate conjugated by the Z gates."""
        return self._exponent

    @property
    def phase_exponent(self) -> Union[float, sympy.Symbol]:
        """The exponent on the Z gates conjugating the X gate."""
        return self._phase_exponent

    @property
    def global_shift(self) -> float:
        return self._global_shift

    def __pow__(self, exponent: Union[float, sympy.Symbol]) -> 'PhasedXPowGate':
        new_exponent = protocols.mul(self._exponent, exponent, NotImplemented)
        if new_exponent is NotImplemented:
            return NotImplemented
        return PhasedXPowGate(phase_exponent=self._phase_exponent,
                              exponent=new_exponent,
                              global_shift=self._global_shift)

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _has_unitary_(self):
        return not self._is_parameterized_()

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        """See `cirq.SupportsUnitary`."""
        if self._is_parameterized_():
            return None
        z = protocols.unitary(cirq.Z**self._phase_exponent)
        x = protocols.unitary(cirq.X**self._exponent)
        p = np.exp(1j * np.pi * self._global_shift * self._exponent)
        return np.dot(np.dot(z, x), np.conj(z)) * p

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if self._is_parameterized_():
            return NotImplemented
        phase_angle = np.pi * self._phase_exponent / 2
        angle = np.pi * self._exponent / 2
        phase = 1j**(2 * self._exponent * (self._global_shift + 0.5))
        return value.LinearDict({
            'I': phase * np.cos(angle),
            'X': -1j * phase * np.sin(angle) * np.cos(2 * phase_angle),
            'Y': -1j * phase * np.sin(angle) * np.sin(2 * phase_angle),
        })

    def _is_parameterized_(self) -> bool:
        """See `cirq.SupportsParameterization`."""
        return (protocols.is_parameterized(self._exponent) or
                protocols.is_parameterized(self._phase_exponent))

    def _parameter_names_(self) -> AbstractSet[str]:
        """See `cirq.SupportsParameterization`."""
        return (protocols.parameter_names(self._exponent) |
                protocols.parameter_names(self._phase_exponent))

    def _resolve_parameters_(self, param_resolver) -> 'PhasedXPowGate':
        """See `cirq.SupportsParameterization`."""
        return PhasedXPowGate(
            phase_exponent=param_resolver.value_of(self._phase_exponent),
            exponent=param_resolver.value_of(self._exponent),
            global_shift=self._global_shift)

    def _phase_by_(self, phase_turns, qubit_index):
        """See `cirq.SupportsPhase`."""
        assert qubit_index == 0
        return PhasedXPowGate(
            exponent=self._exponent,
            phase_exponent=self._phase_exponent + phase_turns * 2,
            global_shift=self._global_shift)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                              ) -> 'cirq.CircuitDiagramInfo':
        """See `cirq.SupportsCircuitDiagramInfo`."""

        return protocols.CircuitDiagramInfo(
            wire_symbols=(f'PhX({args.format_real(self.phase_exponent)})',),
            exponent=value.canonicalize_half_turns(self._exponent))

    def __str__(self) -> str:
        info = protocols.circuit_diagram_info(self)
        if info.exponent == 1:
            return info.wire_symbols[0]
        return f'{info.wire_symbols[0]}^{info.exponent}'

    def __repr__(self) -> str:
        args = [f'phase_exponent={proper_repr(self.phase_exponent)}']
        if self.exponent != 1:
            args.append(f'exponent={proper_repr(self.exponent)}')
        if self._global_shift != 0:
            args.append(f'global_shift={self._global_shift!r}')
        args_str = ', '.join(args)
        return f'cirq.PhasedXPowGate({args_str})'

    def _period(self):
        exponents = [self._global_shift, 1 + self._global_shift]
        real_periods = [abs(2/e) for e in exponents if e != 0]
        int_periods = [int(np.round(e)) for e in real_periods]
        if any(i != r for i, r in zip(real_periods, int_periods)):
            return None
        if len(int_periods) == 1:
            return int_periods[0]
        return int_periods[0] * int_periods[1] / math.gcd(*int_periods)

    @property
    def _canonical_exponent(self):
        period = self._period()
        if not period or isinstance(self._exponent, sympy.Basic):
            return self._exponent

        return self._exponent % period

    def _value_equality_values_cls_(self):
        if self.phase_exponent == 0:
            return common_gates.XPowGate
        if self.phase_exponent == 0.5:
            return common_gates.YPowGate
        return PhasedXPowGate

    def _value_equality_values_(self):
        if self.phase_exponent == 0:
            return common_gates.XPowGate(
                exponent=self._exponent,
                global_shift=self._global_shift)._value_equality_values_()
        if self.phase_exponent == 0.5:
            return common_gates.YPowGate(
                exponent=self._exponent,
                global_shift=self._global_shift)._value_equality_values_()
        return self.phase_exponent, self._canonical_exponent, self._global_shift

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(
            self, ['phase_exponent', 'exponent', 'global_shift'])
