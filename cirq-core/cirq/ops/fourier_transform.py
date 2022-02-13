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

from typing import AbstractSet, Any, Dict, Union

import numpy as np
import sympy

import cirq
from cirq import value, _compat
from cirq.ops import raw_types


@value.value_equality
class QuantumFourierTransformGate(raw_types.Gate):
    """Switches from the computational basis to the frequency basis."""

    def __init__(self, num_qubits: int, *, without_reverse: bool = False):
        """Inits QuantumFourierTransformGate.

        Args:
            num_qubits: The number of qubits the gate applies to.
            without_reverse: Whether or not to include the swaps at the end
                of the circuit decomposition that reverse the order of the
                qubits. These are technically necessary in order to perform the
                correct effect, but can almost always be optimized away by just
                performing later operations on different qubits.
        """
        self._num_qubits = num_qubits
        self._without_reverse = without_reverse

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'num_qubits': self._num_qubits,
            'without_reverse': self._without_reverse,
        }

    def _value_equality_values_(self):
        return self._num_qubits, self._without_reverse

    def num_qubits(self) -> int:
        return self._num_qubits

    def _decompose_(self, qubits):
        if len(qubits) == 0:
            return
        yield cirq.H(qubits[0])
        for i in range(1, len(qubits)):
            yield PhaseGradientGate(num_qubits=i, exponent=0.5).on(*qubits[:i][::-1]).controlled_by(
                qubits[i]
            )
            yield cirq.H(qubits[i])
        if not self._without_reverse:
            for i in range(len(qubits) // 2):
                yield cirq.SWAP(qubits[i], qubits[-i - 1])

    def _has_unitary_(self):
        return True

    def __str__(self) -> str:
        return 'qft[norev]' if self._without_reverse else 'qft'

    def __repr__(self) -> str:
        return (
            'cirq.QuantumFourierTransformGate('
            f'num_qubits={self._num_qubits!r}, '
            f'without_reverse={self._without_reverse!r})'
        )

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return cirq.CircuitDiagramInfo(
            wire_symbols=(str(self),) + tuple(f'#{k+1}' for k in range(1, self._num_qubits)),
            exponent_qubit_index=0,
        )


@value.value_equality
class PhaseGradientGate(raw_types.Gate):
    """Phases each state |k⟩ out of n by e^(2*pi*i*k/n*exponent)."""

    def __init__(self, *, num_qubits: int, exponent: Union[float, sympy.Basic]):
        self._num_qubits = num_qubits
        self.exponent = exponent

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'num_qubits': self._num_qubits,
            'exponent': self.exponent,
        }

    def _value_equality_values_(self):
        return self._num_qubits, self.exponent

    def num_qubits(self) -> int:
        return self._num_qubits

    def _decompose_(self, qubits):
        for i, q in enumerate(qubits):
            yield cirq.Z(q) ** (self.exponent / 2 ** i)

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs'):
        if isinstance(self.exponent, sympy.Basic):
            return NotImplemented

        n = int(np.prod([args.target_tensor.shape[k] for k in args.axes], dtype=np.int64))
        for i in range(n):
            p = 1j ** (4 * i / n * self.exponent)
            args.target_tensor[args.subspace_index(big_endian_bits_int=i)] *= p

        return args.target_tensor

    def __pow__(self, power):
        new_exponent = cirq.mul(self.exponent, power, NotImplemented)
        if new_exponent is NotImplemented:
            # coverage: ignore
            return NotImplemented
        return PhaseGradientGate(num_qubits=self._num_qubits, exponent=new_exponent)

    def _unitary_(self):
        if isinstance(self.exponent, sympy.Basic):
            return NotImplemented

        size = 1 << self._num_qubits
        return np.diag([1j ** (4 * i / size * self.exponent) for i in range(size)])

    def _has_unitary_(self) -> bool:
        return not cirq.is_parameterized(self)

    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self.exponent)

    def _parameter_names_(self) -> AbstractSet[str]:
        return cirq.parameter_names(self.exponent)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'PhaseGradientGate':
        new_exponent = cirq.resolve_parameters(self.exponent, resolver, recursive)
        if new_exponent is self.exponent:
            return self
        return PhaseGradientGate(num_qubits=self._num_qubits, exponent=new_exponent)

    def __str__(self) -> str:
        return f'Grad[{self._num_qubits}]' + (f'^{self.exponent}' if self.exponent != 1 else '')

    def __repr__(self) -> str:
        return (
            'cirq.PhaseGradientGate('
            f'num_qubits={self._num_qubits!r}, '
            f'exponent={_compat.proper_repr(self.exponent)})'
        )

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return cirq.CircuitDiagramInfo(
            wire_symbols=('Grad',) + tuple(f'#{k+1}' for k in range(1, self._num_qubits)),
            exponent=self.exponent,
            exponent_qubit_index=0,
        )


def qft(
    *qubits: 'cirq.Qid', without_reverse: bool = False, inverse: bool = False
) -> 'cirq.Operation':
    """The quantum Fourier transform.

    Transforms a qubit register from the computational basis to the frequency
    basis.

    The inverse quantum Fourier transform is `cirq.qft(*qubits)**-1` or
    equivalently `cirq.inverse(cirq.qft(*qubits))`.

    Args:
        qubits: The qubits to apply the qft to.
        without_reverse: When set, swap gates at the end of the qft are omitted.
            This reverses the qubit order relative to the standard qft effect,
            but makes the gate cheaper to apply.
        inverse: If set, the inverse qft is performed instead of the qft.
            Equivalent to calling `cirq.inverse` on the result, or raising it
            to the -1.

    Returns:
        A `cirq.Operation` applying the qft to the given qubits.
    """
    result = QuantumFourierTransformGate(len(qubits), without_reverse=without_reverse).on(*qubits)
    if inverse:
        result = cirq.inverse(result)
    return result
