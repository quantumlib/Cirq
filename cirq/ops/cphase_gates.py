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
    CZ: Controlled phase gate.
Each of these are implemented as EigenGates, which means that they can be
raised to a power (i.e. cirq.H**0.5). See the definition in EigenGate.
"""

from typing import Optional, Tuple, Union

import numpy as np

import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import gate_features, eigen_gate

from cirq.type_workarounds import NotImplementedType


class CZPowGate00(eigen_gate.EigenGate,
                gate_features.TwoQubitGate,
                gate_features.InterchangeableQubitsGate):
    """A gate that applies a phase to the |00⟩ state of two qubits.
    The unitary matrix of `CZPowGate00(exponent=t)` is:
        [[g, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    where:
        g = exp(i·π·t).
    `cirq.CZ00`, the controlled Z gate, is an instance of this gate at
    `exponent=1`.
    """

    def _eigen_components(self):
        return [
            (0, np.diag([0, 1, 1, 1])),
            (1, np.diag([1, 0, 0, 0])),
        ]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs'
                       ) -> Union[np.ndarray, NotImplementedType]:
        if protocols.is_parameterized(self):
            return NotImplemented

        c = 1j**(2 * self._exponent)
        one_one = args.subspace_index(0b00)
        args.target_tensor[one_one] *= c
        p = 1j**(2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return args.target_tensor

    def _phase_by_(self, phase_turns, qubit_index):
        return self

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                              ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
                wire_symbols=('@', '@00'),
                exponent=self._diagram_exponent(args))

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'CZ00'
        return 'CZ00**{!r}'.format(self._exponent)

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.CZ00'
            return '(cirq.CZ00**{})'.format(proper_repr(self._exponent))
        return (
            'cirq.CZPowGate00(exponent={}, '
            'global_shift={!r})'
        ).format(proper_repr(self._exponent), self._global_shift)


CZ00 = CZPowGate00()
document(
    CZ00, """The controlled Z gate.
    The `exponent=1` instance of `cirq.CZPowGate00`.
    Matrix:
        [[-1 . . .],
         [. 1 . .],
         [. . 1 .],
         [. . . 1]]
    """)
    