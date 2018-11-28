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

"""Classes to represent displays."""

from typing import Any, Dict, Tuple

import abc

import numpy as np

from cirq import protocols
from cirq.ops import pauli_string, raw_types


class WaveFunctionDisplay(raw_types.Operation):
    """A display whose value is computed from the full wavefunction."""

    @abc.abstractmethod
    def value(self,
              state: np.ndarray,
              qubit_index_map: Dict[raw_types.QubitId, int]
              ) -> Any:
        pass


class PauliStringExpectation(WaveFunctionDisplay):

    def __init__(self, pauli_string: pauli_string.PauliString):
        self._pauli_string = pauli_string

    @property
    def qubits(self) -> Tuple[raw_types.QubitId, ...]:
        return self._pauli_string.qubits

    def with_qubits(self,
                    *new_qubits: raw_types.QubitId
                    ) -> 'PauliStringExpectation':
        return PauliStringExpectation(
                self._pauli_string.with_qubits(*new_qubits))

    def value(self,
              state: np.ndarray,
              qubit_index_map: Dict[raw_types.QubitId, int]
              ) -> float:
        num_qubits = state.shape[0].bit_length() - 1
        ket = np.reshape(np.copy(state), (2,) * num_qubits)
        for qubit, pauli in self._pauli_string.items():
            buffer = np.empty(ket.shape, dtype=state.dtype)
            args = protocols.ApplyUnitaryArgs(
                    target_tensor=ket,
                    available_buffer=buffer,
                    axes=(qubit_index_map[qubit],)
                    )
            ket = protocols.apply_unitary(pauli, args)
        ket = np.reshape(ket, state.shape)
        return np.dot(state.conj(), ket)
