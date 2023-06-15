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

import abc
from typing import Any, Dict, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self

from cirq import protocols
from cirq.ops import pauli_string as ps, raw_types

if TYPE_CHECKING:
    import cirq


class PauliStringGateOperation(raw_types.Operation, metaclass=abc.ABCMeta):
    def __init__(self, pauli_string: ps.PauliString) -> None:
        self._pauli_string = pauli_string

    @property
    def pauli_string(self) -> 'cirq.PauliString':
        return self._pauli_string

    def validate_args(self, qubits: Sequence[raw_types.Qid]) -> None:
        if len(qubits) != len(self.pauli_string):
            raise ValueError('Incorrect number of qubits for gate')

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> Self:
        self.validate_args(new_qubits)
        return self.map_qubits(dict(zip(self.pauli_string.qubits, new_qubits)))

    @abc.abstractmethod
    def map_qubits(self, qubit_map: Dict[raw_types.Qid, raw_types.Qid]) -> Self:
        """Return an equivalent operation on new qubits with its Pauli string
        mapped to new qubits.

        new_pauli_string = self.pauli_string.map_qubits(qubit_map)
        """

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        return tuple(self.pauli_string)

    def _pauli_string_diagram_info(
        self, args: 'protocols.CircuitDiagramInfoArgs', exponent: Any = 1
    ) -> 'cirq.CircuitDiagramInfo':
        qubits = self.qubits if args.known_qubits is None else args.known_qubits
        syms = tuple(f'[{self.pauli_string[qubit]}]' for qubit in qubits)
        return protocols.CircuitDiagramInfo(wire_symbols=syms, exponent=exponent)
