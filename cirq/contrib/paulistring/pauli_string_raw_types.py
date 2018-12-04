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

from typing import Any, Dict, Sequence, Tuple, TypeVar

import abc

from cirq import ops, protocols
from cirq.ops.pauli_string import PauliString


TSelf_PauliStringGateOperation = TypeVar('TSelf_PauliStringGateOperation',
                                         bound='PauliStringGateOperation')


class PauliStringGateOperation(ops.Operation,
                               metaclass=abc.ABCMeta):
    def __init__(self, pauli_string: PauliString) -> None:
        self.pauli_string = pauli_string

    def validate_args(self, qubits: Sequence[ops.QubitId]) -> None:
        if len(qubits) != len(self.pauli_string):
            raise ValueError('Incorrect number of qubits for gate')

    def with_qubits(self: TSelf_PauliStringGateOperation,
                    *new_qubits: ops.QubitId
                    ) -> TSelf_PauliStringGateOperation:
        self.validate_args(new_qubits)
        return self.map_qubits(dict(zip(self.pauli_string.qubits,
                                        new_qubits)))

    @abc.abstractmethod
    def map_qubits(self: TSelf_PauliStringGateOperation,
                   qubit_map: Dict[ops.QubitId, ops.QubitId]
                   ) -> TSelf_PauliStringGateOperation:
        """Return an equivalent operation on new qubits with its Pauli string
        mapped to new qubits.

        new_pauli_string = self.pauli_string.map_qubits(qubit_map)
        """
        pass

    @property
    def qubits(self) -> Tuple[ops.QubitId, ...]:
        return tuple(self.pauli_string)

    def _pauli_string_diagram_info(self,
                                   args: protocols.CircuitDiagramInfoArgs,
                                   exponent: Any = 1,
                                   exponent_absorbs_sign: bool = False,
                                   ) -> protocols.CircuitDiagramInfo:
        qubits = self.qubits if args.known_qubits is None else args.known_qubits
        syms = tuple('[{}]'.format(self.pauli_string[qubit])
                     for qubit in qubits)
        if exponent_absorbs_sign and self.pauli_string.negated:
            if isinstance(exponent, float):
                exponent = -exponent
            else:
                exponent = '-{!s}'.format(exponent)
        return protocols.CircuitDiagramInfo(wire_symbols=syms,
                                            exponent=exponent)
