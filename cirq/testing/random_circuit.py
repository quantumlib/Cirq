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

from random import choice, sample, random
from typing import Union, Sequence, TYPE_CHECKING, Dict, Optional

from cirq import ops, Simulator
from cirq.circuits import Circuit

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import List

DEFAULT_GATE_DOMAIN = {
    ops.CNOT: 2,
    ops.CZ: 2,
    ops.H: 1,
    ops.ISWAP: 2,
    ops.CZPowGate(): 2,
    ops.S: 1,
    ops.SWAP: 2,
    ops.T: 1,
    ops.X: 1,
    ops.Y: 1,
    ops.Z: 1
}  # type: Dict[ops.Gate, int]


class RandomCircuit:
    def __init__(self,
                 qubits: Union[Sequence[ops.QubitId], int],
                 n_moments: int,
                 op_density: float,
                 gate_domain: Optional[Dict[ops.Gate, int]] = None
    ):
        self.qubits = qubits
        self.n_moments = n_moments
        self.op_density = op_density
        self.gate_domain = gate_domain

    def random_circuit(self) -> Circuit:
        if not 0 < self.op_density < 1:
            raise ValueError('op_density must be in (0, 1).')
        if self.gate_domain is None:
            self.gate_domain = DEFAULT_GATE_DOMAIN
        if not self.gate_domain:
            raise ValueError('gate_domain must be non-empty')
        max_arity = max(self.gate_domain.values())

        if isinstance(self.qubits, int):
            self.qubits = tuple(ops.NamedQubit(str(i)) for i in range(self.qubits))
        n_qubits = len(self.qubits)
        if n_qubits < 1:
            raise ValueError('At least one qubit must be specified.')

        moments = [] # type: List[ops.Moment]
        for _ in range(self.n_moments):
            operations = []
            free_qubits = set(q for q in self.qubits)
            while len(free_qubits) >= max_arity:
                gate, arity = choice(tuple(self.gate_domain.items()))
                op_qubits = sample(free_qubits, arity)
                free_qubits.difference_update(op_qubits)
                if random() <= self.op_density:
                    operations.append(gate(*op_qubits))
            moments.append(ops.Moment(operations))

        return Circuit(moments)

    def random_superposition(self):
        circuit = self.random_circuit(self.qubits, self.n_density,
                                      self.gate_domain)
        return Simulator().simulate(circuit).final_state




