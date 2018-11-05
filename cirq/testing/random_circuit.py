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

from cirq import ops
from cirq.circuits import Circuit, Moment

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


def random_circuit(qubits: Union[Sequence[ops.QubitId], int],
                   n_moments: int,
                   op_density: float,
                   gate_domain: Optional[Dict[ops.Gate, int]] = None
                   ) -> Circuit:
    """Generates a random circuit.

    Args:
        qubits: the qubits that the circuit acts on. Because the qubits on
            which an operation acts are chosen randomly, not all given qubits
            may be acted upon.
        n_moments: the number of moments in the generated circuit.
        op_density: the expected proportion of qubits that are acted on in any
            moment.
        gate_domain: The set of gates to choose from, with a specified arity.

    Raises:
        ValueError:
            * op_density is not in (0, 1).
            * gate_domain is empty.
            * qubits is an int less than 1 or an empty sequence.

    Returns:
        The randomly generated Circuit.
    """
    if not 0 < op_density < 1:
        raise ValueError('op_density must be in (0, 1).')
    if gate_domain is None:
        gate_domain = DEFAULT_GATE_DOMAIN
    if not gate_domain:
        raise ValueError('gate_domain must be non-empty')
    max_arity = max(gate_domain.values())

    if isinstance(qubits, int):
        qubits = tuple(ops.NamedQubit(str(i)) for i in range(qubits))
    n_qubits = len(qubits)
    if n_qubits < 1:
        raise ValueError('At least one qubit must be specified.')

    moments = [] # type: List[Moment]
    for _ in range(n_moments):
        operations = []
        free_qubits = set(q for q in qubits)
        while len(free_qubits) >= max_arity:
            gate, arity = choice(tuple(gate_domain.items()))
            op_qubits = sample(free_qubits, arity)
            free_qubits.difference_update(op_qubits)
            if random() <= op_density:
                operations.append(gate(*op_qubits))
        moments.append(Moment(operations))

    return Circuit(moments)
