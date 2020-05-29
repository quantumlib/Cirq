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

from typing import List, Union, Sequence, Dict, Optional, TYPE_CHECKING

from cirq import ops, value
from cirq.circuits import Circuit
from cirq._doc import document

if TYPE_CHECKING:
    import cirq

DEFAULT_GATE_DOMAIN: Dict[ops.Gate, int] = {
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
}
document(
    DEFAULT_GATE_DOMAIN,
    """The default gate domain for `cirq.testing.random_circuit`.

This includes the gates CNOT, CZ, H, ISWAP, CZ, S, SWAP, T, X, Y,
and Z gates.
""")


def random_circuit(qubits: Union[Sequence[ops.Qid], int],
                   n_moments: int,
                   op_density: float,
                   gate_domain: Optional[Dict[ops.Gate, int]] = None,
                   random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
                  ) -> Circuit:
    """Generates a random circuit.

    Args:
        qubits: If a sequence of qubits, then these are the qubits that
            the circuit should act on. Because the qubits on which an
            operation acts are chosen randomly, not all given qubits
            may be acted upon. If an int, then this number of qubits will
            be automatically generated, and the qubits will be
            `cirq.NamedQubits` with names given by the integers in
            `range(qubits)`.
        n_moments: The number of moments in the generated circuit.
        op_density: The probability that a gate is selected to operate on
            randomly selected qubits. Note that this is not the expected number
            of qubits that are acted on, since there are cases where the
            number of qubits that a gate acts on does not evenly divide the
            total number of qubits.
        gate_domain: The set of gates to choose from, specified as a dictionary
            where each key is a gate and the value of the key is the number of
            qubits the gate acts on. If not provided, the default gate domain is
            {X, Y, Z, H, S, T, CNOT, CZ, SWAP, ISWAP, CZPowGate()}. Only gates
            which act on a number of qubits less than len(qubits) (or qubits if
            provided as an int) are selected from the gate domain.
        random_state: Random state or random state seed.

    Raises:
        ValueError:
            * op_density is not in (0, 1].
            * gate_domain is empty.
            * qubits is an int less than 1 or an empty sequence.

    Returns:
        The randomly generated Circuit.
    """
    if not 0 < op_density <= 1:
        raise ValueError(f'op_density must be in (0, 1] but was {op_density}.')
    if gate_domain is None:
        gate_domain = DEFAULT_GATE_DOMAIN
    if not gate_domain:
        raise ValueError('gate_domain must be non-empty.')

    if isinstance(qubits, int):
        qubits = tuple(ops.NamedQubit(str(i)) for i in range(qubits))
    n_qubits = len(qubits)
    if n_qubits < 1:
        raise ValueError('At least one qubit must be specified.')
    gate_domain = {k: v for k, v in gate_domain.items() if v <= n_qubits}
    if not gate_domain:
        raise ValueError(f'After removing gates that act on less that '
                         '{n_qubits}, gate_domain had no gates.')
    max_arity = max(gate_domain.values())

    prng = value.parse_random_state(random_state)

    moments: List[ops.Moment] = []
    gate_arity_pairs = sorted(gate_domain.items(), key=repr)
    num_gates = len(gate_domain)
    for _ in range(n_moments):
        operations = []
        free_qubits = set(qubits)
        while len(free_qubits) >= max_arity:
            gate, arity = gate_arity_pairs[prng.randint(num_gates)]
            op_qubits = prng.choice(sorted(free_qubits),
                                    size=arity,
                                    replace=False)
            free_qubits.difference_update(op_qubits)
            if prng.rand() <= op_density:
                operations.append(gate(*op_qubits))
        moments.append(ops.Moment(operations))

    return Circuit(moments)
