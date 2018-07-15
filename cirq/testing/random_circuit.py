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

from typing import Union, Sequence

from cirq.ops import (
        Gate, SingleQubitGate, TwoQubitGate, ThreeQubitGate)
from cirq import ops
from cirq.circuits import Circuit, Moment
from random import choice, sample, random

DEFAULT_GATE_DOMAIN = (
    ops.CNOT,
    ops.CZ,
    ops.H,
    ops.ISWAP,
    ops.Rot11Gate(),
    ops.S,
    ops.SWAP,
    ops.T,
    ops.X,
    ops.Y,
    ops.Z,
    )

def gate_to_arity(gate: Gate) -> int:
    if isinstance(gate, SingleQubitGate):
        return 1
    if isinstance(gate, TwoQubitGate):
        return 2
    if isinstance(gate, ThreeQubitGate):
        return 3
    raise ValueError('Gates in gate_domain must be instances of ('
                     'SingleQubitGate, TwoQubitGate, ThreeQubitGate).'
                     'Gate {} is not.'.format(gate))

def random_circuit(qubits: Union[Sequence[ops.QubitId], int],
                   n_moments: int,
                   op_density: float,
                   gate_domain: Sequence[Gate]= None
                   ) -> Circuit:
    """Generates a random circuit.

    Args:
        qubits: the qubits that the circuit acts on. Because the qubits on
            which an operation acts are chosen randomly, not all given qubits
            may be acted upon.
        n_moments: 
        op_density: the expected proportion of qubits that are acted on in any
            moment.
        gate_domain: The set of gates to choose from. Gates must be instances
            of (SingleQubitGate, TwoQubitGate, ThreeQubitGate).

    Raises:
        ValueError: 
            * op_density is not in (0, 1).
            * gate_domain contains a gate that is not an instance of
                (SingleQubitGate, TwoQubitGate, ThreeQubitGate).
            * gate_domain is empty.
            * qubits is an int less than 1 or an empty sequence.

    Returns:
        The randomly generated Circuit.
    """
    if not 0 < op_density < 1:
        raise ValueError('op_density must be in (0, 1).')
    if gate_domain is None:
        gate_domain = DEFAULT_GATE_DOMAIN
    max_arity = 0
    arities = (gate_to_arity(gate) for gate in gate_domain)
    for gate, arity in zip(gate_domain, arities):
        max_arity = max(max_arity, arity)
    if not max_arity:
        raise ValueError('gate_domain must be non-empty')


    if isinstance(qubits, int):
        qubits = tuple(ops.NamedQubit(str(i)) for i in range(qubits))
    n_qubits = len(qubits)
    if n_qubits < 1:
        raise ValueError('At least one qubit must be specified.')

    moments = [] # type: Moment
    for _ in range(n_moments):
        operations = []
        free_qubits = set(q for q in qubits)
        while len(free_qubits) >= max_arity:
            gate = choice(gate_domain)
            arity = gate_to_arity(gate)
            op_qubits = sample(free_qubits, arity)
            free_qubits.difference_update(op_qubits)
            if random() <= op_density:
                operations.append(gate(*op_qubits))
        moments.append(Moment(operations))

    return Circuit(moments)


