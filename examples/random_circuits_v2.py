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

import random
from typing import Callable, Iterable, TypeVar

import cirq
from cirq.circuits import InsertStrategy
from cirq.devices.grid_qubit import GridQubit
from cirq.ops import common_gates


def generate_random_v2_circuit(qubits: Iterable[GridQubit],
                               cz_depth: int,
                               seed: int) -> cirq.Circuit:


    ND_GATES = [common_gates.X**(1/2), common_gates.Y**(1/2)]
    rand_gen = random.Random(seed).random

    circuit = cirq.Circuit()

    # Add an initial moment of Hadamards
    circuit.append(common_gates.H(qubit) for qubit in qubits)

    layer_index = 0
    if cz_depth:
        layer_index = _add_cz_layer(layer_index, circuit)
        # In the first moment, add T gates when possible
        for qubit in qubits:
            if not circuit.operation_at(qubit, 1):
                circuit.append(common_gates.T(qubit),
                               strategy=InsertStrategy.EARLIEST)

    for moment_index in range(2, cz_depth+1):
        layer_index = _add_cz_layer(layer_index, circuit)
        # Add single qubit gates in the same moment
        for qubit in qubits:
            if not circuit.operation_at(qubit, moment_index):
                last_gate = circuit.operation_at(qubit, moment_index-1)
                if last_gate:
                    # Add a random non diagonal gate after a CZ
                    if last_gate.gate == common_gates.CZ:
                        circuit.append(_choice(rand_gen, ND_GATES).on(qubit),
                                       strategy=InsertStrategy.EARLIEST)
                    # Add a T gate after a non diagonal gate
                    elif not last_gate.gate == common_gates.T:
                        circuit.append(common_gates.T(qubit),
                                       strategy=InsertStrategy.EARLIEST)

    # Add a final moment of Hadamards
    circuit.append(common_gates.H(qubit) for qubit in qubits)

    return circuit


def generate_random_v2_circuit_grid(n_rows: int, n_cols: int,
                                    cz_depth: int, seed: int
                                    ) -> cirq.Circuit:
    """
    Generates Google Random Circuits v2 as in github.com/sboixo/GRCS cz_v2.

    Args:
        n_rows: number of rows of a 2D lattice.
        n_cols: number of columns.
        cz_depth: number of layers with CZ gates.
        seed: seed for the random instance.

    Returns:
        A circuit corresponding to instance
        inst_{n_rows}x{n_cols}_{cz_depth+1}_{seed}

    The mapping of qubits is cirq.GridQubit(j,k) -> q[j*n_cols+k]
    (as in the QASM mapping)
    """
    qubits = [cirq.GridQubit(i, j) for i in range(n_rows)
              for j in range(n_cols)]
    return generate_random_v2_circuit(qubits, cz_depth, seed)


T = TypeVar('T')
def _choice(rand_gen: Callable[[], float], sequence: Iterable[T]) -> T:
    """
    Choose a random element from a non-empty sequence.

    Use this instead of random.choice, with random.random(), for reproducibility
    """
    return sequence[int(rand_gen() * len(sequence))]


def _add_cz_layer(layer_index: int, circuit: cirq.Circuit) -> int:
    cz_layer = None
    while not cz_layer:
        cz_layer = list(_make_cz_layer(circuit.all_qubits(), layer_index))
        layer_index += 1

    circuit.append(cz_layer)
    return layer_index


def _make_cz_layer(qubits: Iterable[GridQubit], layer_index: int
                   ) -> Iterable[cirq.Operation]:
    """
    Each layer index corresponds to a shift/transpose of this CZ pattern:

        ●───●   ●   ●   ●───●   ●   ● . . .

        ●   ●   ●───●   ●   ●   ●───● . . .

        ●───●   ●   ●   ●───●   ●   ● . . .

        ●   ●   ●───●   ●   ●   ●───● . . .

        ●───●   ●   ●   ●───●   ●   ● . . .

        ●   ●   ●───●   ●   ●   ●───● . . .
        .   .   .   .   .   .   .   . .
        .   .   .   .   .   .   .   .   .
        .   .   .   .   .   .   .   .     .

    Labelled edges, showing the exact index-to-CZs mapping (mod 8):

         ●─0─●─2─●─4─●─6─●─0─. . .
        3│  7│  3│  7│  3│
         ●─4─●─6─●─0─●─2─●─4─. . .
        1│  5│  1│  5│  1│
         ●─0─●─2─●─4─●─6─●─0─. . .
        7│  3│  7│  3│  7│
         ●─4─●─6─●─0─●─2─●─4─. . .
        5│  1│  5│  1│  5│
         ●─0─●─2─●─4─●─6─●─0─. . .
        3│  7│  3│  7│  3│
         .   .   .   .   .   .
         .   .   .   .   .     .
         .   .   .   .   .       .

    Note that, for small devices, some layers will be empty because the layer
    only contains edges not present on the device.

    NOTE: This is the almost the function in supremacy.py,
    but with a different order of CZ layers
    """

    # map to an internal layer index to match the cycle order of public circuits
    LAYER_INDEX_MAP = [0,3,2,1,4,7,6,5]
    internal_layer_index = LAYER_INDEX_MAP[layer_index % 8]

    dir_row = internal_layer_index % 2
    dir_col = 1 - dir_row
    shift = (internal_layer_index >> 1) % 4

    for q in qubits:
        q2 = cirq.GridQubit(q.row + dir_row, q.col + dir_col)
        if q2 not in qubits:
            continue  # This edge isn't on the device.
        if (q.row * (2 - dir_row) + q.col * (2 - dir_col)) % 4 != shift:
            continue  # No CZ along this edge for this layer.

        yield common_gates.CZ(q, q2)
