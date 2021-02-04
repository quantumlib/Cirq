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
from typing import Callable, Iterable, TypeVar, cast, Sequence

from cirq.circuits import InsertStrategy
from cirq import circuits, devices, google, ops


def generate_boixo_2018_supremacy_circuits_v2(
    qubits: Iterable[devices.GridQubit], cz_depth: int, seed: int
) -> circuits.Circuit:
    """
    Generates Google Random Circuits v2 as in github.com/sboixo/GRCS cz_v2.
    See also https://arxiv.org/abs/1807.10749

    Args:
        qubits: qubit grid in which to generate the circuit.
        cz_depth: number of layers with CZ gates.
        seed: seed for the random instance.

    Returns:
        A circuit corresponding to instance
        inst_{n_rows}x{n_cols}_{cz_depth+1}_{seed}

    The mapping of qubits is cirq.GridQubit(j,k) -> q[j*n_cols+k]
    (as in the QASM mapping)
    """

    non_diagonal_gates = [ops.pauli_gates.X ** (1 / 2), ops.pauli_gates.Y ** (1 / 2)]
    rand_gen = random.Random(seed).random

    circuit = circuits.Circuit()

    # Add an initial moment of Hadamards
    circuit.append(ops.common_gates.H(qubit) for qubit in qubits)

    layer_index = 0
    if cz_depth:
        layer_index = _add_cz_layer(layer_index, circuit)
        # In the first moment, add T gates when possible
        for qubit in qubits:
            if not circuit.operation_at(qubit, 1):
                circuit.append(ops.common_gates.T(qubit), strategy=InsertStrategy.EARLIEST)

    for moment_index in range(2, cz_depth + 1):
        layer_index = _add_cz_layer(layer_index, circuit)
        # Add single qubit gates in the same moment
        for qubit in qubits:
            if not circuit.operation_at(qubit, moment_index):
                last_op = circuit.operation_at(qubit, moment_index - 1)
                if last_op:
                    gate = cast(ops.GateOperation, last_op).gate
                    # Add a random non diagonal gate after a CZ
                    if gate == ops.CZ:
                        circuit.append(
                            _choice(rand_gen, non_diagonal_gates).on(qubit),
                            strategy=InsertStrategy.EARLIEST,
                        )
                    # Add a T gate after a non diagonal gate
                    elif not gate == ops.T:
                        circuit.append(ops.common_gates.T(qubit), strategy=InsertStrategy.EARLIEST)

    # Add a final moment of Hadamards
    circuit.append(
        [ops.common_gates.H(qubit) for qubit in qubits], strategy=InsertStrategy.NEW_THEN_INLINE
    )

    return circuit


def generate_boixo_2018_supremacy_circuits_v2_grid(
    n_rows: int, n_cols: int, cz_depth: int, seed: int
) -> circuits.Circuit:
    """
    Generates Google Random Circuits v2 as in github.com/sboixo/GRCS cz_v2.
    See also https://arxiv.org/abs/1807.10749

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
    qubits = [devices.GridQubit(i, j) for i in range(n_rows) for j in range(n_cols)]
    return generate_boixo_2018_supremacy_circuits_v2(qubits, cz_depth, seed)


def generate_boixo_2018_supremacy_circuits_v2_bristlecone(
    n_rows: int, cz_depth: int, seed: int
) -> circuits.Circuit:
    """
    Generates Google Random Circuits v2 in Bristlecone.
    See also https://arxiv.org/abs/1807.10749

    Args:
        n_rows: number of rows in a Bristlecone lattice.
          Note that we do not include single qubit corners.
        cz_depth: number of layers with CZ gates.
        seed: seed for the random instance.

    Returns:
        A circuit with given size and seed.
    """

    def get_qubits(n_rows):
        def count_neighbors(qubits, qubit):
            """Counts the qubits that the given qubit can interact with."""
            possibles = [
                devices.GridQubit(qubit.row + 1, qubit.col),
                devices.GridQubit(qubit.row - 1, qubit.col),
                devices.GridQubit(qubit.row, qubit.col + 1),
                devices.GridQubit(qubit.row, qubit.col - 1),
            ]
            return len(list(e for e in possibles if e in qubits))

        assert 2 <= n_rows <= 11
        max_row = n_rows - 1
        dev = google.Bristlecone
        # we need a consistent order of qubits
        qubits = list(dev.qubits)
        qubits.sort()
        qubits = [
            q
            for q in qubits
            if q.row <= max_row and q.row + q.col < n_rows + 6 and q.row - q.col < n_rows - 5
        ]
        qubits = [q for q in qubits if count_neighbors(qubits, q) > 1]
        return qubits

    qubits = get_qubits(n_rows)
    return generate_boixo_2018_supremacy_circuits_v2(qubits, cz_depth, seed)


T = TypeVar('T')


def _choice(rand_gen: Callable[[], float], sequence: Sequence[T]) -> T:
    """Choose a random element from a non-empty sequence.

    Use this instead of random.choice, with random.random(), for reproducibility
    """
    return sequence[int(rand_gen() * len(sequence))]


def _add_cz_layer(layer_index: int, circuit: circuits.Circuit) -> int:
    cz_layer = None
    while not cz_layer:
        qubits = cast(Iterable[devices.GridQubit], circuit.all_qubits())
        cz_layer = list(_make_cz_layer(qubits, layer_index))
        layer_index += 1

    circuit.append(cz_layer, strategy=InsertStrategy.NEW_THEN_INLINE)
    return layer_index


def _make_cz_layer(
    qubits: Iterable[devices.GridQubit], layer_index: int
) -> Iterable[ops.Operation]:
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
    """

    # map to an internal layer index to match the cycle order of public circuits
    layer_index_map = [0, 3, 2, 1, 4, 7, 6, 5]
    internal_layer_index = layer_index_map[layer_index % 8]

    dir_row = internal_layer_index % 2
    dir_col = 1 - dir_row
    shift = (internal_layer_index >> 1) % 4

    for q in qubits:
        q2 = devices.GridQubit(q.row + dir_row, q.col + dir_col)
        if q2 not in qubits:
            continue  # This edge isn't on the device.
        if (q.row * (2 - dir_row) + q.col * (2 - dir_col)) % 4 != shift:
            continue  # No CZ along this edge for this layer.

        yield ops.common_gates.CZ(q, q2)
