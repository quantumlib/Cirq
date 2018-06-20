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

import itertools
import random
from typing import Callable, Iterable

import cirq
from cirq import google


def generate_supremacy_circuit(device: google.XmonDevice, cz_depth: int,
                               seed: int = None, measure: bool = True,
                               ) -> cirq.Circuit:

    randint = random.randint if seed is None else random.Random(seed).randint

    circuit = cirq.Circuit()

    for layer_index in itertools.count():
        if cz_depth <= 0:
            break
        cz_layer = list(_make_cz_layer(device, layer_index))
        if cz_layer:
            circuit.append(_make_random_single_qubit_op_layer(device, randint))
            circuit.append(cz_layer)
            cz_depth -= 1

    circuit.append(_make_random_single_qubit_op_layer(device, randint))
    if measure:
        circuit.append([google.XmonMeasurementGate().on(*device.qubits)])

    return circuit


def _make_random_single_qubit_op_layer(
        device: google.XmonDevice,
        randint: Callable[[int, int], int]) -> Iterable[cirq.Operation]:
    for q in device.qubits:
        angle = randint(0, 3) / 2
        axis = randint(0, 7) / 4
        if angle:
            yield google.ExpWGate(half_turns=angle, axis_half_turns=axis).on(q)


def _make_cz_layer(device: google.XmonDevice, layer_index: int
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
        1│  5│  1│  5│  1│
         ●─4─●─6─●─0─●─2─●─4─. . .
        3│  7│  3│  7│  3│
         ●─0─●─2─●─4─●─6─●─0─. . .
        5│  1│  5│  1│  5│
         ●─4─●─6─●─0─●─2─●─4─. . .
        7│  3│  7│  3│  7│
         ●─0─●─2─●─4─●─6─●─0─. . .
        1│  5│  1│  5│  1│
         .   .   .   .   .   .
         .   .   .   .   .     .
         .   .   .   .   .       .

    Note that, for small devices, some layers will be empty because the layer
    only contains edges not present on the device.
    """

    dir_row = layer_index % 2
    dir_col = 1 - dir_row
    shift = (layer_index >> 1) % 4

    for q in device.qubits:
        q2 = google.XmonQubit(q.row + dir_row, q.col + dir_col)
        if q2 not in device.qubits:
            continue  # This edge isn't on the device.
        if (q.row * (2 - dir_row) + q.col * (2 - dir_col)) % 4 != shift:
            continue  # No CZ along this edge for this layer.

        yield google.Exp11Gate().on(q, q2)
