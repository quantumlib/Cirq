# Copyright 2020 The Cirq Developers
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

from typing import Dict, List, Optional, Sequence, Set, Tuple, cast

import cirq
from cirq.experiments import (
    random_rotations_between_grid_interaction_layers_circuit)


def test_random_rotations_between_grid_interaction_layers():
    # Staggered pattern
    qubits = set(cirq.GridQubit.rect(4, 3))
    depth = 20
    pattern = cirq.experiments.GRID_STAGGERED_PATTERN
    circuit = random_rotations_between_grid_interaction_layers_circuit(
        qubits, depth, seed=1234)

    assert len(circuit) == 2 * depth + 1
    _validate_single_qubit_layers(qubits, circuit[::2])
    _validate_two_qubit_layers(qubits, circuit[1::2], pattern)

    # Aligned pattern
    qubits = set(cirq.GridQubit.rect(4, 5))
    depth = 21
    pattern = cirq.experiments.GRID_ALIGNED_PATTERN
    circuit = random_rotations_between_grid_interaction_layers_circuit(
        qubits, depth, pattern=pattern, seed=1234)

    assert len(circuit) == 2 * depth + 1
    _validate_single_qubit_layers(qubits, circuit[::2])
    _validate_two_qubit_layers(qubits, circuit[1::2], pattern)

    # Staggered pattern with two qubit operation factory
    qubits = set(cirq.GridQubit.rect(5, 4))
    depth = 22
    pattern = cirq.experiments.GRID_STAGGERED_PATTERN

    def two_qubit_op_factory(a, b, prng):
        z_exponents = [prng.uniform(0, 1) for _ in range(4)]
        yield cirq.Z(a)**z_exponents[0]
        yield cirq.Z(b)**z_exponents[1]
        yield cirq.google.SYC(a, b)
        yield cirq.Z(a)**z_exponents[2]
        yield cirq.Z(b)**z_exponents[3]

    circuit = random_rotations_between_grid_interaction_layers_circuit(
        qubits, depth, two_qubit_op_factory=two_qubit_op_factory, seed=1234)

    assert len(circuit) == 4 * depth + 1
    _validate_single_qubit_layers(qubits, circuit[::4])
    _validate_two_qubit_layers(qubits, circuit[2::4], pattern)

    # Aligned pattern omitting final single qubit layer
    qubits = set(cirq.GridQubit.rect(4, 5))
    depth = 23
    pattern = cirq.experiments.GRID_ALIGNED_PATTERN
    circuit = random_rotations_between_grid_interaction_layers_circuit(
        qubits,
        depth,
        pattern=pattern,
        add_final_single_qubit_layer=False,
        seed=1234)

    assert len(circuit) == 2 * depth
    _validate_single_qubit_layers(qubits, circuit[::2])
    _validate_two_qubit_layers(qubits, circuit[1::2], pattern)


def _validate_single_qubit_layers(qubits: Set[cirq.GridQubit],
                                  moments: Sequence[cirq.Moment]) -> None:
    previous_single_qubit_gates = {q: None for q in qubits} \
            # type: Dict[cirq.GridQubit, Optional[cirq.Gate]]

    for moment in moments:
        # All qubits are acted upon
        assert moment.qubits == qubits
        for op in moment:
            # Operation is single-qubit
            assert cirq.num_qubits(op) == 1
            # Gate differs from previous single-qubit gate on this qubit
            q = cast(cirq.GridQubit, op.qubits[0])
            assert op.gate != previous_single_qubit_gates[q]
            previous_single_qubit_gates[q] = op.gate


def _validate_two_qubit_layers(
        qubits: Set[cirq.GridQubit], moments: Sequence[cirq.Moment],
        pattern: Sequence[cirq.experiments.GridInteractionLayer]) -> None:
    coupled_qubit_pairs = _coupled_qubit_pairs(qubits)
    for i, moment in enumerate(moments):
        active_pairs = set()
        for op in moment:
            # Operation is two-qubit
            assert cirq.num_qubits(op) == 2
            # Operation fits pattern
            assert op.qubits in pattern[i % len(pattern)]
            active_pairs.add(op.qubits)
        # All interactions that should be in this layer are present
        assert all(pair in active_pairs
                   for pair in coupled_qubit_pairs
                   if pair in pattern[i % len(pattern)])


def _coupled_qubit_pairs(qubits: Set['cirq.GridQubit'],
                        ) -> List[Tuple['cirq.GridQubit', 'cirq.GridQubit']]:
    pairs = []
    for qubit in qubits:

        def add_pair(neighbor: 'cirq.GridQubit'):
            if neighbor in qubits:
                pairs.append((qubit, neighbor))

        add_pair(cirq.GridQubit(qubit.row, qubit.col + 1))
        add_pair(cirq.GridQubit(qubit.row + 1, qubit.col))

    return pairs
