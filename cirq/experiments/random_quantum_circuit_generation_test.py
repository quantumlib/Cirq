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

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
import pytest

import cirq
from cirq.experiments import (
    GridInteractionLayer,
    random_rotations_between_grid_interaction_layers_circuit,
)

SINGLE_QUBIT_LAYER = Dict[cirq.GridQubit, Optional[cirq.Gate]]


def _syc_with_adjacent_z_rotations(
    a: cirq.GridQubit, b: cirq.GridQubit, prng: np.random.RandomState
):
    z_exponents = [prng.uniform(0, 1) for _ in range(4)]
    yield cirq.Z(a) ** z_exponents[0]
    yield cirq.Z(b) ** z_exponents[1]
    yield cirq.google.SYC(a, b)
    yield cirq.Z(a) ** z_exponents[2]
    yield cirq.Z(b) ** z_exponents[3]


@pytest.mark.parametrize(
    'qubits, depth, two_qubit_op_factory, pattern, '
    'single_qubit_gates, add_final_single_qubit_layer, '
    'seed, expected_circuit_length, single_qubit_layers_slice, '
    'two_qubit_layers_slice',
    (
        (
            cirq.GridQubit.rect(4, 3),
            20,
            lambda a, b, _: cirq.google.SYC(a, b),
            cirq.experiments.GRID_STAGGERED_PATTERN,
            (cirq.X ** 0.5, cirq.Y ** 0.5, cirq.Z ** 0.5),
            True,
            1234,
            41,
            slice(None, None, 2),
            slice(1, None, 2),
        ),
        (
            cirq.GridQubit.rect(4, 5),
            21,
            lambda a, b, _: cirq.google.SYC(a, b),
            cirq.experiments.GRID_ALIGNED_PATTERN,
            (cirq.X ** 0.5, cirq.Y ** 0.5, cirq.Z ** 0.5),
            True,
            1234,
            43,
            slice(None, None, 2),
            slice(1, None, 2),
        ),
        (
            cirq.GridQubit.rect(5, 4),
            22,
            _syc_with_adjacent_z_rotations,
            cirq.experiments.GRID_STAGGERED_PATTERN,
            (cirq.X ** 0.5, cirq.Y ** 0.5, cirq.Z ** 0.5),
            True,
            1234,
            89,
            slice(None, None, 4),
            slice(2, None, 4),
        ),
        (
            cirq.GridQubit.rect(5, 5),
            23,
            lambda a, b, _: cirq.google.SYC(a, b),
            cirq.experiments.GRID_ALIGNED_PATTERN,
            (cirq.X ** 0.5, cirq.Y ** 0.5, cirq.Z ** 0.5),
            False,
            1234,
            46,
            slice(None, None, 2),
            slice(1, None, 2),
        ),
        (
            cirq.GridQubit.rect(5, 5),
            24,
            lambda a, b, _: cirq.google.SYC(a, b),
            cirq.experiments.GRID_ALIGNED_PATTERN,
            (cirq.X ** 0.5, cirq.X ** 0.5),
            True,
            1234,
            49,
            slice(None, None, 2),
            slice(1, None, 2),
        ),
    ),
)
def test_random_rotations_between_grid_interaction_layers(
    qubits: Iterable[cirq.GridQubit],
    depth: int,
    two_qubit_op_factory: Callable[
        [cirq.GridQubit, cirq.GridQubit, np.random.RandomState], cirq.OP_TREE
    ],
    pattern: Sequence[GridInteractionLayer],
    single_qubit_gates: Sequence[cirq.Gate],
    add_final_single_qubit_layer: bool,
    seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE',
    expected_circuit_length: int,
    single_qubit_layers_slice: slice,
    two_qubit_layers_slice: slice,
):
    qubits = set(qubits)
    circuit = random_rotations_between_grid_interaction_layers_circuit(
        qubits,
        depth,
        two_qubit_op_factory=two_qubit_op_factory,
        pattern=pattern,
        single_qubit_gates=single_qubit_gates,
        add_final_single_qubit_layer=add_final_single_qubit_layer,
        seed=seed,
    )

    assert len(circuit) == expected_circuit_length
    _validate_single_qubit_layers(
        qubits,
        cast(Sequence[cirq.Moment], circuit[single_qubit_layers_slice]),
        non_repeating_layers=len(set(single_qubit_gates)) > 1,
    )
    _validate_two_qubit_layers(
        qubits, cast(Sequence[cirq.Moment], circuit[two_qubit_layers_slice]), pattern
    )


def test_grid_interaction_layer_repr():
    layer = GridInteractionLayer(col_offset=0, vertical=True, stagger=False)
    assert repr(layer) == (
        'cirq.experiments.GridInteractionLayer(col_offset=0, vertical=True, stagger=False)'
    )


def _validate_single_qubit_layers(
    qubits: Set[cirq.GridQubit], moments: Sequence[cirq.Moment], non_repeating_layers: bool = True
) -> None:
    previous_single_qubit_gates = {q: None for q in qubits}  # type: SINGLE_QUBIT_LAYER

    for moment in moments:
        # All qubits are acted upon
        assert moment.qubits == qubits
        for op in moment:
            # Operation is single-qubit
            assert cirq.num_qubits(op) == 1
            if non_repeating_layers:
                # Gate differs from previous single-qubit gate on this qubit
                q = cast(cirq.GridQubit, op.qubits[0])
                assert op.gate != previous_single_qubit_gates[q]
                previous_single_qubit_gates[q] = op.gate


def _validate_two_qubit_layers(
    qubits: Set[cirq.GridQubit],
    moments: Sequence[cirq.Moment],
    pattern: Sequence[cirq.experiments.GridInteractionLayer],
) -> None:
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
        assert all(
            pair in active_pairs
            for pair in coupled_qubit_pairs
            if pair in pattern[i % len(pattern)]
        )


def _coupled_qubit_pairs(
    qubits: Set['cirq.GridQubit'],
) -> List[Tuple['cirq.GridQubit', 'cirq.GridQubit']]:
    pairs = []
    for qubit in qubits:

        def add_pair(neighbor: 'cirq.GridQubit'):
            if neighbor in qubits:
                pairs.append((qubit, neighbor))

        add_pair(cirq.GridQubit(qubit.row, qubit.col + 1))
        add_pair(cirq.GridQubit(qubit.row + 1, qubit.col))

    return pairs
