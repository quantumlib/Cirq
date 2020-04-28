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
"""Code for generating random quantum circuits."""

from typing import (Any, Callable, Container, Dict, Iterable, List, Sequence,
                    TYPE_CHECKING, Tuple, Union)

import dataclasses

from cirq import circuits, devices, google, ops, protocols, value
from cirq._doc import document

if TYPE_CHECKING:
    import numpy as np
    import cirq


@dataclasses.dataclass(frozen=True)
class GridInteractionLayer(
        Container[Tuple[devices.GridQubit, devices.GridQubit]]):
    """A layer of aligned or staggered two-qubit interactions on a grid.

    Layers of this type have two different basic structures,
    aligned:

    *-* *-* *-*
    *-* *-* *-*
    *-* *-* *-*
    *-* *-* *-*
    *-* *-* *-*
    *-* *-* *-*

    and staggered:

    *-* *-* *-*
    * *-* *-* *
    *-* *-* *-*
    * *-* *-* *
    *-* *-* *-*
    * *-* *-* *

    Other variants are obtained by offsetting these lattices to the right by
    some number of columns, and/or transposing into the vertical orientation.
    There are a total of 4 aligned and 4 staggered variants.

    The 2x2 unit cells for the aligned and staggered versions of this layer
    are, respectively:

    *-*
    *-*

    and

    *-*
    * *-

    with left/top qubits at (0, 0) and (1, 0) in the aligned case, or
    (0, 0) and (1, 1) in the staggered case. Other variants have the same unit
    cells after transposing and offsetting.

    Args:
        col_offset: Number of columns by which to shift the basic lattice.
        vertical: Whether gates should be oriented vertically rather than
            horizontally.
        stagger: Whether to stagger gates in neighboring rows.
    """
    col_offset: int = 0
    vertical: bool = False
    stagger: bool = False

    def __contains__(self, pair) -> bool:
        """Checks whether a pair is in this layer."""
        if self.vertical:
            # Transpose row, col coords for vertical orientation.
            a, b = pair
            pair = devices.GridQubit(a.col,
                                     a.row), devices.GridQubit(b.col, b.row)

        a, b = sorted(pair)

        # qubits should be 1 column apart.
        if (a.row != b.row) or (b.col != a.col + 1):
            return False

        # mod to get the position in the 2 x 2 unit cell with column offset.
        pos = a.row % 2, (a.col - self.col_offset) % 2
        return pos == (0, 0) or pos == (1, self.stagger)

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(
            self, ['col_offset', 'vertical', 'stagger'])

    def __repr__(self) -> str:
        return ('cirq.experiments.GridInteractionLayer('
                f'col_offset={self.col_offset}, '
                f'vertical={self.vertical}, '
                f'stagger={self.stagger})')


GRID_STAGGERED_PATTERN = (
    GridInteractionLayer(col_offset=0, vertical=True, stagger=True),  # A
    GridInteractionLayer(col_offset=1, vertical=True, stagger=True),  # B
    GridInteractionLayer(col_offset=1, vertical=False, stagger=True),  # C
    GridInteractionLayer(col_offset=0, vertical=False, stagger=True),  # D
    GridInteractionLayer(col_offset=1, vertical=False, stagger=True),  # C
    GridInteractionLayer(col_offset=0, vertical=False, stagger=True),  # D
    GridInteractionLayer(col_offset=0, vertical=True, stagger=True),  # A
    GridInteractionLayer(col_offset=1, vertical=True, stagger=True),  # B
)
document(
    GRID_STAGGERED_PATTERN,
    """A pattern of two-qubit gates that is hard to simulate.

    This pattern of gates was used in the paper
    https://www.nature.com/articles/s41586-019-1666-5
    to demonstrate quantum supremacy.
    """)

GRID_ALIGNED_PATTERN = (
    GridInteractionLayer(col_offset=0, vertical=False, stagger=False),  # E
    GridInteractionLayer(col_offset=1, vertical=False, stagger=False),  # F
    GridInteractionLayer(col_offset=0, vertical=True, stagger=False),  # G
    GridInteractionLayer(col_offset=1, vertical=True, stagger=False),  # H
)
document(
    GRID_ALIGNED_PATTERN,
    """A pattern of two-qubit gates that is easy to simulate.

    This pattern of gates was used in the paper
    https://www.nature.com/articles/s41586-019-1666-5
    to evaluate the performance of a quantum computer.
    """)


def random_rotations_between_grid_interaction_layers_circuit(
        qubits: Iterable['cirq.GridQubit'],
        depth: int,
        *,  # forces keyword arguments
        two_qubit_op_factory: Callable[[
            'cirq.GridQubit', 'cirq.GridQubit', 'np.random.RandomState'
        ], 'cirq.OP_TREE'] = lambda a, b, _: google.SYC(a, b),
        pattern: Sequence[GridInteractionLayer] = GRID_STAGGERED_PATTERN,
        single_qubit_gates: Sequence['cirq.Gate'] = (ops.X**0.5, ops.Y**0.5,
                                                     ops.PhasedXPowGate(
                                                         phase_exponent=0.25,
                                                         exponent=0.5)),
        add_final_single_qubit_layer: bool = True,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> 'cirq.Circuit':
    """Generate a random quantum circuit.

    This construction is based on the circuits used in the paper
    https://www.nature.com/articles/s41586-019-1666-5.

    The generated circuit consists of a number of "cycles", this number being
    specified by `depth`. Each cycle is actually composed of two sub-layers:
    a layer of single-qubit gates followed by a layer of two-qubit gates.
    The single-qubit gates are chosen randomly from the gates specified by
    `single_qubit_gates`, but with the constraint that no qubit is acted upon
    by the same single-qubit gate in consecutive cycles. In the layer of
    two-qubit gates, which pairs of qubits undergo interaction is determined
    by `pattern`, which is a sequence of two-qubit interaction sets. The
    set of interactions in a two-qubit layer rotates through this sequence.
    The two-qubit operations themselves are generated by the call
    `two_qubit_op_factory(a, b, prng)`, where `a` and `b` are the qubits and
    `prng` is the pseudorandom number generator.

    At the end of the circuit, an additional layer of single-qubit gates is
    appended, subject to the same constraint regarding consecutive cycles
    described above.

    If only one choice of single-qubit gate is given, then the constraint
    that excludes repeating single-qubit gates in consecutive cycles is not
    enforced.

    Args:
        qubits: The qubits to use.
        depth: The number of cycles.
        two_qubit_op_factory: A factory to generate two-qubit operations.
            These operations will be generated with calls of the form
            `two_qubit_op_factory(a, b, prng)`, where `a` and `b` are the qubits
            to operate on and `prng` is the pseudorandom number generator.
        pattern: The pattern of grid interaction layers to use.
        single_qubit_gates: The single-qubit gates to use.
        add_final_single_qubit_layer: Whether to include a final layer of
            single-qubit gates after the last cycle.
        seed: A seed or random state to use for the pseudorandom number
            generator.
    """
    prng = value.parse_random_state(seed)
    qubits = list(qubits)
    coupled_qubit_pairs = _coupled_qubit_pairs(qubits)

    circuit = circuits.Circuit()
    previous_single_qubit_layer = {}  # type: Dict[cirq.GridQubit, cirq.Gate]
    if len(set(single_qubit_gates)) == 1:
        single_qubit_layer_factory = _FixedSingleQubitLayerFactory({
            q: single_qubit_gates[0] for q in qubits
        })  # type: _SingleQubitLayerFactory

    else:
        single_qubit_layer_factory = _RandomSingleQubitLayerFactory(
            qubits, single_qubit_gates, prng)
    for i in range(depth):
        single_qubit_layer = single_qubit_layer_factory.new_layer(
            previous_single_qubit_layer)
        two_qubit_layer = _two_qubit_layer(coupled_qubit_pairs,
                                           two_qubit_op_factory,
                                           pattern[i % len(pattern)], prng)
        circuit.append([g.on(q) for q, g in single_qubit_layer.items()],
                       strategy=circuits.InsertStrategy.NEW_THEN_INLINE)
        circuit.append(two_qubit_layer,
                       strategy=circuits.InsertStrategy.EARLIEST)
        previous_single_qubit_layer = single_qubit_layer

    if add_final_single_qubit_layer:
        final_single_qubit_layer = single_qubit_layer_factory.new_layer(
            previous_single_qubit_layer)
        circuit.append([g.on(q) for q, g in final_single_qubit_layer.items()],
                       strategy=circuits.InsertStrategy.NEW_THEN_INLINE)

    return circuit


def _coupled_qubit_pairs(qubits: List['cirq.GridQubit'],
                        ) -> List[Tuple['cirq.GridQubit', 'cirq.GridQubit']]:
    pairs = []
    qubit_set = set(qubits)
    for qubit in qubits:

        def add_pair(neighbor: 'cirq.GridQubit'):
            if neighbor in qubit_set:
                pairs.append((qubit, neighbor))

        add_pair(devices.GridQubit(qubit.row, qubit.col + 1))
        add_pair(devices.GridQubit(qubit.row + 1, qubit.col))

    return pairs


class _RandomSingleQubitLayerFactory:

    def __init__(self, qubits: List['cirq.GridQubit'],
                 single_qubit_gates: Sequence['cirq.Gate'],
                 prng: 'np.random.RandomState') -> None:
        self.qubits = qubits
        self.single_qubit_gates = single_qubit_gates
        self.prng = prng

    def new_layer(
            self,
            previous_single_qubit_layer: Dict['cirq.GridQubit', 'cirq.Gate']
    ) -> Dict['cirq.GridQubit', 'cirq.Gate']:

        def random_gate(qubit: 'cirq.GridQubit') -> 'cirq.Gate':
            excluded_gate = previous_single_qubit_layer.get(qubit, None)
            g = self.single_qubit_gates[self.prng.randint(
                0, len(self.single_qubit_gates))]
            while g == excluded_gate:
                g = self.single_qubit_gates[self.prng.randint(
                    0, len(self.single_qubit_gates))]
            return g

        return {q: random_gate(q) for q in self.qubits}


class _FixedSingleQubitLayerFactory:

    def __init__(self,
                 fixed_single_qubit_layer: Dict['cirq.GridQubit', 'cirq.Gate']
                ) -> None:
        self.fixed_single_qubit_layer = fixed_single_qubit_layer

    def new_layer(
            self,
            previous_single_qubit_layer: Dict['cirq.GridQubit', 'cirq.Gate']
    ) -> Dict['cirq.GridQubit', 'cirq.Gate']:
        return self.fixed_single_qubit_layer


_SingleQubitLayerFactory = Union[_FixedSingleQubitLayerFactory,
                                 _RandomSingleQubitLayerFactory]


def _two_qubit_layer(
        coupled_qubit_pairs: List[Tuple['cirq.GridQubit', 'cirq.GridQubit']],
        two_qubit_op_factory: Callable[[
            'cirq.GridQubit', 'cirq.GridQubit', 'np.random.RandomState'
        ], 'cirq.OP_TREE'], layer: GridInteractionLayer,
        prng: 'np.random.RandomState') -> 'cirq.OP_TREE':
    for a, b in coupled_qubit_pairs:
        if (a, b) in layer:
            yield two_qubit_op_factory(a, b, prng)
