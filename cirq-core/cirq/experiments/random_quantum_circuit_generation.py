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

import dataclasses
import itertools
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Iterable,
    List,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    Union,
    Optional,
    cast,
    Iterator,
)

import networkx as nx
import numpy as np

from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document

if TYPE_CHECKING:
    import cirq

QidPairT = Tuple['cirq.Qid', 'cirq.Qid']
GridQubitPairT = Tuple['cirq.GridQubit', 'cirq.GridQubit']


@dataclasses.dataclass(frozen=True)
class GridInteractionLayer(Container[GridQubitPairT]):
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
            pair = devices.GridQubit(a.col, a.row), devices.GridQubit(b.col, b.row)

        a, b = sorted(pair)

        # qubits should be 1 column apart.
        if (a.row != b.row) or (b.col != a.col + 1):
            return False

        # mod to get the position in the 2 x 2 unit cell with column offset.
        pos = a.row % 2, (a.col - self.col_offset) % 2
        return pos == (0, 0) or pos == (1, self.stagger)

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['col_offset', 'vertical', 'stagger'])

    def __repr__(self) -> str:
        return (
            'cirq.experiments.GridInteractionLayer('
            f'col_offset={self.col_offset}, '
            f'vertical={self.vertical}, '
            f'stagger={self.stagger})'
        )


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
    """,
)

HALF_GRID_STAGGERED_PATTERN = (
    GridInteractionLayer(col_offset=0, vertical=True, stagger=True),  # A
    GridInteractionLayer(col_offset=1, vertical=True, stagger=True),  # B
    GridInteractionLayer(col_offset=1, vertical=False, stagger=True),  # C
    GridInteractionLayer(col_offset=0, vertical=False, stagger=True),  # D
)
document(
    HALF_GRID_STAGGERED_PATTERN,
    """A pattern that is half of GRID_STAGGERED_PATTERN.

    It activates each link in a grid once in a staggered way permits
    easier simulation.
    """,
)

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
    """,
)


def random_rotations_between_two_qubit_circuit(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    depth: int,
    two_qubit_op_factory: Callable[
        ['cirq.Qid', 'cirq.Qid', 'np.random.RandomState'], 'cirq.OP_TREE'
    ] = lambda a, b, _: ops.CZPowGate()(a, b),
    single_qubit_gates: Sequence['cirq.Gate'] = (
        ops.X**0.5,
        ops.Y**0.5,
        ops.PhasedXPowGate(phase_exponent=0.25, exponent=0.5),
    ),
    add_final_single_qubit_layer: bool = True,
    seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> 'cirq.Circuit':
    """Generate a random two-qubit quantum circuit.

    This construction uses a similar structure to those in the paper
    https://www.nature.com/articles/s41586-019-1666-5.

    The generated circuit consists of a number of "cycles", this number being
    specified by `depth`. Each cycle is actually composed of two sub-layers:
    a layer of single-qubit gates followed by a layer of two-qubit gates,
    controlled by their respective arguments, see below.

    Args:
        q0: The first qubit
        q1: The second qubit
        depth: The number of cycles.
        two_qubit_op_factory: A callable that returns a two-qubit operation.
            These operations will be generated with calls of the form
            `two_qubit_op_factory(q0, q1, prng)`, where `prng` is the
            pseudorandom number generator.
        single_qubit_gates: Single-qubit gates are selected randomly from this
            sequence. No qubit is acted upon by the same single-qubit gate in
            consecutive cycles. If only one choice of single-qubit gate is
            given, then this constraint is not enforced.
        add_final_single_qubit_layer: Whether to include a final layer of
            single-qubit gates after the last cycle (subject to the same
            non-consecutivity constraint).
        seed: A seed or random state to use for the pseudorandom number
            generator.
    """
    prng = value.parse_random_state(seed)

    circuit = circuits.Circuit()
    previous_single_qubit_layer = circuits.Moment()
    single_qubit_layer_factory = _single_qubit_gates_arg_to_factory(
        single_qubit_gates=single_qubit_gates, qubits=(q0, q1), prng=prng
    )

    for _ in range(depth):
        single_qubit_layer = single_qubit_layer_factory.new_layer(previous_single_qubit_layer)
        circuit += single_qubit_layer
        circuit += two_qubit_op_factory(q0, q1, prng)
        previous_single_qubit_layer = single_qubit_layer

    if add_final_single_qubit_layer:
        circuit += single_qubit_layer_factory.new_layer(previous_single_qubit_layer)

    return circuit


def generate_library_of_2q_circuits(
    n_library_circuits: int,
    two_qubit_gate: 'cirq.Gate',
    *,
    max_cycle_depth: int = 100,
    q0: 'cirq.Qid' = devices.LineQubit(0),
    q1: 'cirq.Qid' = devices.LineQubit(1),
    random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> List['cirq.Circuit']:
    """Generate a library of two-qubit Circuits.

    For single-qubit gates, this uses PhasedXZGates where the axis-in-XY-plane is one
    of eight eighth turns and the Z rotation angle is one of eight eighth turns. This
    provides 8*8=64 total choices, each implementable with one PhasedXZGate. This is
    appropriate for architectures with microwave single-qubit control.

    Args:
        n_library_circuits: The number of circuits to generate.
        two_qubit_gate: The two qubit gate to use in the circuits.
        max_cycle_depth: The maximum cycle_depth in the circuits to generate. If you are using XEB,
            this must be greater than or equal to the maximum value in `cycle_depths`.
        q0: The first qubit to use when constructing the circuits.
        q1: The second qubit to use when constructing the circuits
        random_state: A random state or seed used to deterministically sample the random circuits.
    """
    rs = value.parse_random_state(random_state)
    exponents = np.linspace(0, 7 / 4, 8)
    single_qubit_gates = [
        ops.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a)
        for a, z in itertools.product(exponents, repeat=2)
    ]
    return [
        random_rotations_between_two_qubit_circuit(
            q0,
            q1,
            depth=max_cycle_depth,
            two_qubit_op_factory=lambda a, b, _: two_qubit_gate(a, b),
            single_qubit_gates=single_qubit_gates,
            seed=rs,
        )
        for _ in range(n_library_circuits)
    ]


def _get_active_pairs(graph: nx.Graph, grid_layer: GridInteractionLayer):
    """Extract pairs of qubits from a device graph and a GridInteractionLayer."""
    for edge in graph.edges:
        if edge in grid_layer:
            yield edge


@dataclasses.dataclass(frozen=True)
class CircuitLibraryCombination:
    """For a given layer (specifically, a set of pairs of qubits), `combinations` is a 2d array
    of shape (n_combinations, len(pairs)) where each row represents a combination (with replacement)
    of two-qubit circuits. The actual values are indices into a list of library circuits.

    `layer` is used for record-keeping. This is the GridInteractionLayer if using
    `get_random_combinations_for_device`, the Moment if using
    `get_random_combinations_for_layer_circuit` and ommitted if using
    `get_random_combinations_for_pairs`.
    """

    layer: Optional[Any]
    combinations: np.array
    pairs: List[QidPairT]


def _get_random_combinations(
    n_library_circuits: int,
    n_combinations: int,
    *,
    pair_gen: Iterator[Tuple[List[QidPairT], Any]],
    random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> List[CircuitLibraryCombination]:
    """For qubit pairs, prepare a set of combinations to efficiently sample
    parallel two-qubit XEB circuits.

    This helper function should be called by one of
    `get_random_comibations_for_device`,
    `get_random_combinations_for_layer_circuit`, or
    `get_random_combinations_for_pairs` which define
    appropriate `pair_gen` arguments.

    Args:
        n_library_circuits: The number of circuits in your library. Likely the value
            passed to `generate_library_of_2q_circuits`.
        n_combinations: The number of combinations (with replacement) to generate
            using the library circuits. Since this function returns a
            `CircuitLibraryCombination`, the combinations will be represented
            by indexes between 0 and `n_library_circuits-1` instead of the circuits
            themselves. The more combinations, the more precise of an estimate for XEB
            fidelity estimation, but a corresponding increase in the number of circuits
            you must sample.
        pair_gen: A generator that yields tuples of (pairs, layer_meta) where pairs is a list
            of qubit pairs and layer_meta is additional data describing the "layer" assigned
            to the CircuitLibraryCombination.layer field.
        random_state: A random-state-like object to seed the random combination generation.

    Returns:
        A list of `CircuitLibraryCombination`, each corresponding to a layer
        generated from `pair_gen`. Each object has a `combinations` matrix of circuit
        indices of shape `(n_combinations, len(pairs))`. This
        returned list can be provided to `sample_2q_xeb_circuits` to efficiently
        sample parallel XEB circuits.
    """
    rs = value.parse_random_state(random_state)

    combinations_by_layer = []
    for pairs, layer in pair_gen:
        combinations = rs.randint(0, n_library_circuits, size=(n_combinations, len(pairs)))
        combinations_by_layer.append(
            CircuitLibraryCombination(layer=layer, combinations=combinations, pairs=pairs)
        )
    return combinations_by_layer


def get_random_combinations_for_device(
    n_library_circuits: int,
    n_combinations: int,
    device_graph: nx.Graph,
    *,
    pattern: Sequence[GridInteractionLayer] = HALF_GRID_STAGGERED_PATTERN,
    random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> List[CircuitLibraryCombination]:
    """For a given device, prepare a set of combinations to efficiently sample
    parallel two-qubit XEB circuits.

    Args:
        n_library_circuits: The number of circuits in your library. Likely the value
            passed to `generate_library_of_2q_circuits`.
        n_combinations: The number of combinations (with replacement) to generate
            using the library circuits. Since this function returns a
            `CircuitLibraryCombination`, the combinations will be represented
            by indexes between 0 and `n_library_circuits-1` instead of the circuits
            themselves. The more combinations, the more precise of an estimate for XEB
            fidelity estimation, but a corresponding increase in the number of circuits
            you must sample.
        device_graph: A graph whose nodes are qubits and whose edges represent
            the possibility of doing a two-qubit gate. This combined with the
            `pattern` argument determines which two qubit pairs are activated
            when.
        pattern: A sequence of `GridInteractionLayer`, each of which has
            a particular set of qubits that are activated simultaneously. These
            pairs of qubits are deduced by combining this argument with `device_graph`.
        random_state: A random-state-like object to seed the random combination generation.

    Returns:
        A list of `CircuitLibraryCombination`, each corresponding to an interaction
        layer in `pattern` where there is a non-zero number of pairs which would be activated.
        Each object has a `combinations` matrix of circuit
        indices of shape `(n_combinations, len(pairs))` where `len(pairs)` may
        be different for each entry (i.e. for each layer in `pattern`). This
        returned list can be provided to `sample_2q_xeb_circuits` to efficiently
        sample parallel XEB circuits.
    """

    def pair_gen():
        for layer in pattern:
            pairs = sorted(_get_active_pairs(device_graph, layer))
            if len(pairs) == 0:
                continue

            yield pairs, layer

    return _get_random_combinations(
        n_library_circuits=n_library_circuits,
        n_combinations=n_combinations,
        random_state=random_state,
        pair_gen=pair_gen(),
    )


def get_random_combinations_for_pairs(
    n_library_circuits: int,
    n_combinations: int,
    all_pairs: List[List[QidPairT]],
    random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> List[CircuitLibraryCombination]:
    """For an explicit nested list of pairs, prepare a set of combinations to efficiently sample
    parallel two-qubit XEB circuits.

    Args:
        n_library_circuits: The number of circuits in your library. Likely the value
            passed to `generate_library_of_2q_circuits`.
        n_combinations: The number of combinations (with replacement) to generate
            using the library circuits. Since this function returns a
            `CircuitLibraryCombination`, the combinations will be represented
            by indexes between 0 and `n_library_circuits-1` instead of the circuits
            themselves. The more combinations, the more precise of an estimate for XEB
            fidelity estimation, but a corresponding increase in the number of circuits
            you must sample.
        all_pairs: A nested list of qubit pairs. The outer list should represent a "layer"
            where the inner pairs should all be able to be activated simultaneously.
        random_state: A random-state-like object to seed the random combination generation.

    Returns:
        A list of `CircuitLibraryCombination`, each corresponding to an interaction
        layer the outer list of `all_pairs`. Each object has a `combinations` matrix of circuit
        indices of shape `(n_combinations, len(pairs))` where `len(pairs)` may
        be different for each entry. This
        returned list can be provided to `sample_2q_xeb_circuits` to efficiently
        sample parallel XEB circuits.
    """

    def pair_gen():
        for pairs in all_pairs:
            yield pairs, None

    return _get_random_combinations(
        n_library_circuits=n_library_circuits,
        n_combinations=n_combinations,
        random_state=random_state,
        pair_gen=pair_gen(),
    )


def _pairs_from_moment(moment: 'cirq.Moment') -> List[QidPairT]:
    """Helper function in `get_random_combinations_for_layer_circuit` pair generator.

    The moment should contain only two qubit operations, which define a list of qubit pairs.
    """
    pairs: List[QidPairT] = []
    for op in moment.operations:
        if len(op.qubits) != 2:
            raise ValueError("Layer circuit contains non-2-qubit operations.")
        qpair = cast(QidPairT, op.qubits)
        pairs.append(qpair)
    return pairs


def get_random_combinations_for_layer_circuit(
    n_library_circuits: int,
    n_combinations: int,
    layer_circuit: 'cirq.Circuit',
    random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> List[CircuitLibraryCombination]:
    """For a layer circuit, prepare a set of combinations to efficiently sample
    parallel two-qubit XEB circuits.

    Args:
        n_library_circuits: The number of circuits in your library. Likely the value
            passed to `generate_library_of_2q_circuits`.
        n_combinations: The number of combinations (with replacement) to generate
            using the library circuits. Since this function returns a
            `CircuitLibraryCombination`, the combinations will be represented
            by indexes between 0 and `n_library_circuits-1` instead of the circuits
            themselves. The more combinations, the more precise of an estimate for XEB
            fidelity estimation, but a corresponding increase in the number of circuits
            you must sample.
        layer_circuit: A calibration-style circuit where each Moment represents a layer.
            Two qubit operations indicate the pair should be activated. This circuit should
            only contain Moments which only contain two-qubit operations.
        random_state: A random-state-like object to seed the random combination generation.

    Returns:
        A list of `CircuitLibraryCombination`, each corresponding to a moment in `layer_circuit`.
        Each object has a `combinations` matrix of circuit
        indices of shape `(n_combinations, len(pairs))` where `len(pairs)` may
        be different for each entry (i.e. for moment). This
        returned list can be provided to `sample_2q_xeb_circuits` to efficiently
        sample parallel XEB circuits.
    """

    def pair_gen():
        for moment in layer_circuit.moments:
            yield _pairs_from_moment(moment), moment

    return _get_random_combinations(
        n_library_circuits=n_library_circuits,
        n_combinations=n_combinations,
        random_state=random_state,
        pair_gen=pair_gen(),
    )


def get_grid_interaction_layer_circuit(
    device_graph: nx.Graph,
    pattern: Sequence[GridInteractionLayer] = HALF_GRID_STAGGERED_PATTERN,
    two_qubit_gate=ops.ISWAP**0.5,
) -> 'cirq.Circuit':
    """Create a circuit representation of a grid interaction pattern on a given device topology.

    The resulting circuit is deterministic, of depth len(pattern), and consists of `two_qubit_gate`
    applied to each pair in `pattern` restricted to available connections in `device_graph`.

    Args:
        device_graph: A graph whose nodes are qubits and whose edges represent the possibility of
            doing a two-qubit gate. This combined with the `pattern` argument determines which
            two qubit pairs are activated when.
        pattern: A sequence of `GridInteractionLayer`, each of which has a particular set of
            qubits that are activated simultaneously. These pairs of qubits are deduced by
            combining this argument with `device_graph`.
        two_qubit_gate: The two qubit gate to use in constructing the circuit layers.
    """
    moments = []
    for layer in pattern:
        pairs = sorted(_get_active_pairs(device_graph, layer))
        if len(pairs) == 0:
            continue
        moments += [circuits.Moment(two_qubit_gate.on(*pair) for pair in pairs)]
    return circuits.Circuit(moments)


def random_rotations_between_grid_interaction_layers_circuit(
    qubits: Iterable['cirq.GridQubit'],
    depth: int,
    *,  # forces keyword arguments
    two_qubit_op_factory: Callable[
        ['cirq.GridQubit', 'cirq.GridQubit', 'np.random.RandomState'], 'cirq.OP_TREE'
    ] = lambda a, b, _: ops.CZPowGate()(a, b),
    pattern: Sequence[GridInteractionLayer] = GRID_STAGGERED_PATTERN,
    single_qubit_gates: Sequence['cirq.Gate'] = (
        ops.X**0.5,
        ops.Y**0.5,
        ops.PhasedXPowGate(phase_exponent=0.25, exponent=0.5),
    ),
    add_final_single_qubit_layer: bool = True,
    seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> 'cirq.Circuit':
    """Generate a random quantum circuit of a particular form.

    This construction is based on the circuits used in the paper
    https://www.nature.com/articles/s41586-019-1666-5.

    The generated circuit consists of a number of "cycles", this number being
    specified by `depth`. Each cycle is actually composed of two sub-layers:
    a layer of single-qubit gates followed by a layer of two-qubit gates,
    controlled by their respective arguments, see below. The pairs of qubits
    in a given entangling layer is controlled by the `pattern` argument,
    see below.

    Args:
        qubits: The qubits to use.
        depth: The number of cycles.
        two_qubit_op_factory: A callable that returns a two-qubit operation.
            These operations will be generated with calls of the form
            `two_qubit_op_factory(q0, q1, prng)`, where `prng` is the
            pseudorandom number generator.
        pattern: A sequence of GridInteractionLayers, each of which determine
            which pairs of qubits are entangled. The layers in a pattern are
            iterated through sequentially, repeating until `depth` is reached.
        single_qubit_gates: Single-qubit gates are selected randomly from this
            sequence. No qubit is acted upon by the same single-qubit gate in
            consecutive cycles. If only one choice of single-qubit gate is
            given, then this constraint is not enforced.
        add_final_single_qubit_layer: Whether to include a final layer of
            single-qubit gates after the last cycle.
        seed: A seed or random state to use for the pseudorandom number
            generator.
    """
    prng = value.parse_random_state(seed)
    qubits = list(qubits)
    coupled_qubit_pairs = _coupled_qubit_pairs(qubits)

    circuit = circuits.Circuit()
    previous_single_qubit_layer = circuits.Moment()
    single_qubit_layer_factory = _single_qubit_gates_arg_to_factory(
        single_qubit_gates=single_qubit_gates, qubits=qubits, prng=prng
    )

    for i in range(depth):
        single_qubit_layer = single_qubit_layer_factory.new_layer(previous_single_qubit_layer)
        circuit += single_qubit_layer

        two_qubit_layer = _two_qubit_layer(
            coupled_qubit_pairs, two_qubit_op_factory, pattern[i % len(pattern)], prng
        )
        circuit += two_qubit_layer
        previous_single_qubit_layer = single_qubit_layer

    if add_final_single_qubit_layer:
        circuit += single_qubit_layer_factory.new_layer(previous_single_qubit_layer)

    return circuit


def _coupled_qubit_pairs(
    qubits: List['cirq.GridQubit'],
) -> List[GridQubitPairT]:
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
    def __init__(
        self,
        qubits: Sequence['cirq.Qid'],
        single_qubit_gates: Sequence['cirq.Gate'],
        prng: 'np.random.RandomState',
    ) -> None:
        self.qubits = qubits
        self.single_qubit_gates = single_qubit_gates
        self.prng = prng

    def new_layer(self, previous_single_qubit_layer: 'cirq.Moment') -> 'cirq.Moment':
        def random_gate(qubit: 'cirq.Qid') -> 'cirq.Gate':
            excluded_op = previous_single_qubit_layer.operation_at(qubit)
            excluded_gate = excluded_op.gate if excluded_op is not None else None
            g = self.single_qubit_gates[self.prng.randint(0, len(self.single_qubit_gates))]
            while g is excluded_gate:
                g = self.single_qubit_gates[self.prng.randint(0, len(self.single_qubit_gates))]
            return g

        return circuits.Moment(random_gate(q).on(q) for q in self.qubits)


class _FixedSingleQubitLayerFactory:
    def __init__(self, fixed_single_qubit_layer: Dict['cirq.Qid', 'cirq.Gate']) -> None:
        self.fixed_single_qubit_layer = fixed_single_qubit_layer

    def new_layer(self, previous_single_qubit_layer: 'cirq.Moment') -> 'cirq.Moment':
        return circuits.Moment(v.on(q) for q, v in self.fixed_single_qubit_layer.items())


_SingleQubitLayerFactory = Union[_FixedSingleQubitLayerFactory, _RandomSingleQubitLayerFactory]


def _single_qubit_gates_arg_to_factory(
    single_qubit_gates: Sequence['cirq.Gate'],
    qubits: Sequence['cirq.Qid'],
    prng: 'np.random.RandomState',
) -> _SingleQubitLayerFactory:
    """Parse the `single_qubit_gates` argument for circuit generation functions.

    If only one single qubit gate is provided, it will be used everywhere.
    Otherwise, we use the factory that excludes operations that were used
    in the previous layer. This check is done by gate identity, not equality.
    """
    if len(set(single_qubit_gates)) == 1:
        return _FixedSingleQubitLayerFactory({q: single_qubit_gates[0] for q in qubits})

    return _RandomSingleQubitLayerFactory(qubits, single_qubit_gates, prng)


def _two_qubit_layer(
    coupled_qubit_pairs: List[GridQubitPairT],
    two_qubit_op_factory: Callable[
        ['cirq.GridQubit', 'cirq.GridQubit', 'np.random.RandomState'], 'cirq.OP_TREE'
    ],
    layer: GridInteractionLayer,
    prng: 'np.random.RandomState',
) -> 'cirq.OP_TREE':
    for a, b in coupled_qubit_pairs:
        if (a, b) in layer:
            yield two_qubit_op_factory(a, b, prng)
