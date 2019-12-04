from typing import (Callable, Container, Iterable, List, Sequence, Set,
                    TYPE_CHECKING, Tuple, cast)

import abc

from cirq import circuits, devices, google, ops, value
from cirq._doc import document

if TYPE_CHECKING:
    from typing import Dict
    import numpy as np
    import cirq

# A set of 2-qubit interactions that can be performed simultaneously on a grid.
TwoQubitInteractionLayer = Container[
    Tuple[devices.GridQubit, devices.GridQubit]]


class GmonLayer(TwoQubitInteractionLayer):
    """A Gmon-style layer of 2-qubit gates where neighbors don't need to idle.

    These layers contain gates involving all qubits in the lattice. They come
    in two variants, unstaggered:

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
    There are a total of 4 staggered and 4 unstaggered variants, and a pattern
    will cycle through one of these groups of 4 to perform all 2-qubit gates.

    The 2x2 unit cells for the unstaggered and staggered versions of this layer
    are, respectively:

    *-*
    *-*

    and

    *-*
    * *-

    with left/top qubits at (0, 0) and (1, 0) in the unstaggered case, or
    (0, 0) and (1, 1) in the staggered case. Other variants have the same unit
    cells after transposing and offsetting.
    """

    def __init__(self,
                 col_offset: int = 0,
                 vertical: bool = False,
                 stagger: bool = False) -> None:
        """
        Args:
            col_offset: Number of columns by which to shift the basic lattice.
            vertical: Whether gates should be oriented vertically rather than
                horizontally.
            stagger: Whether to stagger gates in neighboring rows.
        """
        self.col_offset = col_offset
        self.vertical = vertical
        self.stagger = stagger

    def __contains__(self,
                     pair: Tuple[devices.GridQubit, devices.GridQubit]) -> bool:
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


GMON_HARD_PATTERN = (
    GmonLayer(col_offset=0, vertical=True, stagger=True),  # A
    GmonLayer(col_offset=1, vertical=True, stagger=True),  # B
    GmonLayer(col_offset=1, vertical=False, stagger=True),  # C
    GmonLayer(col_offset=0, vertical=False, stagger=True),  # D
    GmonLayer(col_offset=1, vertical=False, stagger=True),  # C
    GmonLayer(col_offset=0, vertical=False, stagger=True),  # D
    GmonLayer(col_offset=0, vertical=True, stagger=True),  # A
    GmonLayer(col_offset=1, vertical=True, stagger=True),  # B
)
document(
    GMON_HARD_PATTERN, """A pattern of two-qubit gates that is hard to simulate.

    This pattern of gates was used in the paper
    https://www.nature.com/articles/s41586-019-1666-5
    to demonstrate quantum supremacy.
    """)

GMON_EASY_PATTERN = (
    GmonLayer(col_offset=0, vertical=False, stagger=False),  # E
    GmonLayer(col_offset=1, vertical=False, stagger=False),  # F
    GmonLayer(col_offset=0, vertical=True, stagger=False),  # G
    GmonLayer(col_offset=1, vertical=True, stagger=False),  # H
)
document(
    GMON_EASY_PATTERN, """A pattern of two-qubit gates that is easy to simulate.

    This pattern of gates was used in the paper
    https://www.nature.com/articles/s41586-019-1666-5
    to evaluate the performance of a quantum computer.
    """)


def random_quantum_circuit(
        qubits: Iterable['cirq.GridQubit'],
        depth: int,
        two_qubit_op_factory: Callable[[
            'cirq.GridQubit', 'cirq.GridQubit', 'np.random.RandomState'
        ], 'cirq.OP_TREE'] = lambda a, b, _: google.SYC(a, b),
        pattern: Sequence[TwoQubitInteractionLayer] = GMON_HARD_PATTERN,
        single_qubit_gates: Sequence['cirq.Gate'] = (ops.X**0.5, ops.Y**0.5,
                                                     ops.PhasedXPowGate(
                                                         phase_exponent=0.25,
                                                         exponent=0.5)),
        seed: value.RANDOM_STATE_LIKE = None,
) -> 'cirq.Circuit':
    """Generate a random quantum circuit.

    This construction is based on the circuits used in the paper
    https://www.nature.com/articles/s41586-019-1666-5.

    The generated circuit consists of a number of layers, this number being
    specified by `depth`. Each layer is actually composed of two sub-layers:
    a layer of single-qubit gates followed by a layer of two-qubit gates.
    The single-qubit gates are chosen randomly from the gates specified by
    `single_qubit_gates`, but with the constraint that no qubit is acted upon
    by the same single-qubit gate in consecutive layers. In the layer of
    two-qubit gates, which pairs of qubits undergo interaction is determined
    by `pattern`, which is a sequence of two-qubit interaction sets. The
    set of interactions a two-qubit layer rotates through this sequence.
    The two-qubit operations themselves are generated by the call
    `two_qubit_op_factory(a, b, prng)`, where `a` and `b` are the qubits and
    `prng` is the pseudorandom number generator.

    At the end of the circuit, an additional layer of single-qubit gates is
    appended, subject to the same constraint regarding consecutive layers
    described above.
    """
    prng = value.parse_random_state(seed)
    qubits = list(qubits)
    coupled_qubit_pairs = _coupled_qubit_pairs(qubits)

    circuit = circuits.Circuit()
    previous_single_qubit_layer = []  # type: List[cirq.GateOperation]
    for i in range(depth):
        single_qubit_layer = _single_qubit_layer(qubits, single_qubit_gates,
                                                 previous_single_qubit_layer,
                                                 prng)
        two_qubit_layer = _two_qubit_layer(coupled_qubit_pairs,
                                           two_qubit_op_factory,
                                           pattern[i % len(pattern)], prng)
        circuit.append(single_qubit_layer,
                       strategy=circuits.InsertStrategy.NEW_THEN_INLINE)
        circuit.append(two_qubit_layer,
                       strategy=circuits.InsertStrategy.EARLIEST)
        previous_single_qubit_layer = single_qubit_layer
    circuit.append(_single_qubit_layer(qubits, single_qubit_gates,
                                       previous_single_qubit_layer, prng),
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


def _single_qubit_layer(
        qubits: List['cirq.GridQubit'],
        single_qubit_gates: Sequence['cirq.Gate'],
        previous_single_qubit_layer: List['cirq.GateOperation'],
        prng: 'np.random.RandomState',
) -> List['cirq.GateOperation']:

    excluded_gates = {}  # type: Dict[cirq.GridQubit, cirq.Gate]
    if previous_single_qubit_layer:
        for op in previous_single_qubit_layer:
            excluded_gates[cast('cirq.GridQubit', op.qubits[0])] = op.gate

    def random_gate(qubit: 'cirq.GridQubit',
                    prng: 'np.random.RandomState') -> 'cirq.Gate':
        excluded_gate = excluded_gates.get(qubit, None)
        allowed_gates = [g for g in single_qubit_gates if g != excluded_gate]
        return allowed_gates[prng.randint(0, len(allowed_gates))]

    return [random_gate(q, prng).on(q) for q in qubits]


def _two_qubit_layer(
        coupled_qubit_pairs: List[Tuple['cirq.GridQubit', 'cirq.GridQubit']],
        two_qubit_op_factory: Callable[[
            'cirq.GridQubit', 'cirq.GridQubit', 'np.random.RandomState'
        ], 'cirq.OP_TREE'], layer: TwoQubitInteractionLayer,
        prng: 'np.random.RandomState') -> 'cirq.OP_TREE':
    for a, b in coupled_qubit_pairs:
        if (a, b) in layer:
            yield two_qubit_op_factory(a, b, prng)
