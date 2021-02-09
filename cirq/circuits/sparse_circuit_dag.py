from typing import (
    Any,
    Callable,
    Iterator,
    Iterable,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import itertools
import operator
import networkx

from cirq import devices
from cirq.circuits import circuit
from cirq.ops.moment import Moment

if TYPE_CHECKING:
    import cirq


def _disjoint_qubits(
    term1: Tuple[int, 'cirq.Operation'], term2: Tuple[int, 'cirq.Operation']
) -> bool:
    """Returns true if and only if the operations have no qubits in common."""
    _, op1 = term1
    _, op2 = term2
    return not set(op1.qubits) & set(op2.qubits)


class SparseCircuitDag(networkx.DiGraph):
    """A representation of a Circuit as a directed acyclic graph.

    Nodes of the graph are instances of Tuple[operation, moment_index] for operations in a circuit.

    Edges of the graph are tuples of nodes.  Each edge specifies a required
    application order between two operations.  The first must be applied before
    the second.

    Unlike CircuitDag, this graph is not maximalist (the given operation is directly connected
    only to its immediate predecessors).
    """

    def __init__(
        self,
        can_reorder: Callable[
            [Tuple[int, 'cirq.Operation'], Tuple[int, 'cirq.Operation']], bool
        ] = _disjoint_qubits,
        incoming_graph_data: Any = None,
        device: devices.Device = devices.UNCONSTRAINED_DEVICE,
    ) -> None:
        """Initializes a SparseCircuitDag.

        Args:
            can_reorder: A predicate that determines if two operations may be
                reordered.  Graph edges are created for pairs of operations
                where this returns False.
                The default predicate allows reordering only when the operations
                don't share common qubits.
            incoming_graph_data: Data in initialize the graph.  This can be any
                value supported by networkx.DiGraph() e.g. an edge list or
                another graph.
            device: Hardware that the circuit should be able to run on.
        """
        super().__init__(incoming_graph_data)
        self.can_reorder = can_reorder
        self.device = device

    @staticmethod
    def from_circuit(
        circuit: circuit.Circuit,
        can_reorder: Callable[
            [Tuple[int, 'cirq.Operation'], Tuple[int, 'cirq.Operation']], bool
        ] = _disjoint_qubits,
    ) -> 'SparseCircuitDag':
        """Creates instance of SparseCircuitDag class given the circuit."""
        circuit_ops_by_moment_index = (
            (index, op) for index, moment in enumerate(circuit.moments) for op in moment.operations
        )
        return SparseCircuitDag.from_ops_by_moment_index(
            circuit_ops_by_moment_index, can_reorder=can_reorder, device=circuit.device
        )

    @staticmethod
    def from_ops_by_moment_index(
        operations_by_moment_index: Iterable[Tuple[int, 'cirq.Operation']],
        can_reorder: Callable[
            [Tuple[int, 'cirq.Operation'], Tuple[int, 'cirq.Operation']], bool
        ] = _disjoint_qubits,
        device: devices.Device = devices.UNCONSTRAINED_DEVICE,
    ) -> 'SparseCircuitDag':
        """Creates instance of SparseCircuitDag class given the sequence of
        (moment_index, moment_operations) pairs.

        Uses the following algorithm:

        - Maintain a "frontier" that keeps track of most recent operations
        on all qubits before the current moment.
        - Iterate over the (moment_index, operation) pairs.
        - For every operation, add edges to the operations in the frontier
        which have qubits in common.
        - Add every such operation to the next version of the frontier.
        - When the moment index changes (it is expected to never decrease),
        add to the next frontier all the operations from the current frontier
        if they were defined on qubits that didn't participate in the current
        moment.

        Example:
              1   2   3   4
        0: ───────@───H───H───
                  │
        1: ───H───@───────────

        2: ───H───────────────

        translates into the following operations_by_moment sequence:
        (1, cirq.H(cirq.LineQubit(1))), (1, cirq.H(cirq.LineQubit(2))),
        (2, cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1))),
        (3, cirq.H(cirq.LineQubit(3))),
        (4, cirq.H(cirq.LineQubit(4))),
        and the folllowing sequence of frontiers:
        (
        (1, cirq.H(cirq.LineQubit(1))), (1, cirq.H(cirq.LineQubit(2)))
        ),
        (
        (2, cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1))),
        (1, cirq.H(cirq.LineQubit(2)))
        ),
        (
        (3, cirq.H(cirq.LineQubit(0))),
        (2, cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1))),
        (1, cirq.H(cirq.LineQubit(2)))
        )
        The last iteration invoves one pair: (4, cirq.H(cirq.LineQubit(0))),
        which is connected (i.e. has matching qubits) with two of the three
        "nodes" from the last frontier.
        In this case, the circuit has two independent factors.

        Another example:
              1   2   3   4
        0: ───────@───H───H───
                  │
        1: ───H───@───────@───
                          │
        2: ───H───────────@───

        translates into the following operations_by_moment sequence:
        (1, cirq.H(cirq.LineQubit(1))), (1, cirq.H(cirq.LineQubit(2))),
        (2, cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1))),
        (3, cirq.H(cirq.LineQubit(3))),
        (4, cirq.H(cirq.LineQubit(4))),
        and the folllowing sequence of frontiers:
        (
        (1, cirq.H(cirq.LineQubit(1))), (1, cirq.H(cirq.LineQubit(2)))
        ),
        (
        (2, cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1))),
        (1, cirq.H(cirq.LineQubit(2)))
        ),
        (
        (3, cirq.H(cirq.LineQubit(0))),
        (2, cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1))),
        (1, cirq.H(cirq.LineQubit(2)))
        )
        The last iteration invoves two pairs:
        (4, cirq.H(cirq.LineQubit(0))),
        (4, cirq.CZ(cirq.LineQubit(1), cirq.LineQubit(2)))
        of which the first is connected to the first two and the second
        is connected to the second two nodes from the last frontier.
        In this case, the circuit has a single independent factor
        (i.e. it cannot be factorized)

        """
        dag = SparseCircuitDag(can_reorder=can_reorder, device=device)
        cur_index = 0
        # All qubits participating in the operations before the current moment.
        all_qubits: Set['cirq.Qid'] = set()
        # Frontier is keeping track of most recent operations on all qubits
        # which participated in the operations before the current moment.
        # These are the only "nodes" which could be connected to the operatioins
        # present in the current moment.
        frontier: Set[Tuple[int, 'cirq.Operation']] = set()
        # Next iteration of the frontier.
        next_frontier = set()
        # Iterate over (op, moment_num) pairs
        # (the assumption is that the moments are listed in order).
        # Keep track of all qubits which are, directly or indirectly,
        # affected by these operations.
        for index, op in operations_by_moment_index:
            # Assumpiton: next index corresponds to the next moment
            # in the circuit
            if index != cur_index:
                # Expectation: the sequence is ordered by moment indices.
                if index < cur_index:
                    raise ValueError("Moment indices expected to increase.")

                for prev_node in frontier:
                    if any(q not in all_qubits for q in prev_node[1].qubits):
                        next_frontier.add(prev_node)
                frontier = next_frontier
                next_frontier = set()
                all_qubits = set()

            new_node = (index, op)
            dag.add_node(new_node)
            next_frontier.add(new_node)
            all_qubits |= set(op.qubits)
            for prev_node in frontier:
                if not dag.can_reorder(prev_node, new_node):
                    dag.add_edge(prev_node, new_node)
        return dag

    def to_circuit(self) -> circuit.Circuit:
        return circuit.Circuit(
            Moment((op for index, op in ops))
            for i, ops in itertools.groupby(
                sorted(self.nodes, key=operator.itemgetter(0)), operator.itemgetter(0)
            )
        )

    def factorize(self) -> Iterator['cirq.SparseCircuitDag']:
        """Tries to factorize the underlying graph, using connected components algorithm.
        If no factorization is possible, returns a sequence with a single element (itself).
        """
        for c in networkx.weakly_connected_components(self):
            yield SparseCircuitDag.from_ops_by_moment_index(
                sorted(c, key=operator.itemgetter(0)),
                can_reorder=self.can_reorder,
                device=self.device,
            )
