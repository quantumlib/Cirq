from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Iterable,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import itertools
import operator
import networkx

from cirq import ops, devices
from cirq.circuits import circuit
from cirq.ops.moment import Moment

if TYPE_CHECKING:
    import cirq


def _disjoint_qubits(op1: Tuple[int, 'cirq.Operation'], op2: Tuple[int, 'cirq.Operation']) -> bool:
    """Returns true only if the operations have qubits in common."""
    return not set(op1[1].qubits) & set(op2[1].qubits)


class SparseCircuitDag(networkx.DiGraph):
    """A representation of a Circuit as a directed acyclic graph.

    Nodes of the graph are instances of Tuple[operation, moment_index] for operations in a circuit.

    Edges of the graph are tuples of nodes.  Each edge specifies a required
    application order between two operations.  The first must be applied before
    the second.

    Unlike CircuitDag, this graph is not maximalist (the given operation is directly connected
    only to its immediate predecessors).
    """

    disjoint_qubits = staticmethod(_disjoint_qubits)

    def __init__(
        self,
        can_reorder: Callable[
            [Tuple[int, 'cirq.Operation'], Tuple[int, 'cirq.Operation']], bool
        ] = _disjoint_qubits,
        incoming_graph_data: Any = None,
        device: devices.Device = devices.UNCONSTRAINED_DEVICE,
    ) -> None:
        """Initializes a CircuitDag.

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
        circuit_ops_by_moments = (
            (index, op) for index, moment in enumerate(circuit.moments) for op in moment.operations
        )
        return SparseCircuitDag.from_ops_by_moments(
            circuit_ops_by_moments, can_reorder=can_reorder, device=circuit.device
        )

    @staticmethod
    def from_ops_by_moments(
        operations_by_moment: Iterable[Tuple[int, 'cirq.Operation']],
        can_reorder: Callable[
            [Tuple[int, 'cirq.Operation'], Tuple[int, 'cirq.Operation']], bool
        ] = _disjoint_qubits,
        device: devices.Device = devices.UNCONSTRAINED_DEVICE,
    ) -> 'SparseCircuitDag':
        dag = SparseCircuitDag(can_reorder=can_reorder, device=device)
        next_frontier = set()
        cur_index = 0
        all_qubits: Set['cirq.Qid'] = set()
        frontier: Set[Tuple[int, 'cirq.Operation']] = set()
        for index, op in operations_by_moment:
            if index != cur_index:
                for prev_node in frontier:
                    if not set(prev_node[1].qubits) <= all_qubits:
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
            yield SparseCircuitDag.from_ops_by_moments(
                sorted(c, key=operator.itemgetter(0)),
                can_reorder=self.can_reorder,
                device=self.device,
            )
