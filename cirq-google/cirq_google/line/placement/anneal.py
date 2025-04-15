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

from typing import Any, Callable, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

import cirq
from cirq_google.line.placement import optimization, place_strategy
from cirq_google.line.placement.chip import above, chip_as_adjacency_list, EDGE, right_of
from cirq_google.line.placement.sequence import GridQubitLineTuple, LineSequence

if TYPE_CHECKING:
    import cirq_google

_STATE = Tuple[List[List[cirq.GridQubit]], Set[EDGE]]


class AnnealSequenceSearch:
    """Simulated annealing search heuristic."""

    def __init__(self, device: 'cirq_google.GridDevice', seed=None) -> None:
        """Greedy sequence search constructor.

        Args:
          device: Chip description.
          seed: Optional seed value for random number generator.
        """
        self._c = device.metadata.qubit_set
        self._c_adj = chip_as_adjacency_list(device)
        self._rand = np.random.RandomState(seed)

    def search(
        self,
        trace_func: Optional[
            Callable[[List[LineSequence], float, float, float, bool], None]
        ] = None,
    ) -> List[LineSequence]:
        """Issues new linear sequence search.

        Each call to this method starts new search.

        Args:
          trace_func: Optional callable which will be called for each simulated
            annealing step with arguments: solution candidate (list of linear
            sequences on the chip), current temperature (float), candidate cost
            (float), probability of accepting candidate (float), and acceptance
            decision (boolean).

        Returns:
          List of linear sequences on the chip found by this method.
        """

        def search_trace(
            state: _STATE, temp: float, cost: float, probability: float, accepted: bool
        ):
            if trace_func:
                trace_seqs, _ = state
                trace_func(trace_seqs, temp, cost, probability, accepted)

        seqs, _ = optimization.anneal_minimize(
            self._create_initial_solution(),
            self._quadratic_sum_cost,
            self._force_edges_active_move,
            self._rand.random_sample,
            trace_func=search_trace,
        )
        return seqs

    def _quadratic_sum_cost(self, state: _STATE) -> float:
        """Cost function that sums squares of lengths of sequences.

        Args:
          state: Search state, not mutated.

        Returns:
          Cost which is minus the normalized quadratic sum of each linear
          sequence section in the state. This promotes single, long linear
          sequence solutions and converges to number -1. The solution with a
          lowest cost consists of every node being a single sequence and is
          always less than 0.
        """
        cost = 0.0
        total_len = float(len(self._c))
        seqs, _ = state
        for seq in seqs:
            cost += (len(seq) / total_len) ** 2
        return -cost

    def _force_edges_active_move(self, state: _STATE) -> _STATE:
        """Move function which repeats _force_edge_active_move a few times.

        Args:
          state: Search state, not mutated.

        Returns:
          New search state which consists of incremental changes of the
          original state.
        """
        for _ in range(self._rand.randint(1, 4)):
            state = self._force_edge_active_move(state)
        return state

    def _force_edge_active_move(self, state: _STATE) -> _STATE:
        """Move which forces a random edge to appear on some sequence.

        This move chooses random edge from the edges which do not belong to any
        sequence and modifies state in such a way, that this chosen edge
        appears on some sequence of the search state.

        Args:
          state: Search state, not mutated.

        Returns:
          New search state with one of the unused edges appearing in some
          sequence.
        """
        seqs, edges = state
        unused_edges = edges.copy()

        # List edges which do not belong to any linear sequence.
        for seq in seqs:
            for i in range(1, len(seq)):
                unused_edges.remove(self._normalize_edge((seq[i - 1], seq[i])))

        edge = self._choose_random_edge(unused_edges)
        if not edge:
            return seqs, edges

        return (self._force_edge_active(seqs, edge, lambda: bool(self._rand.randint(2))), edges)

    def _force_edge_active(
        self, seqs: List[List[cirq.GridQubit]], edge: EDGE, sample_bool: Callable[[], bool]
    ) -> List[List[cirq.GridQubit]]:
        """Move which forces given edge to appear on some sequence.

        Args:
          seqs: List of linear sequences covering chip.
          edge: Edge to be activated.
          sample_bool: Callable returning random bool.

        Returns:
          New list of linear sequences with given edge on some of the
          sequences.
        """

        n0, n1 = edge

        # Make a copy of original sequences.
        seqs = list(seqs)

        # Localize edge nodes within current solution.
        i0, j0 = index_2d(seqs, n0)
        i1, j1 = index_2d(seqs, n1)
        s0 = seqs[i0]
        s1 = seqs[i1]

        # Handle case when nodes belong to different linear sequences,
        # separately from the case where they belong to a single linear
        # sequence.
        if i0 != i1:

            # Split s0 and s1 in two parts: s0 in parts before n0, and after n0
            # (without n0); s1 in parts before n1, and after n1 (without n1).
            part = [s0[:j0], s0[j0 + 1 :]], [s1[:j1], s1[j1 + 1 :]]

            # Remove both sequences from original list.
            del seqs[max(i0, i1)]
            del seqs[min(i0, i1)]

            # Choose part of s0 which will be attached to n0, and make sure it
            # can be attached in the end.
            c0 = 0 if not part[0][1] else 1 if not part[0][0] else sample_bool()
            if c0:
                part[0][c0].reverse()

            # Choose part of s1 which will be attached to n1, and make sure it
            # can be attached in the beginning.
            c1 = 0 if not part[1][1] else 1 if not part[1][0] else sample_bool()
            if not c1:
                part[1][c1].reverse()

            # Append newly formed sequence from the chosen parts and new edge.
            seqs.append(part[0][c0] + [n0, n1] + part[1][c1])

            # Append the left-overs to the solution, if they exist.
            other = [1, 0]
            seqs.append(part[0][other[c0]])
            seqs.append(part[1][other[c1]])
        else:
            # Swap nodes so that n0 always precedes n1 on sequence.
            if j0 > j1:
                j0, j1 = j1, j0
                n0, n1 = n1, n0

            # Split sequence in three parts, without nodes n0 an n1 present:
            # head might end with n0, inner might begin with n0 and end with
            # n1, tail might begin with n1.
            head = s0[:j0]
            inner = s0[j0 + 1 : j1]
            tail = s0[j1 + 1 :]

            # Remove original sequence from sequences list.
            del seqs[i0]

            # Either append edge to inner section, or attach it between head
            # and tail.
            if sample_bool():
                # Append edge either before or after inner section.
                if sample_bool():
                    seqs.append(inner + [n1, n0] + head[::-1])
                    seqs.append(tail)
                else:
                    seqs.append(tail[::-1] + [n1, n0] + inner)
                    seqs.append(head)
            else:
                # Form a new sequence from head, tail, and new edge.
                seqs.append(head + [n0, n1] + tail)
                seqs.append(inner)

        return [e for e in seqs if e]

    def _create_initial_solution(self) -> _STATE:
        """Creates initial solution based on the chip description.

        Initial solution is constructed in a greedy way.

        Returns:
          Valid search state.
        """

        def extract_sequences() -> List[List[cirq.GridQubit]]:
            """Creates list of sequences for initial state.

            Returns:
              List of lists of sequences constructed on the chip.
            """
            seqs = []
            prev = None
            seq = []
            for node in self._c:
                if prev is None:
                    seq = [node]
                else:
                    if node in self._c_adj[prev]:
                        # Expand current sequence.
                        seq.append(node)
                    else:
                        # Create new sequence, there is no connection between
                        # nodes.
                        seqs.append(seq)
                        seq = [node]
                prev = node
            if seq:
                seqs.append(seq)
            return seqs

        def assemble_edges() -> Set[EDGE]:
            """Creates list of edges for initial state.

            Returns:
              List of all possible edges.
            """
            nodes_set = set(self._c)
            edges = set()
            for n in self._c:
                if above(n) in nodes_set:
                    edges.add(self._normalize_edge((n, above(n))))
                if right_of(n) in nodes_set:
                    edges.add(self._normalize_edge((n, right_of(n))))
            return edges

        return extract_sequences(), assemble_edges()

    def _normalize_edge(self, edge: EDGE) -> EDGE:
        """Gives unique representative of the edge.

        Two edges are equivalent if they form an edge between the same nodes.
        This method returns representative of this edge which can be compared
        using equality operator later.

        Args:
          edge: Edge to normalize.

        Returns:
          Normalized edge with lexicographically lower node on the first
          position.
        """

        def lower(n: cirq.GridQubit, m: cirq.GridQubit) -> bool:
            return n.row < m.row or (n.row == m.row and n.col < m.col)

        n1, n2 = edge
        return (n1, n2) if lower(n1, n2) else (n2, n1)

    def _choose_random_edge(self, edges: Set[EDGE]) -> Optional[EDGE]:
        """Picks random edge from the set of edges.

        Args:
          edges: Set of edges to pick from.

        Returns:
          Random edge from the supplied set, or None for empty set.
        """
        if edges:
            index = self._rand.randint(len(edges))
            for e in edges:
                if not index:
                    return e
                index -= 1
        return None


class AnnealSequenceSearchStrategy(place_strategy.LinePlacementStrategy):
    """Linearized sequence search using simulated annealing method.

    TODO: This line search strategy is still work in progress and requires
    efficiency improvements.
    Github issue: https://github.com/quantumlib/Cirq/issues/2217
    """

    def __init__(
        self,
        trace_func: Optional[
            Callable[[List[LineSequence], float, float, float, bool], None]
        ] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Linearized sequence search using simulated annealing method.

        Args:
            trace_func: Optional callable which will be called for each
                        simulated annealing step with arguments: solution
                        candidate (list of linear sequences on the chip),
                        current temperature (float), candidate cost (float),
                        probability of accepting candidate (float), and
                        acceptance decision (boolean).
            seed: Optional seed value for random number generator.

        Returns:
            List of linear sequences on the chip found by simulated annealing
            method.
        """
        self.trace_func = trace_func
        self.seed = seed

    def place_line(self, device: 'cirq_google.GridDevice', length: int) -> GridQubitLineTuple:
        """Runs line sequence search.

        Args:
            device: Chip description.
            length: Required line length.

        Returns:
            List of linear sequences on the chip found by simulated annealing
            method.
        """
        seqs = AnnealSequenceSearch(device, self.seed).search(self.trace_func)
        return GridQubitLineTuple.best_of(seqs, length)


def index_2d(seqs: List[List[Any]], target: Any) -> Tuple[int, int]:
    """Finds the first index of a target item within a list of lists.

    Args:
        seqs: The list of lists to search.
        target: The item to find.

    Raises:
        ValueError: Item is not present.
    """
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            if seqs[i][j] == target:
                return i, j
    raise ValueError('Item not present.')
