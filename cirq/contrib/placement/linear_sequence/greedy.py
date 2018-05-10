# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import collections

from typing import Dict, List, Optional, Set
from cirq.contrib.placement.linear_sequence.chip import \
    chip_as_adjacency_list, yx_cmp
from cirq.google import XmonDevice, XmonQubit


class GreedySequenceSearch(object):
    """Base class for greedy search heuristics.

    Specialized greedy heuristics should implement abstrace _sequence_search
    method.
    """

    def __init__(self, device: XmonDevice, start: XmonQubit) -> None:
        """Greedy sequence search constructor.

        Args:
          device: Chip description.
          start: Starting qubit.

        Raises:
          ValueError: When start qubit is not part of a chip.
        """
        if start not in device.qubits:
            raise ValueError('Starting qubit must be a qubit on the chip')

        self._c = device.qubits
        self._c_adj = chip_as_adjacency_list(device)
        self._start = start
        self._sequence = None  # type: Optional[List[XmonQubit]]

    def get_or_search(self) -> List[XmonQubit]:
        """Starts the search or gives previously calculated sequence.

        Returns:
          The linear qubit sequence found.
        """
        if not self._sequence:
            self._sequence = self._find_sequence()
        return self._sequence

    @abc.abstractmethod
    def _choose_next_qubit(self, qubit: XmonQubit,
                           used: Set[XmonQubit]) -> Optional[XmonQubit]:
        """Selects next qubit on the linear sequence.

        Args:
          qubit: Last qubit which is already present on the linear sequence of
                 qubits.
          used: Set of forbidden qubits which can not be used.

        Returns: Next qubit to be appended to the linear sequence, chosen
          according to the greedy heursitic method. The returned qubit will be
          the one passed to the next invocation of this method. Returns None if
          no more qubits are available and search should stop.
        """

    def _find_sequence(self) -> List[XmonQubit]:
        """Looks for a sequence starting at a given qubit.

        Search is issued twice from the starting qubit, so that longest possible
        sequence is found. Starting qubit might not be the first qubit on the
        returned sequence.

        Returns:
          The longest sequence found by this method.
        """
        # Run the first pass and drop starting qubit from the found sequence.
        tail = self._sequence_search(self._start, [])
        tail.pop(0)

        # Run the second pass and keep the starting qubit.
        head = self._sequence_search(self._start, tail)
        head.reverse()

        return self._expand_sequence(head + tail)

    def _sequence_search(self, start: XmonQubit,
                         current: List[XmonQubit]) -> List[XmonQubit]:
        """Search for the continuous linear sequence from the given qubit.

        This method is called twice for the same starting qubit, so that
        sequences that begin and end on this qubit are searched for.

        Args:
          start: The first qubit, where search should be trigerred from.
          current: Previously found linear sequence, which qubits are forbidden
                   to use during the search.

        Returns:
          Continuous linear sequence that begins with the starting qubit and
          does not contain any qubits from the current list.
        """
        used = set(current)
        seq = []
        n = start  # type: Optional[XmonQubit]
        while n is not None:
            # Append qubit n to the sequence and mark it is as visited.
            seq.append(n)
            used.add(n)
            # Advance search to the next qubit.
            n = self._choose_next_qubit(n, used)
        return seq

    def _expand_sequence(self, seq: List[XmonQubit]) -> List[XmonQubit]:
        """Tries to expand given sequence with more qubits.

        Args:
          seq: Linear sequence of qubits.

        Returns:
          New continuous linear sequence which contains all the qubits from seq
          and possibly new qubits inserted in between.
        """
        i = 1
        while i < len(seq):
            path = self._find_path_between(seq[i - 1], seq[i], set(seq))
            if path:
                seq = seq[:i] + path + seq[i:]
            else:
                i += 1
        return seq

    def _find_path_between(self, p: XmonQubit, q: XmonQubit,
                           used: Set[XmonQubit]) -> Optional[List[XmonQubit]]:
        """Searches for continuous sequence between two qubits.

        This method runs two BFS algorithms in palarel (alternating variable s
        in each iteration); the first one starting from qubit p, and the second
        one starting from qubit q. If at some point a qubit reachable from p is
        found to be on the set of qubits already reached from q (or vice versa),
        the search is stopped and new path returned.

        Args:
          p: The first qubit, start of the sequence.
          q: The second qubit, end of the sequence.
          used: Set of forbidden qubits which cannot appear on the sequence.

        Returns:
          Continues sequence of qubits with new path between p and q, or None if
          no path was found.
        """

        def assemble_path(n: XmonQubit, parent: Dict[XmonQubit, XmonQubit]):
            path = [n]
            while n in parent:
                n = parent[n]
                path.append(n)
            return path

        other = {p: q, q: p}
        parents = {p: dict(), q: dict()}  \
            # type: Dict[XmonQubit, Dict[XmonQubit, XmonQubit]]
        visited = {p: set(), q: set()}  # type: Dict[XmonQubit, Set[XmonQubit]]

        queue = collections.deque([(p, p), (q, q)])

        # Run two BFSs simulteanously.
        while queue:
            n, s = queue.popleft()
            for n_adj in self._c_adj[n]:
                if n_adj in visited[other[s]]:
                    # Connection has been found, construct the path and return.
                    path_s = assemble_path(n, parents[s])[-2::-1]
                    path_other = assemble_path(n_adj, parents[other[s]])[:-1]
                    path = path_s + path_other
                    if s == q:
                        path.reverse()
                    return path
                elif n_adj not in used and n_adj not in visited[s]:
                    # Append n_adj to the end of queue of qubit s.
                    queue.append((n_adj, s))
                    visited[s].add(n_adj)
                    parents[s][n_adj] = n

        return None


class MinimalConnectivityGreedySequenceSearch(GreedySequenceSearch):
    """Minimal qubit connectivity greedy heuristic for linear sequence.

    Traverses the grid by choosing the qubit which has the least number of still
    available neighbours in each step.
    """

    def __init__(self, c: XmonDevice, start: XmonQubit) -> None:
        GreedySequenceSearch.__init__(self, c, start)

    def _choose_next_qubit(self, qubit: XmonQubit,
                           used: Set[XmonQubit]) -> Optional[XmonQubit]:
        best = None
        best_size = None
        for m in self._c_adj[qubit]:
            if m not in used:
                connected = [k for k in self._c_adj[m] if k not in used]
                if best is None or best_size > len(connected):
                    best = m
                best_size = len(connected)
        return best


class LargestAreaGreedySequenceSearch(GreedySequenceSearch):
    """Largest area greedy heuristic for linear sequence.

    Traverses the grid by choosing the qubit which is connected with the largest
    part of the chip, when this qubit is added to the sequence.
    """

    def __init__(self, c: XmonDevice, start: XmonQubit) -> None:
        GreedySequenceSearch.__init__(self, c, start)

    def _choose_next_qubit(self, qubit: XmonQubit,
                           used: Set[XmonQubit]) -> Optional[XmonQubit]:
        analyzed = set()  # type: Set[XmonQubit]
        best = None
        best_size = None
        for m in self._c_adj[qubit]:
            if m not in used and m not in analyzed:
                reachable = self._collect_unused(m, used)
                analyzed.update(reachable)

                # Update the best choice only if it is not yet set or if number
                # of qubits that could be reached by qubit m is larger than the
                # previous choice.
                if best is None or best_size < len(reachable):
                    best = m
                    best_size = len(reachable)

        return best

    def _collect_unused(self, start: XmonQubit,
                        used: Set[XmonQubit]) -> Set[XmonQubit]:
        """Lists all the qubits that are reachable from given qubit.

        Args:
          start: The first qubit for which connectivity should be calculated.
                 Might be a member of used set.
          used: Already used qubits, which cannot be used during the collection.

        Returns:
          Set of qubits that are reachable from starting qubit without
          traversing any of the used qubits.
        """

        def collect(n: XmonQubit, visited: Set[XmonQubit]):
            visited.add(n)
            for m in self._c_adj[n]:
                if m not in used and m not in visited:
                    collect(m, visited)

        visited = set()  # type: Set[XmonQubit]
        collect(start, visited)
        return visited


def greedy_sequence(device: XmonDevice,
                    method_opts: dict = None) -> List[List[XmonQubit]]:
    """Greedy search for linear sequence of qubits on a chip.

    Args:
      c: Chip description.
      method_opts: Dictionary with heuristic configuration, unused.

    Returns:
      List of linear sequences found on the chip.
    """
    del method_opts

    def lower_left():
        cand = None
        for n in device.qubits:
            if cand is None or yx_cmp(n, cand) < 0:
                cand = n
        return cand

    start = lower_left()

    greedy_search = {
        'minimal_connectivity':
            MinimalConnectivityGreedySequenceSearch(device, start),
        'largest_area':
            LargestAreaGreedySequenceSearch(device, start)
    }

    sequence = None
    for method in greedy_search:
        candidate = greedy_search[method].get_or_search()
        if sequence is None or len(sequence) < len(candidate):
            sequence = candidate

    return [sequence] if sequence else []
