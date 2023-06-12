# Copyright 2023 The Cirq Developers
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

from typing import Iterable, List, Set

import cirq


class GreedyQubitManager(cirq.QubitManager):
    """Greedy allocator that maximizes/minimizes qubit reuse based on a configurable parameter.

    GreedyQubitManager can be configured, using `maximize_reuse` flag, to work in one of two modes:
    - Minimize qubit reuse (maximize_reuse=False): For a fixed width, this mode uses a FIFO (First
             in First out) strategy s.t. next allocated qubit is one which was freed the earliest.
    - Maximize qubit reuse (maximize_reuse=True): For a fixed width, this mode uses a LIFO (Last in
            First out) strategy s.t. the next allocated qubit is one which was freed the latest.

    If the requested qubits are more than the set of free qubits, the qubit manager automatically
    resizes the size of the managed qubit pool and adds new free qubits, that have their last
    freed time to be -infinity.

    For borrowing qubits, the qubit manager simply delegates borrow requests to `self.qalloc`, thus
    always allocating new clean qubits.
    """

    def __init__(self, prefix: str, *, size: int = 0, maximize_reuse: bool = False):
        """Initializes `GreedyQubitManager`

        Args:
            prefix: The prefix to use for naming new clean ancillas allocated by the qubit manager.
                    The i'th allocated qubit is of the type `cirq.NamedQubit(f'{prefix}_{i}')`.
            size: The initial size of the pool of ancilla qubits managed by the qubit manager. The
                    qubit manager can automatically resize itself when the allocation request
                    exceeds the number of available qubits.
            maximize_reuse: Flag to control a FIFO vs LIFO strategy, defaults to False (FIFO).
        """
        self._prefix = prefix
        self._used_qubits: Set[cirq.Qid] = set()
        self._free_qubits: List[cirq.Qid] = []
        self._size = 0
        self.maximize_reuse = maximize_reuse
        self.resize(size)

    def _allocate_qid(self, name: str, dim: int) -> cirq.Qid:
        return cirq.q(name) if dim == 2 else cirq.NamedQid(name, dimension=dim)

    def resize(self, new_size: int, dim: int = 2) -> None:
        if new_size <= self._size:
            return
        new_qubits: List[cirq.Qid] = [
            self._allocate_qid(f'{self._prefix}_{s}', dim) for s in range(self._size, new_size)
        ]
        self._free_qubits = new_qubits + self._free_qubits
        self._size = new_size

    def qalloc(self, n: int, dim: int = 2) -> List[cirq.Qid]:
        if not n:
            return []
        self.resize(self._size + n - len(self._free_qubits), dim=dim)
        ret_qubits = self._free_qubits[-n:] if self.maximize_reuse else self._free_qubits[:n]
        self._free_qubits = self._free_qubits[:-n] if self.maximize_reuse else self._free_qubits[n:]
        self._used_qubits.update(ret_qubits)
        return ret_qubits

    def qfree(self, qubits: Iterable[cirq.Qid]) -> None:
        qs = set(qubits)
        assert self._used_qubits.issuperset(qs), "Only managed qubits currently in-use can be freed"
        self._used_qubits -= qs
        self._free_qubits.extend(qs)

    def qborrow(self, n: int, dim: int = 2) -> List[cirq.Qid]:
        return self.qalloc(n, dim)
