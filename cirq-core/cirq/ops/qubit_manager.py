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

import abc
import dataclasses
from typing import Iterable, List, TYPE_CHECKING, Tuple
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


class QubitManager(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def qalloc(self, n: int, dim: int = 2) -> List['cirq.Qid']:
        """Allocate `n` clean qubits, i.e. qubits guaranteed to be in state |0>."""

    @abc.abstractmethod
    def qborrow(self, n: int, dim: int = 2) -> List['cirq.Qid']:
        """Allocate `n` dirty qubits, i.e. the returned qubits can be in any state."""

    @abc.abstractmethod
    def qfree(self, qubits: Iterable['cirq.Qid']) -> None:
        """Free pre-allocated clean or dirty qubits managed by this qubit manager."""


@dataclasses.dataclass(frozen=True)
class _BaseAncillaQid(raw_types.Qid):
    id: int
    dim: int = 2
    prefix: str = ''

    def _comparison_key(self) -> Tuple[str, int]:
        return self.prefix, self.id

    @property
    def dimension(self) -> int:
        return self.dim

    def with_dimension(self, dimension: int) -> '_BaseAncillaQid':
        return dataclasses.replace(self, dim=dimension)

    def __repr__(self) -> str:
        dim_str = f', dim={self.dim}' if self.dim != 2 else ''
        prefix_str = f', prefix={self.prefix!r}' if self.prefix != '' else ''
        return f"cirq.ops.{type(self).__name__}({self.id}{dim_str}{prefix_str})"


class CleanQubit(_BaseAncillaQid):
    """An internal qid type that represents a clean ancilla allocation."""

    def __str__(self) -> str:
        dim_str = f' (d={self.dimension})' if self.dim != 2 else ''
        return f"{self.prefix}_c({self.id}){dim_str}"


class BorrowableQubit(_BaseAncillaQid):
    """An internal qid type that represents a dirty ancilla allocation."""

    def __str__(self) -> str:
        dim_str = f' (d={self.dimension})' if self.dim != 2 else ''
        return f"{self.prefix}_b({self.id}){dim_str}"


class SimpleQubitManager(QubitManager):
    """Allocates a new `CleanQubit`/`BorrowableQubit` for every `qalloc`/`qborrow` request."""

    def __init__(self, prefix: str = ''):
        self._clean_id = 0
        self._borrow_id = 0
        self._prefix = prefix

    def qalloc(self, n: int, dim: int = 2) -> List['cirq.Qid']:
        self._clean_id += n
        return [CleanQubit(i, dim, self._prefix) for i in range(self._clean_id - n, self._clean_id)]

    def qborrow(self, n: int, dim: int = 2) -> List['cirq.Qid']:
        self._borrow_id = self._borrow_id + n
        return [
            BorrowableQubit(i, dim, self._prefix)
            for i in range(self._borrow_id - n, self._borrow_id)
        ]

    def qfree(self, qubits: Iterable['cirq.Qid']) -> None:
        for q in qubits:
            good = isinstance(q, CleanQubit) and q.id < self._clean_id
            good |= isinstance(q, BorrowableQubit) and q.id < self._borrow_id
            if not good:
                raise ValueError(f"{q} was not allocated by {self}")
