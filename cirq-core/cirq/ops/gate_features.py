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

"""Marker classes for indicating which additional features gates support.

For example: some gates are reversible, some have known matrices, etc.
"""

import abc
import itertools
from typing import Iterable, Iterator, List, Protocol, Union

from cirq.ops import raw_types


class InterchangeableQubitsGate(metaclass=abc.ABCMeta):
    """Indicates operations should be equal under some qubit permutations."""

    def qubit_index_to_equivalence_group_key(self, index: int) -> int:
        """Returns a key that differs between non-interchangeable qubits."""
        return 0


class QidTree(Protocol):
    """The recursive type consumed by circuit builder methods.

    An OpTree is a type protocol, satisfied by anything that can be recursively
    flattened into Operations. We also define the Union type OP_TREE which
    can be an OpTree or just a single Operation.

    For example:
    - An Operation is an OP_TREE all by itself.
    - A list of operations is an OP_TREE.
    - A list of tuples of operations is an OP_TREE.
    - A list with a mix of operations and lists of operations is an OP_TREE.
    - A generator yielding operations is an OP_TREE.

    Note: once mypy supports recursive types this could be defined as an alias:

    OP_TREE = Union[Operation, Iterable['OP_TREE']]

    See: https://github.com/python/mypy/issues/731
    """

    def __iter__(self) -> Iterator[Union[raw_types.Qid, 'QidTree']]:
        pass


QID_TREE = Union[raw_types.Qid, QidTree]


def _qid_tree_flatten(qids: QID_TREE) -> Iterator[raw_types.Qid]:
    if isinstance(qids, raw_types.Qid):
        yield qids
    elif isinstance(qids, Iterable) and not isinstance(qids, str):
        for q in qids:
            yield from _qid_tree_flatten(q)
    else:
        raise ValueError(f'Expected Qids, got {type(qids)}: {qids}')



def _qid_tree_flatten_partial(qids: QID_TREE) -> Iterator[Iterable[raw_types.Qid]]:
    if isinstance(qids, (raw_types.Qid, str)) or not isinstance(qids, Iterable):
        raise ValueError(f'Expected Iterable[Qid], got {type(qids)}: {qids}')
    qids = list(qids)
    if not qids:
        return
    if all(isinstance(q, raw_types.Qid) for q in qids):
        yield qids
    else:
        for q in qids:
            yield from _qid_tree_flatten_partial(q)


class SupportsOnEachGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate that can be applied to exactly one qubit."""

    def on_each(self, *targets: QID_TREE) -> List[raw_types.Operation]:
        """Returns a list of operations applying the gate to all targets.

        Args:
            *targets: The qubits to apply this gate to.

        Returns:
            Operations applying this gate to the target qubits.

        Raises:
            ValueError if targets are not instances of Qid or List[Qid].
            ValueError if the gate operates on two or more Qids.
        """
        if self._num_qubits_() == 1:
            return [self.on(target) for target in _qid_tree_flatten(targets)]
        else:
            return [self.on(*target) for target in _qid_tree_flatten_partial(targets)]


class SingleQubitGate(SupportsOnEachGate, metaclass=abc.ABCMeta):
    """A gate that must be applied to exactly one qubit."""

    def _num_qubits_(self) -> int:
        return 1


class TwoQubitGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate that must be applied to exactly two qubits."""

    def _num_qubits_(self) -> int:
        return 2


class ThreeQubitGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate that must be applied to exactly three qubits."""

    def _num_qubits_(self) -> int:
        return 3
