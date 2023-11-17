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
import functools
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from cirq import protocols
from cirq.ops import raw_types


if TYPE_CHECKING:
    import cirq


@functools.total_ordering
class _BaseNamedQid(raw_types.Qid):
    """The base class for `NamedQid` and `NamedQubit`."""

    _name: str
    _dimension: int
    _comp_key: Optional[str] = None
    _hash: Optional[int] = None

    def __getstate__(self):
        # Don't save hash when pickling; see #3777.
        state = self.__dict__
        if "_hash" in state or "_comp_key" in state:
            state = state.copy()
            if "_hash" in state:
                del state["_hash"]
            if "_comp_key" in state:
                del state["_comp_key"]
        return state

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._name, self._dimension))
        return self._hash

    def __eq__(self, other):
        # Explicitly implemented for performance (vs delegating to Qid).
        if isinstance(other, _BaseNamedQid):
            return self._name == other._name and self._dimension == other._dimension
        return NotImplemented

    def __ne__(self, other):
        # Explicitly implemented for performance (vs delegating to Qid).
        if isinstance(other, _BaseNamedQid):
            return self._name != other._name or self._dimension != other._dimension
        return NotImplemented

    def _comparison_key(self):
        if self._comp_key is None:
            self._comp_key = _pad_digits(self._name)
        return self._comp_key

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension

    def with_dimension(self, dimension: int) -> 'NamedQid':
        return NamedQid(self._name, dimension=dimension)


class NamedQid(_BaseNamedQid):
    """A qid identified by name.

    By default, `NamedQid` has a lexicographic order. However, numbers within
    the name are handled correctly. So, for example, if you print a circuit
    containing `cirq.NamedQid('qid22', dimension=3)` and
    `cirq.NamedQid('qid3', dimension=3)`, the wire for 'qid3' will
    correctly come before 'qid22'.
    """

    def __init__(self, name: str, dimension: int) -> None:
        """Initializes a `NamedQid` with a given name and dimension.

        Args:
            name: The name.
            dimension: The dimension of the qid's Hilbert space, i.e.
                the number of quantum levels.
        """
        self.validate_dimension(dimension)
        self._name = name
        self._dimension = dimension

    def __repr__(self) -> str:
        return f'cirq.NamedQid({self._name!r}, dimension={self._dimension})'

    def __str__(self) -> str:
        return f'{self._name} (d={self._dimension})'

    @staticmethod
    def range(*args, prefix: str, dimension: int) -> List['NamedQid']:
        """Returns a range of ``NamedQid``\\s.

        The range returned starts with the prefix, and followed by a qid for
        each number in the range, e.g.:

            >>> cirq.NamedQid.range(3, prefix='a', dimension=3)
            ... # doctest: +NORMALIZE_WHITESPACE
            [cirq.NamedQid('a0', dimension=3), cirq.NamedQid('a1', dimension=3),
                cirq.NamedQid('a2', dimension=3)]
            >>> cirq.NamedQid.range(2, 4, prefix='a', dimension=3)
            [cirq.NamedQid('a2', dimension=3), cirq.NamedQid('a3', dimension=3)]

        Args:
            *args: Args to be passed to Python's standard range function.
            prefix: A prefix for constructed NamedQids.
            dimension: The dimension of the qid's Hilbert space, i.e.
                the number of quantum levels.
        Returns:
            A list of ``NamedQid``\\s.
        """
        return [NamedQid(f"{prefix}{i}", dimension=dimension) for i in range(*args)]

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['name', 'dimension'])


class NamedQubit(_BaseNamedQid):
    """A qubit identified by name.

    By default, `NamedQubit` has a lexicographic order. However, numbers within
    the name are handled correctly. So, for example, if you print a circuit
    containing `cirq.NamedQubit('qubit22')` and `cirq.NamedQubit('qubit3')`, the
    wire for 'qubit3' will correctly come before 'qubit22'.
    """

    _dimension = 2

    def __init__(self, name: str) -> None:
        """Initializes a `NamedQubit` with a given name.

        Args:
            name: The name.
        """
        self._name = name

    def _cmp_tuple(self):
        cls = NamedQid if type(self) is NamedQubit else type(self)
        # Must be same as Qid._cmp_tuple but with cls in place of type(self).
        return (cls.__name__, repr(cls), self._comparison_key(), self._dimension)

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f'cirq.NamedQubit({self._name!r})'

    @staticmethod
    def range(*args, prefix: str) -> List['NamedQubit']:
        r"""Returns a range of `cirq.NamedQubit`s.

        The range returned starts with the prefix, and followed by a qubit for
        each number in the range, e.g.:

            >>> cirq.NamedQubit.range(3, prefix='a')
            ... # doctest: +NORMALIZE_WHITESPACE
            [cirq.NamedQubit('a0'), cirq.NamedQubit('a1'),
                cirq.NamedQubit('a2')]
            >>> cirq.NamedQubit.range(2, 4, prefix='a')
            [cirq.NamedQubit('a2'), cirq.NamedQubit('a3')]

        Args:
            *args: Args to be passed to Python's standard range function.
            prefix: A prefix for constructed NamedQubits.

        Returns:
            A list of ``NamedQubit``\\s.
        """
        return [NamedQubit(f"{prefix}{i}") for i in range(*args)]

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['name'])


def _pad_digits(text: str) -> str:
    """A str method with hacks to support better lexicographic ordering.

    The output strings are not intended to be human readable.

    The returned string will have digit-runs zero-padded up to at least 8
    digits. That way, instead of 'a10' coming before 'a2', 'a000010' will come
    after 'a000002'.

    Also, the original length of each digit-run is appended after the
    zero-padded run. This is so that 'a0' continues to come before 'a00'.
    """

    was_on_digits = False
    last_transition = 0
    chunks = []

    def handle_transition_at(k):
        chunk = text[last_transition:k]
        if was_on_digits:
            chunk = chunk.rjust(8, '0') + ':' + str(len(chunk))
        chunks.append(chunk)

    for i in range(len(text)):
        on_digits = text[i].isdigit()
        if was_on_digits != on_digits:
            handle_transition_at(i)
            was_on_digits = on_digits
            last_transition = i

    handle_transition_at(len(text))
    return ''.join(chunks)
