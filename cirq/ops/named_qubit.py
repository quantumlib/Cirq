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
from typing import Any, Dict, List, TYPE_CHECKING, TypeVar

import abc

from cirq import protocols, ops
from cirq.ops import raw_types


if TYPE_CHECKING:
    import cirq

TSelf = TypeVar("TSelf", bound="_BaseNamedQid")


@functools.total_ordering
class _BaseNamedQid(raw_types.Qid):
    """The base class for `NamedQid` and `NamedQubit`."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._comp_key = _pad_digits(name)

    def _comparison_key(self):
        return self._comp_key

    @property
    def name(self) -> str:
        return self._name

    def with_dimension(self, dimension: int) -> "NamedQid":
        return NamedQid(self._name, dimension)

    @abc.abstractmethod
    def _with_name(self: TSelf, name: str) -> TSelf:
        """
        Returns a qubit with the same type but a different value of `name`
        """


class NamedQid(_BaseNamedQid):
    """A qid identified by name.

    By default, NamedQid has a lexicographic order. However, numbers within
    the name are handled correctly. So, for example, if you print a circuit
    containing `cirq.NamedQid('qid22')` and `cirq.NamedQid('qid3')`, the
    wire for 'qid3' will correctly come before 'qid22'.
    """

    def __init__(self, name: str, dimension: int) -> None:
        super().__init__(name)
        self._dimension = dimension
        self.validate_dimension(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._name

    def _with_name(self, name: str) -> "NamedQid":
        return NamedQid(name, dimension=self.dimension)

    def __repr__(self) -> str:
        return f"cirq.NamedQid({self.name!r}, " f"dimension={self.dimension})"

    def __str__(self) -> str:
        return f"{self.name} (d={self.dimension})"

    @staticmethod
    def range(*args, prefix: str, dimension: int) -> List["NamedQid"]:
        """Returns a range of NamedQids.

        The range returned starts with the prefix, and followed by a qid for
        each number in the range, e.g.:

        NamedQid.range(3, prefix="a") -> ["a1", "a2", "a3]
        NamedQid.range(2, 4, prefix="a") -> ["a2", "a3]

        Args:
            *args: Args to be passed to Python's standard range function.
            prefix: A prefix for constructed NamedQids.

        Returns:
            A list of NamedQids.
            """
        return [
            NamedQid(prefix + str(i), dimension=dimension) for i in range(*args)
        ]

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ["name", "dimension"])


class NamedQubit(_BaseNamedQid):
    """A qubit identified by name.

    By default, NamedQubit has a lexicographic order. However, numbers within
    the name are handled correctly. So, for example, if you print a circuit
    containing `cirq.NamedQubit('qubit22')` and `cirq.NamedQubit('qubit3')`, the
    wire for 'qubit3' will correctly come before 'qubit22'.
    """

    @property
    def dimension(self) -> int:
        return 2

    def _cmp_tuple(self):
        cls = NamedQid if type(self) is NamedQubit else type(self)
        # Must be same as Qid._cmp_tuple but with cls in place of type(self).
        return (cls.__name__, repr(cls), self._comparison_key(), self.dimension)

    def _with_name(self, name: str) -> "NamedQubit":
        return NamedQubit(name)

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"cirq.NamedQubit({self._name!r})"

    @staticmethod
    def range(*args, prefix: str) -> List["NamedQubit"]:
        """Returns a range of NamedQubits.

        The range returned starts with the prefix, and followed by a qubit for
        each number in the range, e.g.:

        NamedQubit.range(3, prefix="a") -> ["a1", "a2", "a3]
        NamedQubit.range(2, 4, prefix="a") -> ["a2", "a3]

        Args:
            *args: Args to be passed to Python's standard range function.
            prefix: A prefix for constructed NamedQubits.

        Returns:
            A list of NamedQubits.
        """
        return [NamedQubit(prefix + str(i)) for i in range(*args)]

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ["name"])


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
            chunk = chunk.rjust(8, "0") + ":" + str(len(chunk))
        chunks.append(chunk)

    for i in range(len(text)):
        on_digits = text[i].isdigit()
        if was_on_digits != on_digits:
            handle_transition_at(i)
            was_on_digits = on_digits
            last_transition = i

    handle_transition_at(len(text))
    return "".join(chunks)

