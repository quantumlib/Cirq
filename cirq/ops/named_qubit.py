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
from typing import Any, Dict

from cirq import protocols
from cirq.ops import raw_types


class NamedQubit(raw_types.Qid):
    """A qubit identified by name.

    By default, NamedQubit has a lexicographic order. However, numbers within
    the name are handled correctly. So, for example, if you print a circuit
    containing `cirq.NamedQubit('qubit22')` and `cirq.NamedQubit('qubit3')`, the
    wire for 'qubit3' will correctly come before 'qubit22'.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._comp_key = _pad_digits(name)

    def _comparison_key(self):
        return self._comp_key

    @property
    def dimension(self) -> int:
        return 2

    def __str__(self) -> str:
        return self._name

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f'cirq.NamedQubit({self._name!r})'

    @staticmethod
    def range(*args, prefix: str):
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
